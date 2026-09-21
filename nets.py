"""Network building blocks.

Two networks live here:

  BaselineNet    deterministic conditional mean  mu(x),  stage 1
  DiffusionUNet  EDM denoiser for the residual   r = y - mu,  stage 2

Both are fully convolutional and use GroupNorm, never BatchNorm: they are trained on
96x96 patches but BaselineNet is applied to the full 1024x912 field in one pass, and
BatchNorm statistics would not survive that change of input size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class PixelShuffle2D(nn.Module):
    """Anisotropic pixel shuffle: (B, C*rh*rw, H, W) -> (B, C, H*rh, W*rw).

    torch's nn.PixelShuffle only does a single square factor; the DUACS -> SWOT
    refinement is 8 in latitude and 3 in longitude.
    """

    def __init__(self, rh, rw):
        super().__init__()
        self.rh, self.rw = rh, rw

    def forward(self, x):
        b, c, h, w = x.shape
        rh, rw = self.rh, self.rw
        c_out = c // (rh * rw)
        x = x.view(b, c_out, rh, rw, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(b, c_out, h * rh, w * rw)


def block_repeat(x, rh=C.REFINE_LAT, rw=C.REFINE_LON):
    """Piecewise-constant expansion of the coarse field onto the fine grid.

    This is NOT interpolation -- it is the exact block structure of the dataset
    (sla[i,j] governs ssha[8i:8i+8, 3j:3j+3]), so it stamps no kernel into the
    input, which is what the dataset contract warns against.
    """
    return x.repeat_interleave(rh, dim=-2).repeat_interleave(rw, dim=-1)


def smooth_upsample(x, mode="bicubic"):
    """Interpolate the coarse field onto the fine grid, smoothly.

    Used only as the RESIDUAL BASE, never as a network input.  The contract warns
    against pre-interpolating the input because a network can learn to invert the
    interpolation kernel and score well without learning anything; that argument does
    not apply here, since the base is a fixed non-learned function of the coarse field
    and the network still receives the raw 128x304 sla.

    What it buys: block_repeat is piecewise constant, so its gradient is a delta comb
    at every 8x3 block edge.  Measured on swath geometry, that control sits five
    orders above smooth DUACS below 20 km, and mu inherits it.
    """
    return F.interpolate(x, scale_factor=(C.REFINE_LAT, C.REFINE_LON),
                         mode=mode, align_corners=False)


def gn(c):
    """GroupNorm with the largest valid group count up to 32.

    num_channels must be divisible by num_groups, so a plain min(32, c//4) breaks for
    any width that is not a multiple of 32 (e.g. 176).  Widths already in use -- 128,
    80, 64, 32 -- are unaffected, so existing checkpoints still load.
    """
    g = min(32, max(1, c // 4))
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    """Pre-activation residual block, optionally FiLM-conditioned on a time/noise
    embedding (used by the diffusion net, unused by the baseline)."""

    def __init__(self, c_in, c_out, emb_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = gn(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Linear(emb_dim, 2 * c_out) if emb_dim else None
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.emb is not None and emb is not None:
            scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over the spatial grid.  Only used at the two
    coarsest UNet levels, where the token count is small (24x24 and 12x12)."""

    def __init__(self, c):
        super().__init__()
        self.norm = gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ a.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


def noise_embedding(sigma, dim):
    """Sinusoidal embedding of log-sigma (EDM's c_noise)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=sigma.device) / half)
    ang = sigma.float().reshape(-1, 1) * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


# --------------------------------------------------------------------------------
# Stage 1: deterministic conditional mean
# --------------------------------------------------------------------------------
class BaselineNet(nn.Module):
    """mu(x) = block_repeat(sla) + f(sla).

    Predicting the increment over the block-repeated coarse field rather than the
    absolute field: sla and ssha are the same physical quantity in the same units,
    so block_repeat(sla) is already a strong predictor and the network only has to
    supply the fine-scale deviation.

    Input  (B, 2, 128, 304)  normalised sla with NaN->0, plus its validity mask
    Output (B, 1, 1024, 912) normalised mean field
    """

    def __init__(self, width=128, fine_width=64, n_coarse=6, n_fine=3,
                 base="block", up="pixelshuffle"):
        """base: 'block' (piecewise constant) or 'bicubic'/'bilinear' (smooth)
        up:   'pixelshuffle' (sub-pixel conv) or 'resize' (resize-convolution)

        Defaults reproduce the original network so existing checkpoints still load;
        `--base bicubic --up resize` is the artifact-free variant.
        """
        super().__init__()
        self.base, self.up = base, up
        self.stem = nn.Conv2d(2, width, 3, padding=1)
        self.coarse = nn.ModuleList(
            [ResBlock(width, width) for _ in range(n_coarse)])
        if up == "pixelshuffle":
            self.to_fine = nn.Conv2d(
                width, fine_width * C.REFINE_LAT * C.REFINE_LON, 3, padding=1)
            self.shuffle = PixelShuffle2D(C.REFINE_LAT, C.REFINE_LON)
        else:
            # Resize-convolution (Odena et al): upsample smoothly, then convolve.
            # Channels are cut to fine_width BEFORE upsampling, otherwise the fine-grid
            # activation would be width x 1024 x 912.
            self.to_fine = nn.Conv2d(width, fine_width, 3, padding=1)
            self.post_up = nn.Conv2d(fine_width, fine_width, 3, padding=1)
        self.fine = nn.ModuleList(
            [ResBlock(fine_width, fine_width) for _ in range(n_fine)])
        self.out_norm = gn(fine_width)
        self.out = nn.Conv2d(fine_width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_coarse):
        sla = x_coarse[:, :1]
        h = self.stem(x_coarse)
        for blk in self.coarse:
            h = blk(h)
        if self.up == "pixelshuffle":
            h = self.shuffle(self.to_fine(h))
        else:
            h = self.post_up(smooth_upsample(self.to_fine(h), "bilinear"))
        for blk in self.fine:
            h = blk(h)
        base = (block_repeat(sla) if self.base == "block"
                else smooth_upsample(sla, self.base))
        return base + self.out(F.silu(self.out_norm(h)))


def baseline_from_ckpt(ck):
    """Rebuild a BaselineNet with the architecture its checkpoint was trained with.

    Always use this rather than BaselineNet(width, fine_width): the base and upsampler
    are now options, and constructing with the defaults silently builds the OLD
    architecture, which then fails to load a smooth checkpoint (or, worse, would load
    a matching one and quietly be a different network).  Checkpoints predating the
    options have no 'base'/'up' keys and fall back to the original behaviour.
    """
    a = ck["args"]
    return BaselineNet(a["width"], a["fine_width"],
                       base=a.get("base", "block"),
                       up=a.get("up", "pixelshuffle"))


# --------------------------------------------------------------------------------
# Stage 2: EDM denoiser for the residual
# --------------------------------------------------------------------------------
class DiffusionUNet(nn.Module):
    """Denoiser F(r_noisy, cond, sigma) on the fine grid.

    Conditioning channels:
      mu            stage-1 mean on this patch (large scales the residual sits on)
      coarse feats  learned PixelShuffle(8,3) lift of the coarse patch -- learned
                    rather than bicubic, so no interpolation kernel is stamped in
      fill mask     1 where ssha was observed, 0 where r_0 was filled.  Sampling
                    passes all-ones, which is the known train/inference mismatch;
                    --center-residual corrects the offset it produces.
    """

    def __init__(self, ch=(64, 128, 256), emb_dim=256, coarse_feats=8,
                 dropout=0.1, attn_levels=(2,)):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.emb_dim = emb_dim

        # 2 coarse input channels: sla and its validity mask.
        self.coarse_lift = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, coarse_feats * C.REFINE_LAT * C.REFINE_LON, 3, padding=1))
        self.coarse_shuffle = PixelShuffle2D(C.REFINE_LAT, C.REFINE_LON)

        c_in = 1 + 1 + coarse_feats + 1          # +1 = the fill-mask channel
        self.stem = nn.Conv2d(c_in, ch[0], 3, padding=1)

        self.down, self.down_attn, self.pool = (nn.ModuleList() for _ in range(3))
        prev = ch[0]
        for lvl, c in enumerate(ch):
            self.down.append(nn.ModuleList(
                [ResBlock(prev, c, emb_dim, dropout),
                 ResBlock(c, c, emb_dim, dropout)]))
            self.down_attn.append(SelfAttention(c) if lvl in attn_levels else nn.Identity())
            self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1)
                             if lvl < len(ch) - 1 else nn.Identity())
            prev = c

        self.mid1 = ResBlock(prev, prev, emb_dim, dropout)
        self.mid_attn = SelfAttention(prev)
        self.mid2 = ResBlock(prev, prev, emb_dim, dropout)

        self.up, self.up_attn = nn.ModuleList(), nn.ModuleList()
        for lvl, c in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList(
                [ResBlock(prev + c, c, emb_dim, dropout),
                 ResBlock(c, c, emb_dim, dropout)]))
            self.up_attn.append(SelfAttention(c) if lvl in attn_levels else nn.Identity())
            prev = c

        self.out_norm = gn(prev)
        self.out = nn.Conv2d(prev, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, r_noisy, c_noise, mu, x_coarse, obs_mask=None):
        emb = self.emb(noise_embedding(c_noise, self.emb_dim))
        cf = self.coarse_shuffle(self.coarse_lift(x_coarse))
        # Sampling passes no obs_mask, so the channel is all-ones there while
        # training always sees the day's real mask.  That mismatch is the known
        # approximation recorded in train_diffusion.py; --center-residual corrects
        # the offset it produces.
        if obs_mask is None:
            obs_mask = torch.ones_like(r_noisy)
        h = self.stem(torch.cat([r_noisy, mu, cf, obs_mask], dim=1))

        skips = []
        for blocks, attn, pool in zip(self.down, self.down_attn, self.pool):
            for blk in blocks:
                h = blk(h, emb)
            h = attn(h)
            skips.append(h)
            h = pool(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for blocks, attn in zip(self.up, self.up_attn):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
            h = attn(h)

        return self.out(F.silu(self.out_norm(h)))

class CoarseDiffusionUNet(nn.Module):
    """EDM denoiser for the coarse increment delta_c on the WHOLE 128x304 domain.

    The fine model cannot express mesoscale uncertainty: its tiles are 167x152 km and
    every member shares one frozen mu, so the ensemble says nothing about the field
    DUACS got wrong (measured: spread-skill 0.727, and the increment a member adds
    over DUACS correlates with the true increment at r=0.02).  This network supplies
    exactly that missing degree of freedom, and it is cheap because 128x304 is small
    enough to denoise in one pass -- no tiling, so 50-500 km is native here.

    Conditioning: DUACS sla, its validity mask, and the per-cell SWOT coverage.
    """

    def __init__(self, ch=(64, 128, 192, 256), emb_dim=256, dropout=0.1,
                 attn_levels=(3,), cond_ch=3):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.emb_dim = emb_dim
        self.stem = nn.Conv2d(1 + cond_ch, ch[0], 3, padding=1)

        self.down, self.down_attn, self.pool = (nn.ModuleList() for _ in range(3))
        prev = ch[0]
        for lvl, c in enumerate(ch):
            self.down.append(nn.ModuleList(
                [ResBlock(prev, c, emb_dim, dropout),
                 ResBlock(c, c, emb_dim, dropout)]))
            self.down_attn.append(
                SelfAttention(c) if lvl in attn_levels else nn.Identity())
            self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1)
                             if lvl < len(ch) - 1 else nn.Identity())
            prev = c

        self.mid1 = ResBlock(prev, prev, emb_dim, dropout)
        self.mid_attn = SelfAttention(prev)
        self.mid2 = ResBlock(prev, prev, emb_dim, dropout)

        self.up, self.up_attn = nn.ModuleList(), nn.ModuleList()
        for lvl, c in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList(
                [ResBlock(prev + c, c, emb_dim, dropout),
                 ResBlock(c, c, emb_dim, dropout)]))
            self.up_attn.append(
                SelfAttention(c) if lvl in attn_levels else nn.Identity())
            prev = c

        self.out_norm = gn(prev)
        self.out = nn.Conv2d(prev, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, d_noisy, c_noise, cond):
        emb = self.emb(noise_embedding(c_noise, self.emb_dim))
        h = self.stem(torch.cat([d_noisy, cond], dim=1))
        skips = []
        for blocks, attn, pool in zip(self.down, self.down_attn, self.pool):
            for blk in blocks:
                h = blk(h, emb)
            h = attn(h)
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for blocks, attn in zip(self.up, self.up_attn):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
            h = attn(h)
        return self.out(F.silu(self.out_norm(h)))
