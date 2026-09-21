"""EDM (Karras et al. 2022) denoiser wrapper, masked training loss, and samplers.

EDM rather than DDPM: its sigma-parameterisation is far less sensitive to schedule
hyperparameters, and deterministic Heun reaches good samples in ~30 network
evaluations, which is what makes the tiled full-field sampler affordable.

The two samplers here differ in one crucial respect:

  sample_patch   independent sample of one patch.  Fine for validation figures.
  sample_field   MultiDiffusion: ONE reverse process over the whole 1024x912 canvas,
                 with the network evaluated on overlapping tiles and the DENOISED
                 ESTIMATES averaged at every step, plus a single global noise draw.

Never blend independently drawn samples.  Averaging two independent draws of a
zero-mean field with weights w and 1-w gives variance (w^2+(1-w)^2)*sigma^2 -- half
the variance at a 50/50 seam.  That shows up as a grid of smooth, low-energy bands
and a notch in the PSD at the tile pitch.  Averaging the denoised estimate instead is
safe because D(x;sigma) is a conditional mean, which varies smoothly across overlaps.
"""
import numpy as np
import torch

import config as C

P_MEAN, P_STD = -1.2, 1.2
SIGMA_DATA = 1.0
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0


def precond(sigma, sigma_data=SIGMA_DATA):
    """EDM preconditioning coefficients for a (B,) tensor of noise levels."""
    s = sigma.reshape(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
    c_out = s * sigma_data / (s ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_data ** 2 + s ** 2).sqrt()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def denoise(net, r_noisy, sigma, mu, x_coarse, obs_mask=None):
    """D(r; sigma): the network's estimate of the clean residual."""
    c_skip, c_out, c_in, c_noise = precond(sigma)
    f = net(c_in * r_noisy, c_noise, mu, x_coarse, obs_mask)
    return c_skip * r_noisy + c_out * f


def edm_loss(net, r0, mask, mu, x_coarse, obs_mask=None):
    """Masked EDM denoising loss.

    The loss is averaged over OBSERVED pixels only.  r0 is filled with 0 where the
    target is unobserved -- 0 is the conditional mean of the residual, not a physical
    sea level, so this is not the zero-fill the dataset contract warns against.

    """
    b = r0.shape[0]
    sigma = (torch.randn(b, device=r0.device) * P_STD + P_MEAN).exp()
    noise = torch.randn_like(r0) * sigma.reshape(-1, 1, 1, 1)
    d = denoise(net, r0 + noise, sigma, mu, x_coarse, obs_mask)

    weight = ((sigma ** 2 + SIGMA_DATA ** 2)
              / (sigma * SIGMA_DATA) ** 2).reshape(-1, 1, 1, 1)
    se = weight * (d - r0) ** 2
    denom = mask.sum().clamp(min=1.0)
    return (se * mask).sum() / denom


def karras_sigmas(n, device):
    i = torch.arange(n, device=device, dtype=torch.float64)
    a, b = SIGMA_MAX ** (1 / RHO), SIGMA_MIN ** (1 / RHO)
    s = (a + i / max(n - 1, 1) * (b - a)) ** RHO
    return torch.cat([s, torch.zeros(1, device=device, dtype=torch.float64)]).float()


@torch.no_grad()
def sample_patch(net, mu, x_coarse, n_steps=32, obs_mask=None, generator=None):
    """Independent Heun sample of the residual on one batch of patches."""
    sig = karras_sigmas(n_steps, mu.device)
    r = torch.randn(mu.shape, device=mu.device, generator=generator) * sig[0]
    for k in range(n_steps):
        s_hat, s_next = sig[k], sig[k + 1]
        sv = s_hat.expand(mu.shape[0])
        d = denoise(net, r, sv, mu, x_coarse, obs_mask)
        deriv = (r - d) / s_hat
        r_next = r + (s_next - s_hat) * deriv
        if s_next > 0:                                  # Heun correction
            d2 = denoise(net, r_next, s_next.expand(mu.shape[0]),
                         mu, x_coarse, obs_mask)
            deriv2 = (r_next - d2) / s_next
            r_next = r + (s_next - s_hat) * 0.5 * (deriv + deriv2)
        r = r_next
    return r


def _window(h, w, device):
    """Separable raised cosine, strictly positive so the weight sum never vanishes
    at the domain edges, where a pixel may be covered by a single tile."""
    def axis(n):
        i = torch.arange(n, device=device, dtype=torch.float32)
        return 0.5 * (1 - torch.cos(2 * np.pi * (i + 0.5) / n)) + 1e-3
    return axis(h)[:, None] * axis(w)[None, :]


class TiledDenoiser:
    """Evaluates the denoiser on overlapping tiles and reassembles one full-field
    estimate.  This is the MultiDiffusion step: the averaging happens here, on D,
    inside the reverse process -- never on finished samples."""

    def __init__(self, net, mu, x_coarse, tile_batch=64, obs_mask=None,
                 origins=None):
        self.net, self.mu, self.x = net, mu, x_coarse
        self.tile_batch = tile_batch
        self.obs_mask = obs_mask
        if origins is None:
            oi_c, oj_c = C.patch_origins()
            origins = [(i, j) for i in oi_c for j in oj_c]
        self.origins = origins
        self.ph_f, self.pw_f = C.PATCH_F
        self.ph_c, self.pw_c = C.PATCH_C
        self.win = _window(self.ph_f, self.pw_f, mu.device)

        wsum = torch.zeros_like(mu)
        for i, j in self.origins:
            i0, j0 = i * C.REFINE_LAT, j * C.REFINE_LON
            wsum[..., i0:i0 + self.ph_f, j0:j0 + self.pw_f] += self.win
        self.wsum = wsum.clamp(min=1e-6)

    @torch.no_grad()
    def __call__(self, r, sigma):
        acc = torch.zeros_like(r)
        for s in range(0, len(self.origins), self.tile_batch):
            chunk = self.origins[s:s + self.tile_batch]
            rt, mt, xt, ot = [], [], [], []
            for i, j in chunk:
                i0, j0 = i * C.REFINE_LAT, j * C.REFINE_LON
                rt.append(r[0, :, i0:i0 + self.ph_f, j0:j0 + self.pw_f])
                mt.append(self.mu[0, :, i0:i0 + self.ph_f, j0:j0 + self.pw_f])
                xt.append(self.x[0, :, i:i + self.ph_c, j:j + self.pw_c])
                if self.obs_mask is not None:
                    ot.append(self.obs_mask[0, :, i0:i0 + self.ph_f,
                                            j0:j0 + self.pw_f])
            rt = torch.stack(rt)
            sv = sigma.expand(rt.shape[0])
            d = denoise(self.net, rt, sv, torch.stack(mt), torch.stack(xt),
                        torch.stack(ot) if ot else None)
            for k, (i, j) in enumerate(chunk):
                i0, j0 = i * C.REFINE_LAT, j * C.REFINE_LON
                acc[0, :, i0:i0 + self.ph_f, j0:j0 + self.pw_f] += self.win * d[k]
        return acc / self.wsum


def offset_origins(di, dj):
    """Tile origins shifted by (di, dj) coarse cells, for the seam diagnostic: a
    correct sampler's field statistics must not depend on where the grid is laid."""
    oi_c, oj_c = C.patch_origins()
    return [(i + di, j + dj) for i in oi_c for j in oj_c
            if 0 <= i + di <= C.NLAT_C - C.PATCH_C[0]
            and 0 <= j + dj <= C.NLON_C - C.PATCH_C[1]]


@torch.no_grad()
def sample_field(net, mu, x_coarse, n_steps=32, tile_batch=64, obs_mask=None,
                 generator=None, verbose=False, origins=None, init_noise=None,
                 ):
    """MultiDiffusion Heun sampler over the whole field.

    mu, x_coarse are (1, C, H, W) full-field tensors.  Returns the residual on the
    full fine grid; the caller adds mu back and de-normalises.
    """
    tiled = TiledDenoiser(net, mu, x_coarse, tile_batch, obs_mask, origins)
    sig = karras_sigmas(n_steps, mu.device)
    # One global noise draw, shared by every tile: this is what keeps the sample
    # coherent across tile boundaries.
    #
    # init_noise lets the CALLER own that draw, which is what makes temporal
    # experiments possible: the reverse process is deterministic given the noise and
    # the conditioning, so holding the noise fixed (or correlating it) across days
    # makes consecutive samples vary only as mu does.  It must be unit-variance --
    # it is scaled by sig[0] here, exactly as a fresh draw would be.
    if init_noise is None:
        init_noise = torch.randn(mu.shape, device=mu.device, generator=generator)
    r = init_noise * sig[0]
    for k in range(n_steps):
        s_hat, s_next = sig[k], sig[k + 1]
        d = tiled(r, s_hat)
        deriv = (r - d) / s_hat
        r_next = r + (s_next - s_hat) * deriv
        if s_next > 0:
            d2 = tiled(r_next, s_next)
            r_next = r + (s_next - s_hat) * 0.5 * (deriv + (r_next - d2) / s_next)
        r = r_next
        if verbose:
            print(f"  step {k + 1}/{n_steps}  sigma {float(s_hat):.4f}",
                  flush=True)
    return r
