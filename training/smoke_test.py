"""Shape and wiring checks for every component, on one GPU, in about a minute.

Run this before submitting any long training job.  It does not check that the model
learns anything -- only that shapes line up, the masked loss ignores unobserved
pixels, the samplers run, and the tiled sampler covers the field exactly.
"""
import numpy as np
import torch

import config as C
import edm
from nets import BaselineNet, DiffusionUNet, PixelShuffle2D, block_repeat

dev = torch.device("cuda")
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("PixelShuffle2D / block_repeat")
ps = PixelShuffle2D(C.REFINE_LAT, C.REFINE_LON)
z = torch.arange(1 * 24 * 2 * 3, dtype=torch.float32).reshape(1, 24, 2, 3)
check("shuffle shape", tuple(ps(z).shape) == (1, 1, 16, 9), tuple(ps(z).shape))
c = torch.randn(1, 1, 4, 5)
br = block_repeat(c)
check("block_repeat shape", tuple(br.shape) == (1, 1, 32, 15))
check("block_repeat is piecewise constant",
      torch.allclose(br[0, 0, 8:16, 3:6], c[0, 0, 1, 1].expand(8, 3)))

print("\nBaselineNet")
net = BaselineNet(width=32, fine_width=16).to(dev).eval()
with torch.no_grad():
    xi = torch.randn(1, 2, *C.PATCH_C, device=dev)
    d0 = float((net(xi) - block_repeat(xi[:, :1])).abs().max())
check("zero-init net == block_repeat(sla)", d0 == 0.0,
      f"max|d| {d0:.1e} -- training starts from the trivial predictor")
with torch.no_grad():
    xp = torch.randn(2, 2, *C.PATCH_C, device=dev)
    yp = net(xp)
    check("patch out", tuple(yp.shape) == (2, 1, *C.PATCH_F), tuple(yp.shape))
    xf = torch.randn(1, 2, C.NLAT_C, C.NLON_C, device=dev)
    yf = net(xf)
    check("full-field out", tuple(yf.shape) == (1, 1, C.NLAT_F, C.NLON_F),
          tuple(yf.shape))
# Measure the coarse receptive field directly: perturb the output at one pixel and
# see how far the gradient reaches back across the coarse input.  This is the reason
# stage 1 trains on whole fields -- the RF exceeds the patch height, so a
# patch-trained net would lean on zero padding that is absent at full-field
# inference.  If someone shrinks the trunk, this number moves and the decision should
# be revisited.  Deliberately OUTSIDE the no_grad blocks above: it needs a graph.
with torch.no_grad():
    for p in net.out.parameters():          # undo the zero-init, which would make
        torch.nn.init.normal_(p, std=0.05)  # the net exactly block_repeat(sla)
xg = torch.zeros(1, 2, C.NLAT_C, C.NLON_C, device=dev, requires_grad=True)
net(xg)[0, 0, C.NLAT_F // 2, C.NLON_F // 2].backward()
hit = (xg.grad[0, 0].abs() > 0).nonzero()
rf_lat = int(hit[:, 0].max() - hit[:, 0].min()) + 1
rf_lon = int(hit[:, 1].max() - hit[:, 1].min()) + 1
check("coarse RF exceeds patch height (=> full-field stage 1)",
      rf_lat > C.PATCH_C[0],
      f"RF {rf_lat}x{rf_lon} coarse cells vs patch {C.PATCH_C[0]}x{C.PATCH_C[1]}")

print("\nDiffusionUNet + EDM loss")
dn = DiffusionUNet(ch=(32, 64, 128)).to(dev)
B = 4
r0 = torch.randn(B, 1, *C.PATCH_F, device=dev)
mu = torch.randn(B, 1, *C.PATCH_F, device=dev)
xc = torch.randn(B, 2, *C.PATCH_C, device=dev)
mask = (torch.rand(B, 1, *C.PATCH_F, device=dev) < 0.6).float()
r0 = r0 * mask
with torch.no_grad():
    d = edm.denoise(dn, r0, torch.full((B,), 1.0, device=dev), mu, xc, mask)
check("denoise out", tuple(d.shape) == (B, 1, *C.PATCH_F), tuple(d.shape))

loss = edm.edm_loss(dn, r0, mask, mu, xc, obs_mask=mask)
loss.backward()
gnorm = sum(float(p.grad.pow(2).sum()) for p in dn.parameters()
            if p.grad is not None) ** 0.5
check("loss finite + grads flow", np.isfinite(loss.item()) and gnorm > 0,
      f"loss {loss.item():.4f} |g| {gnorm:.3e}")

# The masked loss must be blind to unobserved pixels: perturbing the target there
# cannot change it.  Same seed both times so the sigma/noise draw is identical.
torch.manual_seed(0)
l1 = edm.edm_loss(dn, r0, mask, mu, xc, obs_mask=mask).item()
torch.manual_seed(0)
r0b = r0 + 5.0 * (1 - mask)
l2 = edm.edm_loss(dn, r0b, mask, mu, xc, obs_mask=mask).item()
check("loss ignores unobserved pixels", abs(l1 - l2) < 1e-4 * max(abs(l1), 1),
      f"{l1:.6f} vs {l2:.6f}")

print("\nSamplers")
dn.eval()
with torch.no_grad():
    rp = edm.sample_patch(dn, mu, xc, n_steps=4)
check("sample_patch", tuple(rp.shape) == (B, 1, *C.PATCH_F)
      and torch.isfinite(rp).all())

muf = torch.randn(1, 1, C.NLAT_F, C.NLON_F, device=dev)
xcf = torch.randn(1, 2, C.NLAT_C, C.NLON_C, device=dev)
td = edm.TiledDenoiser(dn, muf, xcf, tile_batch=64)
cover = torch.zeros_like(muf)
for i, j in td.origins:
    cover[..., i * 8:i * 8 + C.PATCH_F[0], j * 3:j * 3 + C.PATCH_F[1]] += 1
check("tiles cover every pixel", int(cover.min()) >= 1,
      f"min {int(cover.min())} max {int(cover.max())} tiles/pixel")
check("tile weight sum positive", float(td.wsum.min()) > 0,
      f"min {float(td.wsum.min()):.2e} (domain corner, covered by one tile edge)")

with torch.no_grad():
    rf = edm.sample_field(dn, muf, xcf, n_steps=3, tile_batch=96)
check("sample_field", tuple(rf.shape) == (1, 1, C.NLAT_F, C.NLON_F)
      and torch.isfinite(rf).all(), f"std {float(rf.std()):.3f}")

print("\nData")
try:
    from data import PatchDataset, eval_box_slices
    ds = PatchDataset("train", cov_thresh=0.5)
    s = ds[0]
    check("patch sample shapes",
          tuple(s["x"].shape) == (2, *C.PATCH_C)
          and tuple(s["y"].shape) == (1, *C.PATCH_F)
          and tuple(s["mask"].shape) == (1, *C.PATCH_F))
    check("target zero where unobserved",
          float((s["y"] * (1 - s["mask"])).abs().max()) == 0.0)
    check("coverage above threshold", float(s["mask"].mean()) > 0.5,
          f"{float(s['mask'].mean()):.3f}")
    si, sj = eval_box_slices()
    check("eval box non-empty", (si.stop - si.start) > 0 and (sj.stop - sj.start) > 0,
          f"{si.stop-si.start} x {sj.stop-sj.start}")
except Exception as e:
    check("data pipeline", False, f"{type(e).__name__}: {e}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
raise SystemExit(0 if all(ok) else 1)
