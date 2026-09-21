"""Map figures.  Kept in step between the parent project and Extension2 so the two
sets of figures can be compared panel for panel, but they are NOT byte-identical:
Extension2's `plot_day_geo` has no `vmax`/`spread_max`, so it scales per day while the
parent can fix one scale over a record.  Port changes to the other copy by hand.

Two files per day:
  day_<date>.png      sea level
  day_<date>_geo.png  the geostrophic current speed derived from it
"""
import numpy as np

import config as C

EXTENT = [C.LON0, C.LON0 + C.NLON_C * C.DLON_C,
          C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C]
LON1, LAT1 = EXTENT[1], EXTENT[3]
ASPECT = 1.0 / np.cos(np.deg2rad(70.0))   # 1 deg lon ~38 km, 1 deg lat ~111 km
OMEGA, G = 7.2921e-5, 9.81


def land_fine(land_coarse):
    return np.repeat(np.repeat(land_coarse, C.REFINE_LAT, axis=0),
                     C.REFINE_LON, axis=1)


def fine_coords():
    lat = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    return lat, lon


def geo_speed(eta_m, lat_deg, dlat_deg, dlon_deg):
    """|u_g| = (g/f) |grad eta|, in m/s.

    Currents are a derivative of sea level, so they weight fine scales far more
    heavily than the field itself does -- which is exactly why they are worth
    plotting separately.  A reconstruction can look plausible in SSH and still have
    visibly wrong fronts here.  NaN in eta propagates, so gappy truth stays gappy.
    """
    f = 2 * OMEGA * np.sin(np.deg2rad(lat_deg))[:, None]
    dy = dlat_deg * 111_320.0
    dx = (dlon_deg * 111_320.0 * np.cos(np.deg2rad(lat_deg)))[:, None]
    dedy = np.full_like(eta_m, np.nan)
    dedx = np.full_like(eta_m, np.nan)
    dedy[1:-1] = (eta_m[2:] - eta_m[:-2]) / (2 * dy)
    dedx[:, 1:-1] = (eta_m[:, 2:] - eta_m[:, :-2]) / (2 * dx)
    return (G / np.abs(f)) * np.hypot(dedx, dedy)


def graticule(ax, dlon=2.5, dlat=2.0, fontsize=6):
    lon_e = list(np.arange(C.LON0, LON1 - 1e-9, dlon)) + [LON1]
    lat_e = list(np.arange(C.LAT0, LAT1 - 1e-9, dlat)) + [LAT1]
    for x in lon_e:
        ax.axvline(x, color="k", lw=0.4, alpha=0.45)
    for y in lat_e:
        ax.axhline(y, color="k", lw=0.4, alpha=0.45)
    if fontsize:
        for i in range(len(lat_e) - 1):
            for j in range(len(lon_e) - 1):
                ax.text(0.5 * (lon_e[j] + lon_e[j + 1]),
                        0.5 * (lat_e[i] + lat_e[i + 1]),
                        f"{chr(65 + j)}{i + 1}", ha="center", va="center",
                        fontsize=fontsize, color="k", alpha=0.75, clip_on=True)
    ax.set_xlim(C.LON0, LON1)
    ax.set_ylim(C.LAT0, LAT1)


def _dress(ax, title, dlon, dlat, fs=6, plt=None):
    ax.set_title(title, fontsize=10)
    ax.set_aspect(ASPECT)
    graticule(ax, dlon, dlat, fontsize=fs)
    ax.add_patch(plt.Rectangle(
        (C.EVAL_LON[0], C.EVAL_LAT[0]),
        C.EVAL_LON[1] - C.EVAL_LON[0], C.EVAL_LAT[1] - C.EVAL_LAT[0],
        fill=False, ec="k", lw=1.6, ls="--"))


def plot_day(outdir, date, sla_c, ens, truth, land_c, mean, std,
             mu=None, context=None, holdout=None, dlon=2.5, dlat=2.0, tag="",
             vmin=None, vmax=None, spread_max=None, extra=None, prefix="day"):
    """Sea level.  Three rows, one shared colour scale for everything except the
    spread panel, which is in cm and carries its own bar.

      row 1  DUACS input | mean of the members | SWOT truth
      row 2  deterministic mu | SWOT painted over the reconstruction | member spread
      row 3  four members

    Row 2 reads left to right as the anatomy of the reconstruction: the smooth field
    the diffusion starts from, whether the result agrees with the data, and where the
    members disagree with each other -- i.e. where the answer is least determined.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    land = land_fine(land_c)
    phys = lambda a: a * std + mean
    truth_p = np.where(np.isfinite(truth), phys(truth), np.nan)
    ens_p = np.where(land[None] > 0, phys(ens), np.nan)
    recon_p = np.nanmean(ens_p, axis=0)
    spread = np.where(land > 0, 100.0 * std * ens.std(axis=0), np.nan)
    sla_p = np.where(land_c > 0, phys(sla_c), np.nan)

    finite = truth_p[np.isfinite(truth_p)]
    # A per-day scale makes consecutive days incomparable: the same colour means a
    # different sea level in each figure, and an animation appears to pulse.  Callers
    # plotting a whole record pass one scale computed over the record.
    if vmin is not None and vmax is not None:
        lo, hi = vmin, vmax
    else:
        lo, hi = (np.percentile(finite, [1, 99]) if finite.size > 10000
                  else np.nanpercentile(recon_p, [1, 99]))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    over = plt.get_cmap("RdBu_r").copy()
    over.set_bad(alpha=0.0)
    hot = plt.get_cmap("magma").copy()
    hot.set_bad("0.85")
    kw = dict(origin="lower", extent=EXTENT, vmin=lo, vmax=hi, cmap=cmap,
              interpolation="nearest")
    lat_f, lon_f = fine_coords()

    def dress(ax, title, fs=6):
        _dress(ax, title, dlon, dlat, fs, plt)

    n_show = min(4, ens.shape[0])
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.0, 1.0, 0.78],
                          hspace=0.10, wspace=0.35)

    ax_in = fig.add_subplot(gs[0, 0:4])
    ax_mn = fig.add_subplot(gs[0, 4:8])
    ax_tr = fig.add_subplot(gs[0, 8:12])
    im = ax_in.imshow(sla_p, **kw)
    dress(ax_in, f"DUACS sla -- input, 1/8 deg ({sla_c.shape[0]}x{sla_c.shape[1]})")
    ax_in.set_ylabel("latitude")
    ax_mn.imshow(recon_p, **kw)
    dress(ax_mn, f"Reconstruction -- mean of {ens.shape[0]} members")
    ax_tr.imshow(truth_p, **kw)
    dress(ax_tr, f"SWOT ssha -- truth ({100*np.isfinite(truth).mean():.1f}% observed)")

    ax_mu = fig.add_subplot(gs[1, 0:4])
    ax_mu.imshow(np.where(land > 0, phys(mu), np.nan)
                 if mu is not None else np.full_like(recon_p, np.nan), **kw)
    dress(ax_mu, "Deterministic mu (stage 1) -- no generated texture")
    ax_mu.set_ylabel("latitude")

    ax_ov = fig.add_subplot(gs[1, 4:8])
    ax_ov.imshow(recon_p, **kw)
    ax_ov.imshow(np.ma.masked_invalid(truth_p), **{**kw, "cmap": over})
    ax_ov.contour(lon_f, lat_f, np.isfinite(truth).astype(float), levels=[0.5],
                  colors="k", linewidths=0.35, alpha=0.55)
    for mask, col in ((context, "tab:blue"), (holdout, "tab:red")):
        if mask is not None:
            ax_ov.contour(lon_f, lat_f, mask.astype(float), levels=[0.5],
                          colors=col, linewidths=0.7)
    sub = ("blue = conditioned on, red = held out"
           if holdout is not None else "conditioned on every swath")
    dress(ax_ov, f"SWOT painted over the reconstruction -- {sub}")

    ax_sp = fig.add_subplot(gs[1, 8:12])
    im_sp = ax_sp.imshow(spread, origin="lower", extent=EXTENT, cmap=hot,
                         interpolation="nearest",
                         vmax=(spread_max if spread_max is not None
                               else np.nanpercentile(spread, 99)))
    dress(ax_sp, f"Spread of the {ens.shape[0]} members [cm] -- where the "
                 f"prediction is least determined")
    fig.colorbar(im_sp, ax=ax_sp, fraction=0.035, pad=0.02, label="std [cm]")

    for k in range(n_show):
        axk = fig.add_subplot(gs[2, 3 * k:3 * k + 3])
        # `extra` replaces the LAST member panel with a named field -- used to put
        # NorKyst beside the members on the same colour scale.  The last slot rather
        # than a middle one so the members stay numbered consecutively.
        if extra is not None and k == n_show - 1:
            ename, efield = extra
            axk.imshow(np.where(land > 0, phys(efield), np.nan), **kw)
            dress(axk, ename, fs=4)
        else:
            axk.imshow(ens_p[k], **kw)
            dress(axk, f"member {k + 1}", fs=4)
        axk.set_xlabel("longitude")
        if k == 0:
            axk.set_ylabel("latitude")

    # Anchored to row 1 only.  Spanning row 2 as well puts it on top of the spread
    # panel's own colorbar; rows 1 and 2 share the same scale either way.
    fig.colorbar(im, ax=[ax_in, ax_mn, ax_tr],
                 fraction=0.015, pad=0.01, label="sea level [m]")
    fig.suptitle(f"{date}   {tag}   grid {dlon} deg lon x {dlat} deg lat", fontsize=13)
    path = outdir / f"{prefix}_{date}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_day_geo(outdir, date, sla_c, ens, truth, land_c, mean, std,
                 mu=None, dlon=2.5, dlat=2.0, tag="", vmax=None,
                 spread_max=None, extra=None, prefix="Bday"):
    """Geostrophic current speed, 3 x 3, same layout and colour scale throughout.

    This is the harder test of the two figures.  Speed is |grad eta| scaled by g/f,
    so it amplifies exactly the short scales the model has to invent: DUACS gives
    broad, weak currents, SWOT shows sharp filaments, and a reconstruction that looks
    convincing in sea level can still be visibly wrong here.

    SPEED IS A MAGNITUDE, SO THE ORDER OF OPERATIONS MATTERS.  `sp_mean` below is
    mean_i |grad eta_i| -- the mean of the member SPEEDS -- and NOT |grad mean_i eta_i|,
    the speed of the mean field.  The triangle inequality makes the first >= the second
    always, and `diag_geo_bias.py` measures the gap at 1.61x on archive_wh13 (15.575
    against 9.652 cm/s).  The panel was titled "mean of N members", which reads as the
    second and is off by that factor; it now names the quantity.  The SWOT-painted
    panel uses a single MEMBER, because a realisation is the like-for-like comparison
    against an observation and CLAUDE.md's standing rule is to use one member for
    anything involving derivatives.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    land = land_fine(land_c)
    phys = lambda a: a * std + mean
    lat_f, lon_f = fine_coords()
    lat_c = C.coarse_lat(np.arange(C.NLAT_C))
    dfy, dfx = C.DLAT_C / C.REFINE_LAT, C.DLON_C / C.REFINE_LON

    def fine_speed(a, valid=None):
        s = 100.0 * geo_speed(phys(a), lat_f, dfy, dfx)     # cm/s
        return np.where(land > 0 if valid is None else valid, s, np.nan)

    sp_in = 100.0 * geo_speed(phys(sla_c), lat_c, C.DLAT_C, C.DLON_C)
    sp_in = np.where(land_c > 0, sp_in, np.nan)
    sp_ens = np.stack([fine_speed(e) for e in ens])
    sp_mean = np.nanmean(sp_ens, axis=0)
    sp_spread = np.nanstd(sp_ens, axis=0)
    sp_tr = 100.0 * geo_speed(np.where(np.isfinite(truth), phys(truth), np.nan),
                              lat_f, dfy, dfx)
    sp_mu = fine_speed(mu) if mu is not None else np.full_like(sp_mean, np.nan)

    ref = sp_tr[np.isfinite(sp_tr)]
    hi = (vmax if vmax is not None
          else np.nanpercentile(ref if ref.size > 10000 else sp_mean, 98))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    over = plt.get_cmap("viridis").copy()
    over.set_bad(alpha=0.0)
    hot = plt.get_cmap("magma").copy()
    hot.set_bad("0.85")
    kw = dict(origin="lower", extent=EXTENT, vmin=0, vmax=hi, cmap=cmap,
              interpolation="nearest")

    def dress(ax, title, fs=6):
        _dress(ax, title, dlon, dlat, fs, plt)

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 9, height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.10, wspace=0.35)
    axes = [[fig.add_subplot(gs[r, 3 * c:3 * c + 3]) for c in range(3)]
            for r in range(3)]

    im = axes[0][0].imshow(sp_in, **kw)
    dress(axes[0][0], "DUACS -- geostrophic speed, 1/8 deg")
    axes[0][1].imshow(sp_mean, **kw)
    dress(axes[0][1], f"Mean of the {ens.shape[0]} member SPEEDS "
                      f"-- NOT the speed of the mean field")
    axes[0][2].imshow(sp_tr, **kw)
    dress(axes[0][2], "SWOT -- truth, observed only")

    axes[1][0].imshow(sp_mu, **kw)
    dress(axes[1][0], "Deterministic mu (stage 1)")
    axes[1][1].imshow(sp_ens[0], **kw)
    axes[1][1].imshow(np.ma.masked_invalid(sp_tr), **{**kw, "cmap": over})
    axes[1][1].contour(lon_f, lat_f, np.isfinite(sp_tr).astype(float), levels=[0.5],
                       colors="k", linewidths=0.35, alpha=0.55)
    dress(axes[1][1], "SWOT painted over member 1 (a realisation)")
    im_sp = axes[1][2].imshow(sp_spread, origin="lower", extent=EXTENT, cmap=hot,
                              interpolation="nearest",
                              vmax=(spread_max if spread_max is not None
                                    else np.nanpercentile(sp_spread, 99)))
    dress(axes[1][2], f"Spread of the {ens.shape[0]} members [cm/s]")
    fig.colorbar(im_sp, ax=axes[1][2], fraction=0.035, pad=0.02, label="std [cm/s]")

    for c in range(3):
        if extra is not None and c == 2:
            # same convention as plot_day: the last member panel becomes the named
            # field, here shown as ITS geostrophic speed so the panel is comparable
            ename, efield = extra
            axes[2][c].imshow(fine_speed(efield), **kw)
            dress(axes[2][c], ename)
        else:
            axes[2][c].imshow(sp_ens[c] if c < sp_ens.shape[0]
                              else np.full_like(sp_mean, np.nan), **kw)
            dress(axes[2][c], f"member {c + 1}")
        axes[2][c].set_xlabel("longitude")
    for r in range(3):
        axes[r][0].set_ylabel("latitude")

    # Row 1 only, for the same reason as in plot_day: spanning the lower rows
    # collides with the spread panel's colorbar.  All nine panels share this scale.
    fig.colorbar(im, ax=axes[0], fraction=0.015, pad=0.01,
                 label="geostrophic speed [cm/s]")
    fig.suptitle(f"{date}   geostrophic currents   {tag}   "
                 f"grid {dlon} deg lon x {dlat} deg lat", fontsize=13)
    path = outdir / f"{prefix}_{date}_geo.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
