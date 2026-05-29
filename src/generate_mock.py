from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from scipy.special import j0, jv
import treecorr
import healpy as hp
import euclidlib as el

cls = el.le3.pk_wl.angular_power_spectra("/net/home/fohlen13/reischke/Git/qml_ec/data/theory_for_fit.fits")

cls_data = cls[('SHE', 'SHE', 1, 1)].array[0,0,:]
ell_data = cls[('SHE', 'SHE', 1, 1)].ell
cl_EE_data = cls_data

# Spline in log-log space; extrapolates as a power law beyond the data range
_log_ell = np.log(ell_data[2:])
_log_cl  = np.log(cl_EE_data[2:])
_spline_log = UnivariateSpline(_log_ell, _log_cl, s=0, ext=0)

def cl_EE_extrap(ell):
    """Evaluate C_l^EE with power-law extrapolation to ell beyond the data."""
    ell = np.asarray(ell, dtype=float)
    log_ell_max = _log_ell[-1]
    slope = _spline_log.derivative()(log_ell_max)   # d ln C / d ln ell at last point

    log_ell = np.log(ell)
    inside  = log_ell <= log_ell_max
    result  = np.empty_like(ell)
    result[inside]  = np.exp(_spline_log(log_ell[inside]))
    result[~inside] = np.exp(
        _spline_log(log_ell_max) + slope * (log_ell[~inside] - log_ell_max)
    )
    return result


def generate_emode_maps(nside, lmax=None):
    """Return (Q, U) HEALPy maps of a pure E-mode Gaussian random field."""
    if lmax is None:
        lmax = 3 * nside - 1

    ells = np.arange(lmax + 1, dtype=float)
    cl_EE = np.zeros(lmax + 1)
    cl_EE[2:] = cl_EE_extrap(ells[2:])   # monopole and dipole are zero

    E_alm = hp.synalm(cl_EE, lmax=lmax, new=True)
    B_alm = np.zeros_like(E_alm)
    T_alm = np.zeros_like(E_alm)

    # alm2map with three alm arrays returns (T, Q, U); discard T
    _, Q, U = hp.alm2map([T_alm, E_alm, B_alm], nside=nside, lmax=lmax)
    return Q, U


def make_mask(nside, b_cut_deg=20.0, dec_min=-60.0, dec_max=5.0,
              n_holes=0, hole_radius_range_deg=(0.5, 3.0), seed=None):
    """Return a HEALPy mask (1 = observed) with a galactic-plane cut and dec limits.

    The default limits roughly mimic a southern-hemisphere weak-lensing survey
    (Euclid/KiDS-like footprint without the ecliptic exclusion).

    Parameters
    ----------
    n_holes              : int, number of random circular holes (bright-star style)
    hole_radius_range_deg: (min, max) hole radii in degrees
    seed                 : RNG seed for hole placement
    """
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    theta_eq, phi_eq = hp.pix2ang(nside, ipix)           # equatorial co-lat/lon

    dec = 90.0 - np.degrees(theta_eq)

    # Rotate equatorial → galactic to get galactic latitude
    rot = hp.Rotator(coord=['C', 'G'])
    theta_gal, _ = rot(theta_eq, phi_eq)
    b = 90.0 - np.degrees(theta_gal)

    mask = np.ones(npix, dtype=np.float64)
    mask[np.abs(b) < b_cut_deg] = 0.0
    mask[dec < dec_min] = 0.0
    mask[dec > dec_max] = 0.0

    if n_holes > 0:
        rng = np.random.default_rng(seed)
        inside = np.where(mask > 0)[0]
        centres = rng.choice(inside, size=n_holes, replace=True)
        theta_c, phi_c = hp.pix2ang(nside, centres)
        r_min, r_max = np.radians(hole_radius_range_deg)
        radii = rng.uniform(r_min, r_max, size=n_holes)
        for tc, pc, r in zip(theta_c, phi_c, radii):
            mask[hp.query_disc(nside, hp.ang2vec(tc, pc), r)] = 0.0

    return mask


def sample_emode_catalog(Q, U, mask, n_sources, seed=None):
    """Sample n_sources random positions uniformly within the mask.

    Returns
    -------
    ra, dec : degrees (equatorial)
    gamma1, gamma2 : shear components read from Q, U at each position
    """
    nside = hp.npix2nside(len(mask))
    valid = np.where(mask > 0)[0]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid, size=n_sources, replace=True)

    theta, phi = hp.pix2ang(nside, chosen)
    ra  = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    return ra, dec, Q[chosen], U[chosen]


def xi_pm_theory(theta_deg, lmax=int(1e5), n_ell=int(1e5)):
    """Compute ξ±(θ) from C_l^EE via the flat-sky Hankel transform.

    ξ+(θ) = (1/2π) ∫ dℓ ℓ C_ℓ J_0(ℓθ)
    ξ-(θ) = (1/2π) ∫ dℓ ℓ C_ℓ J_4(ℓθ)

    Uses a log-spaced ell grid so the extrapolated power-law tail at high ℓ
    is sampled efficiently.  lmax should satisfy lmax ≫ 1/θ_min (rad).
    n_ell is kept odd so Simpson's rule covers every sub-interval exactly.
    """
    theta_rad = np.atleast_1d(np.radians(theta_deg))
    ells = np.geomspace(2, lmax, n_ell)
    cl   = cl_EE_extrap(ells)

    # lt[i, j] = ells[j] * theta_rad[i]  →  shape (n_theta, n_ell)
    lt     = np.outer(theta_rad, ells)
    prefac = ells * cl / (2.0 * np.pi)

    xi_p = simpson(prefac * j0(lt),    x=ells, axis=1)
    xi_m = simpson(prefac * jv(4, lt), x=ells, axis=1)
    return xi_p, xi_m


def make_density_map(ra, dec, nside, density_fn, mask=None, smooth_fwhm_deg=10.0):
    """Build a KDE-smoothed HEALPy map of local galaxy density.

    Accumulates density_fn values per pixel and convolves with a Gaussian beam
    (hp.smoothing). The smoothed map naturally drops near mask boundaries because
    part of the kernel lands in empty sky — no normalisation by smoothed counts
    is applied, so edge effects are preserved.

    Parameters
    ----------
    ra, dec         : degrees
    nside           : HEALPy resolution
    density_fn      : callable(ra, dec) → array of shape (n,)
    mask            : optional float/bool HEALPy map (1 = observed)
    smooth_fwhm_deg : Gaussian FWHM in degrees for the KDE kernel

    Returns
    -------
    density_map : HEALPy map of shape (npix,)
    """
    npix = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra))

    values = np.asarray(density_fn(ra, dec), dtype=float)
    weighted_sum = np.bincount(ipix, weights=values, minlength=npix).astype(float)

    fwhm_rad = np.radians(smooth_fwhm_deg)
    smooth_map = hp.smoothing(weighted_sum, fwhm=fwhm_rad, verbose=False)

    if mask is not None:
        observed = mask > 0
    else:
        observed = np.ones(npix, dtype=bool)

    density_map = np.full(npix, hp.UNSEEN)
    density_map[observed] = smooth_map[observed]
    return density_map


def compute_w_gamma_weights(ra, dec, density_fn, xi_eff, sigma_e=0.26):
    """Compute shear weights using

    w_gamma(theta, theta_a) = 1 / (2 sigma_e^2 + n(theta) xi_eff(theta_a)).

    Parameters
    ----------
    ra, dec    : degrees
    density_fn : callable(ra, dec) → local number density n_g
    xi_eff     : float or array of shape (n_bins,)
                 Effective signal ξ_eff(θ_a) = (1/2π)∫ ℓ dℓ C_ℓ J_{0/4}(ℓθ_a)
                 evaluated at TreeCorr bin centres via xi_pm_theory.
    sigma_e    : float, per-component shape noise (default 0.26)

    Returns
    -------
    w : (n_gal,) if xi_eff is scalar, (n_gal, n_bins) if xi_eff is an array
    """
    ng     = np.asarray(density_fn(ra, dec), dtype=float)       # (n_gal,)
    xi_eff = np.atleast_1d(np.asarray(xi_eff, dtype=float))     # (n_bins,)
    return 1.0 / (2.0 * sigma_e**2 + ng[:, None] * xi_eff[None, :])


def measure_xi_pm(ra, dec, g1, g2, w=None,
                  min_sep=15.0, max_sep=400.0, nbins=20, bin_slop=0.05):
    """Estimate ξ±(θ) from a shear catalogue using TreeCorr.

    Applies the HEALPy→TreeCorr convention fix (g1 = −Q, g2 = +U) so
    the inputs should be the raw Q, U values from sample_emode_catalog.

    Parameters
    ----------
    w : None, (n_gal,), or (n_gal, n_bins)
        Per-galaxy weights. If shape is (n_gal, n_bins) (scale-dependent w_gamma),
        TreeCorr is run once per bin using w[:, b] and only bin b is retained,
        implementing the per-band weight of Eq. 11.

    Returns
    -------
    theta   : bin centres in arcmin
    xi_p, xi_m : measured ξ+, ξ−
    err_p, err_m : 1σ errors from TreeCorr's variance estimate
    """
    # Scale-dependent weights: run TreeCorr once per bin, keep that bin.
    if w is not None and np.ndim(w) == 2:
        n_bins = w.shape[1]
        theta_arr = np.empty(n_bins)
        xip_arr   = np.empty(n_bins)
        xim_arr   = np.empty(n_bins)
        errp_arr  = np.empty(n_bins)
        errm_arr  = np.empty(n_bins)
        for b in range(n_bins):
            t, xp, xm, ep, em = measure_xi_pm(
                ra, dec, g1, g2, w=w[:, b],
                min_sep=min_sep, max_sep=max_sep,
                nbins=n_bins, bin_slop=bin_slop)
            theta_arr[b] = t[b]
            xip_arr[b]   = xp[b]
            xim_arr[b]   = xm[b]
            errp_arr[b]  = ep[b]
            errm_arr[b]  = em[b]
        return theta_arr, xip_arr, xim_arr, errp_arr, errm_arr

    # HEALPy IAU: Q>0 is N-S, U>0 is NE-SW from N.
    # TreeCorr:   g1>0 is E-W, g2>0 is NE-SW from E.  → g1 = -Q
    cat = treecorr.Catalog(ra=ra, dec=dec, g1=-g1, g2=g2, w=w,
                           ra_units='deg', dec_units='deg')
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                nbins=nbins, sep_units='arcmin',
                                bin_slop=bin_slop)
    gg.process(cat)
    return (np.exp(gg.meanlogr),
            gg.xip, gg.xim,
            np.sqrt(gg.varxip), np.sqrt(gg.varxim))


# ---------------------------------------------------------------------------
# Multi-realisation runner
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    NSIDE      = 256
    N_GAL      = 200_000
    N_REAL     = 1000
    USE_W_GAMMA = True  # set False for uniform weights
    CRAZY_MASK = True  # set True to add 500 random bright-star-style holes
    DENSITY_SMOOTH_FWHM_DEG = 10.0  # smoothing scale for local density used in w_gamma
    MIN_SEP_ARCMIN = 3 * 10800.0 / NSIDE   # ell < NSIDE → pixel effects negligible
    MAX_SEP_ARCMIN = 400.0
    NBINS          = 8

    PLOT_DIR = Path("./../plots")
    OUT_DIR  = Path("./../output")
    PLOT_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    tag      = ('wgamma' if USE_W_GAMMA else 'uniform') + ('_crazy' if CRAZY_MASK else '')
    out_file = OUT_DIR / f"xi_pm_realisations_{tag}.npz"

    pixel_size_arcmin = hp.nside2resol(NSIDE, arcmin=True)

    # ---- fixed mask and galaxy positions (sampled once) ----
    print("Building mask …")
    mask  = make_mask(NSIDE, n_holes=500 if CRAZY_MASK else 0, seed=0)
    valid = np.where(mask > 0)[0]

    print(f"Sampling {N_GAL} positions …")
    rng    = np.random.default_rng(42)
    chosen = rng.choice(valid, size=N_GAL, replace=True)
    theta_pix, phi_pix = hp.pix2ang(NSIDE, chosen)
    ra  = np.degrees(phi_pix)
    dec = 90.0 - np.degrees(theta_pix)

    # Build a spatially varying local-density estimate from sampled positions.
    # This makes w_gamma respond to survey depth/coverage fluctuations.
    npix = hp.nside2npix(NSIDE)
    counts_map = np.bincount(chosen, minlength=npix).astype(float)
    density_template = hp.smoothing(
        counts_map,
        fwhm=np.radians(DENSITY_SMOOTH_FWHM_DEG),
        verbose=False,
    )
    density_template = np.clip(density_template, 1e-12, None)

    def density_fn(ra_eval, dec_eval):
        pix_eval = hp.ang2pix(NSIDE, np.radians(90.0 - dec_eval), np.radians(ra_eval))
        return density_template[pix_eval]

    n_bar = float(density_fn(ra, dec).mean())
    print(f"Local-density template ready (FWHM={DENSITY_SMOOTH_FWHM_DEG:.1f} deg): n_bar={n_bar:.4e}")

    # ---- theory ξ± at TreeCorr bin centres (fixed geometry) ----
    print("Computing theory ξ± …")
    theta_cents_arcmin = np.geomspace(MIN_SEP_ARCMIN, MAX_SEP_ARCMIN, NBINS)
    xi_p_th, xi_m_th   = xi_pm_theory(theta_cents_arcmin / 60.0)
    theta_th_arcmin    = theta_cents_arcmin

    # xi_eff(theta_a) from theory xi prediction.
    # Use a positive effective amplitude to keep the denominator stable.
    xi_eff = 0.5 * (np.abs(xi_p_th) + np.abs(xi_m_th))   # shape (NBINS,)

    # ---- accumulate results ----
    all_xip   = []
    all_xim   = []
    theta_out = None

    for i_real in range(N_REAL):
        print(f"Realisation {i_real + 1}/{N_REAL} …")

        # regenerate field only; positions stay fixed
        Q, U = generate_emode_maps(NSIDE)
        g1, g2 = Q[chosen], U[chosen]
        if i_real == 0:
            sigma_e = np.sqrt(0.5 * (np.var(g1) + np.var(g2)))
        if USE_W_GAMMA:
            w_gamma = compute_w_gamma_weights(ra, dec, density_fn, xi_eff,
                                              sigma_e=sigma_e)
            # w_gamma shape: (n_gal, NBINS)
        else:
            w_gamma = None

        # ---- plots for first realisation only ----
        if i_real == 0:
            density_map = make_density_map(
                ra,
                dec,
                NSIDE,
                density_fn=lambda ra_loc, dec_loc: np.ones(len(ra_loc)),
                mask=mask,
            )

            hp.mollview(density_map, title="Local galaxy density",
                        unit=r"$n$ [arb.]", cmap="magma")
            plt.savefig(PLOT_DIR / "density_map.pdf", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved density_map.pdf")

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("Pure E-mode mock catalogue (realisation 1)", fontsize=14)
            plt.axes(axes[0, 0])
            hp.mollview(Q, title=r"$Q$ map (E-mode realisation)", unit=r"$\gamma_1$",
                        hold=True, cmap="RdBu_r")
            plt.axes(axes[0, 1])
            hp.mollview(mask, title="Survey mask", hold=True, cmap="binary_r")
            ax = axes[1, 0]
            sc = ax.scatter(ra, dec, c=g1, s=0.3, cmap="RdBu_r",
                            vmin=np.percentile(g1, 1), vmax=np.percentile(g1, 99),
                            rasterized=True)
            fig.colorbar(sc, ax=ax, label=r"$\gamma_1$")
            ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
            ax.set_title("Sampled catalogue positions")
            ax = axes[1, 1]
            ax.hist(g1, bins=80, histtype="step", label=r"$\gamma_1$", density=True)
            ax.hist(g2, bins=80, histtype="step", label=r"$\gamma_2$", density=True)
            ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("PDF")
            ax.set_title("Shear distributions"); ax.legend()
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "emode_mock.pdf", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved emode_mock.pdf")

            if USE_W_GAMMA and w_gamma is not None:
                # w_gamma has shape (n_gal, NBINS); use middle bin as representative
                mid = w_gamma.shape[1] // 2
                w_mid = w_gamma[:, mid]
                theta_mid = theta_cents_arcmin[mid]
                print(
                    f"w_gamma stats (bin {mid}, θ≈{theta_mid:.1f}'): "
                    f"min={w_mid.min():.4e}, max={w_mid.max():.4e}, "
                    f"std={w_mid.std():.4e}"
                )
                # Histogram of per-galaxy weights for each bin.
                fig, ax = plt.subplots(figsize=(7.5, 5.0))
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, NBINS))
                for b in range(NBINS):
                    ax.hist(w_gamma[:, b], bins=80, density=True, histtype="step",
                            color=colors[b],
                            label=fr"$\theta_a={theta_cents_arcmin[b]:.0f}'$")
                ax.set_xlabel(r"$w_\gamma$")
                ax.set_ylabel("PDF")
                ax.set_title(r"$w_\gamma$ distribution per bin (realisation 1)")
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False, fontsize=7, ncol=2)
                plt.tight_layout()
                plt.savefig(PLOT_DIR / "w_gamma_histogram.pdf", dpi=150, bbox_inches="tight")
                plt.close()
                print("Saved w_gamma_histogram.pdf")

                # HEALPix map of mean weight (averaged over bins) per pixel.
                npix_plot = hp.nside2npix(NSIDE)
                w_mean_gal = w_gamma.mean(axis=1)
                w_sum = np.bincount(chosen, weights=w_mean_gal, minlength=npix_plot).astype(float)
                w_cnt = np.bincount(chosen, minlength=npix_plot).astype(float)
                w_map = np.full(npix_plot, hp.UNSEEN)
                occupied = w_cnt > 0
                w_map[occupied] = w_sum[occupied] / w_cnt[occupied]

                hp.mollview(w_map, title=r"Mean $w_\gamma$ per pixel (realisation 1)",
                            unit=r"$\langle w_\gamma\rangle$", cmap="viridis")
                plt.savefig(PLOT_DIR / "w_gamma_map.pdf", dpi=150, bbox_inches="tight")
                plt.close()
                print("Saved w_gamma_map.pdf")

        # ---- measure ξ± ----
        # pixel window function W_ell ~ 1 only for ell < NSIDE;
        # corresponding angular scale: theta > pi/NSIDE rad = 10800/NSIDE arcmin
        theta_meas, xi_p, xi_m, err_p, err_m = measure_xi_pm(
            ra, dec, g1, g2, w=w_gamma,
            min_sep=3*10800.0 / NSIDE, max_sep=400.0, nbins=8,
        )

        if theta_out is None:
            theta_out = theta_meas

        all_xip.append(xi_p)
        all_xim.append(xi_m)

        # ---- xi_pm comparison plot for first realisation ----
        if i_real == 0:
            if theta_th_arcmin is None:
                theta_th_arcmin = np.geomspace(theta_meas.min() * 0.5,
                                               theta_meas.max() * 1.5, 300)
                xi_p_th, xi_m_th = xi_pm_theory(theta_th_arcmin / 60.0)

            scale = 1e4
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(r"Shear two-point functions: measured vs theory "
                         r"(realisation 1)", fontsize=13)
            ax1.fill_between(theta_meas, scale*(xi_p-err_p), scale*(xi_p+err_p),
                             alpha=0.3, label="TreeCorr ±1σ")
            ax1.semilogx(theta_meas, scale*xi_p, 'o', ms=5)
            ax1.semilogx(theta_th_arcmin, scale*xi_p_th, 'k-', lw=1.5, label="Theory")
            ax1.set_xlabel(r"$\theta$ [arcmin]")
            ax1.set_ylabel(r"$\xi_+(\theta) \times 10^4$")
            ax1.set_title(r"$\xi_+$"); ax1.legend()
            ax2.fill_between(theta_meas, scale*(xi_m-err_m), scale*(xi_m+err_m),
                             alpha=0.3, label="TreeCorr ±1σ")
            ax2.semilogx(theta_meas, scale*xi_m, 'o', ms=5)
            ax2.semilogx(theta_th_arcmin, scale*xi_m_th, 'k-', lw=1.5, label="Theory")
            ax2.set_xlabel(r"$\theta$ [arcmin]")
            ax2.set_ylabel(r"$\xi_-(\theta) \times 10^4$")
            ax2.set_title(r"$\xi_-$"); ax2.legend()
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "xi_pm_comparison.pdf", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved xi_pm_comparison.pdf")

        # ---- continuously save results ----
        np.savez(out_file,
                 theta=theta_out,
                 xip=np.array(all_xip),
                 xim=np.array(all_xim),
                 use_fkp=USE_W_GAMMA,
                 theta_th=theta_th_arcmin,
                 xi_p_th=xi_p_th,
                 xi_m_th=xi_m_th)

    print(f"Done. {N_REAL} realisations saved to {out_file}")
