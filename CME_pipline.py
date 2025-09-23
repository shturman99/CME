#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# CME / Pencil Code analysis pipeline (single-file, maintainable)
# ============================================================

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse, re, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pencil as pc
from pencil import read

# -----------------------------
# 1) Configuration & style
# -----------------------------
@dataclass
class Config:
    # Paths
    HOME: str = os.path.expanduser("~")
    ROOT: str = os.path.join(os.path.expanduser("~"),
                             "programming", "test", "mhd_project",
                             "tmp", "murman", "CME", "5012", "1D_update")
    FIG_DIR: str = os.path.join(ROOT, "figs_cmi")
    VID_DIR: str = os.path.join(ROOT, "Video")
    # Plotting & behavior
    use_tex: bool = True
    dpi: int = 130
    k_scale: float = 1000.0   # multiply pow.krms by this (wav1=1000)
    log_time_cmap: bool = True
    t_offset: float = 0.0     # subtract from times if your data needs t-1 etc.
    color_map: str = "plasma"

    # Which figures to make
    make_ts: bool = True
    make_brms: bool = True
    make_mag_alltimes: bool = True
    make_hel_alltimes: bool = True
    make_helicity_fraction: bool = True
    make_final_spectra: bool = True

    # Safety
    eps: float = 1e-30

def setup_style(cfg: Config) -> None:
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    mpl.rcParams["figure.dpi"] = cfg.dpi
    mpl.rcParams["text.usetex"] = cfg.use_tex
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["savefig.bbox"] = "tight"


# -----------------------------
# 2) Discovery & reading
# -----------------------------
def discover_sims(root: str) -> List[pc.sim.simulation]:
    runs = sorted([
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".") and name != "CVS"
    ])
    return [pc.get_sim(os.path.join(root, r)) for r in runs]

def read_everything(sims: List[pc.sim.simulation]):
    pow_all: Dict[str, object] = {}
    params_all: Dict[str, object] = {}
    params_all1: Dict[str, object] = {}
    ts_all: Dict[str, object] = {}

    for sim in sims:
        name = sim.name
        try:
            pow_all[name] = read.power(datadir=sim.datadir)
        except Exception as e:
            print(f"[{name}] power reading failed: {e}")

        try:
            params_all[name] = read.param(datadir=sim.datadir, param2=True)
        except Exception as e:
            print(f"[{name}] params reading failed: {e}")

        try:
            params_all1[name] = read.param(datadir=sim.datadir, param1=True)
        except Exception as e:
            print(f"[{name}] params1 reading failed: {e}")

        try:
            ts_all[name] = read.ts(datadir=sim.datadir, file_name="time_series.dat")
        except Exception as e:
            print(f"[{name}] time series reading failed: {e}")

    return pow_all, params_all,params_all1, ts_all


# -----------------------------
# 3) Metrics & helper math
# -----------------------------
def compute_time_markers(ts, p) -> List[Tuple[float, str]]:
    """Return vertical time markers: start, 1/Gamma5, t_phi, finish."""
    t = np.asarray(ts.t)
    t_start = t[0] if t.size else np.nan
    t_finish = t[-1] if t.size else np.nan

    tg = float("nan")
    if hasattr(p, "gammaf5") and p.gammaf5 != 0:
        tg = 1.0 / p.gammaf5

    tp = getattr(p, "source5_expt2", float("nan"))
    S = getattr(p,"source5",float("nan"))
    if S == 0:
        S = float("nan")
    eta = getattr(p,"eta",float("nan"))
    tc = 2 ** (2/3) * tp ** (2/3) * (tg) ** (-2/3) * S ** (-2/3) * eta ** (-1/3)
    out = []
    if np.isfinite(t_start):  out.append((t_start, r"$t_i$"))
    if np.isfinite(tg):      out.append((tg, r"$\Gamma_5^{-1}$"))
    if np.isfinite(tp):      out.append((tp, r"$t_\phi$"))
    if np.isfinite(t_finish): out.append((t_finish, r"$t_f$"))
    if np.isfinite(tc): out.append((tc, r"$t_c$"))
    return out

def compute_k_markers(p) -> List[Tuple[float, str]]:
    """
    Two example k markers:
      k_cross ~~ 2^(-1/3) t_phi^(-1/3) Γ^(-1/3) S5^(1/3) η^(-1/3)
      k_phi   ~~ (1/2) S5 / Γ
    """
    try:
        tp = float(getattr(p, "source5_expt2"))
        g5 = float(getattr(p, "gammaf5"))
        S5 = float(getattr(p, "source5"))
        eta = float(getattr(p, "eta"))
        k_cross = (2.0 ** (-1/3)) * (tp ** (-1/3)) * (g5 ** (-1/3)) * (S5 ** (1/3)) * (eta ** (-1/3))
        k_phi   = 0.5 * S5 / g5
        out = []
        if np.isfinite(k_cross): out.append((k_cross, r"$k_{\rm cross}$"))
        if np.isfinite(k_phi):   out.append((k_phi, r"$k_\phi$"))
        return out
    except Exception:
        return []

def integ_trapz_yx(y: np.ndarray, x: np.ndarray) -> float:
    """Safe trapezoidal ∫ y dx with basic checks."""
    if y.size < 2 or x.size < 2:
        return np.nan
    return np.trapz(y, x)

def compute_helicity_timeseries(pw, use_abs=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    H(t) = ∫ H(k,t) dk (absolute-valued if use_abs=True).
    Returns (t_arr, H_timeseries).
    """
    t_arr = np.asarray(pw.t)
    k = np.asarray(pw.krms)
    H = np.asarray(pw.hel_mag)  # (nt, nk)

    out = []
    for i in range(H.shape[0]):
        spec = np.nan_to_num(H[i], nan=0.0, posinf=0.0, neginf=0.0)
        if use_abs:
            spec = np.abs(spec)
        valid = np.isfinite(k) & np.isfinite(spec)
        if valid.sum() < 2:
            out.append(np.nan)
        else:
            out.append(integ_trapz_yx(spec[valid], k[valid]))
    return t_arr, np.array(out)

def compute_B2_lcoh_weighted(pw, k_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute B^2, ξ_M, and B^2 ξ_M with
      B^2(t)     = ∫ E(k,t) dk
      ξ_M(t)     = [∫ (2π/k) E(k,t) dk] / [∫ E(k,t) dk]
    Returns t_arr, (B2ξ)_t, (B2)_t, (ξ)_t
    """
    t_arr = np.asarray(pw.t)
    k = np.asarray(pw.krms) * k_scale
    E = np.asarray(pw.mag)

    B2, Xi, B2Xi = [], [], []
    for i in range(E.shape[0]):
        Ek = np.nan_to_num(np.abs(E[i]), nan=0.0, posinf=0.0, neginf=0.0)
        valid = (k > 0) & np.isfinite(k) & np.isfinite(Ek)
        if valid.sum() < 2:
            B2.append(np.nan); Xi.append(np.nan); B2Xi.append(np.nan); continue
        kv, Ev = k[valid], Ek[valid]
        b2 = integ_trapz_yx(Ev, kv)
        num = integ_trapz_yx((2.0 * np.pi / kv) * Ev, kv)
        xi = num / b2 if b2 > 0 else np.nan
        B2.append(b2); Xi.append(xi); B2Xi.append(b2 * xi)
    return t_arr, np.array(B2Xi), np.array(B2), np.array(Xi)

def helicity_fraction(k: np.ndarray, Em: np.ndarray, Hm: np.ndarray, eps: float = 1e-30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hfrac(k) = (k H) / (2 E). Clip to [-1,1] for display.
    """
    valid = (k > 0) & np.isfinite(k) & np.isfinite(Em) & np.isfinite(Hm) & (np.abs(Em) > 0)
    kv = k[valid]
    frac = 0.5 * kv * Hm[valid] / (Em[valid] + eps)
    return kv, np.clip(frac, -1.0, 1.0)

def get_param_val(p, name: str) -> float:
    try:
        v = getattr(p, name)
        return float(v)
    except Exception:
        return float("nan")

def _filter_sims(sims, only_runs=None, like=None, regex=None):
    names = [s.name for s in sims]
    keep = set(names)

    if only_runs:
        keep &= set(only_runs)

    if like:
        keep &= {n for n in names if like in n}

    if regex:
        pat = re.compile(regex)
        keep &= {n for n in names if pat.search(n)}

    return [s for s in sims if s.name in keep]

def _pick_geo_window(k, frac_range=(0.2, 0.6)):
    """Pick a contiguous window in k (by fraction of sorted positive k)."""
    kpos = np.asarray(k)
    kpos = kpos[np.isfinite(kpos) & (kpos > 0)]
    if kpos.size < 4:
        return None
    kpos.sort()
    i0 = int(np.clip(frac_range[0] * kpos.size, 0, kpos.size - 2))
    i1 = int(np.clip(frac_range[1] * kpos.size, i0 + 2, kpos.size))
    return kpos[i0:i1]

def add_slope_guides(ax, k, Pk_ref, exponents=(3,),
                          end_frac=0.8, color="0.4"):
    """
    Draw reference slope lines starting at the LEFT edge of k and ending
    at a chosen fraction of the k-domain (default: midpoint).

    - Anchor y to the first finite P(k) at the first positive k.
    - k must be iterable; Pk_ref is one spectrum (same length as k).
    - end_frac in (0,1]: 0.8 = stop at middle of k-domain.
    """
    k = np.asarray(k)
    P = np.asarray(Pk_ref)

    # Keep only positive, finite k and matching finite P
    valid = np.isfinite(k) & (k > 0) & np.isfinite(P) & (np.abs(P) > 0)
    if valid.sum() < 3:
        return

    # Ensure increasing k
    order = np.argsort(k[valid])
    k_valid = k[valid][order]
    P_valid = np.abs(P[valid][order])

    # Anchor at the very first valid (leftmost) point
    k0 = k_valid[0]
    y0 = P_valid[0]

    # Determine end index (midpoint by default)
    i_end = max(2, int(np.floor(end_frac * (len(k_valid) - 1))))
    k_end = k_valid[i_end]

    # Use the existing k grid segment (clean for loglog)
    seg_mask = (k_valid >= k0) & (k_valid <= k_end)
    kseg = k_valid[seg_mask]

    for n in exponents:
        yseg = y0 * (kseg / k0) ** n
        ax.loglog(kseg, yseg, ls="--", lw=1.0, color=color, alpha=0.9)
        # place label near the segment end
        ax.text(kseg[-1], yseg[-1], rf"$k^{n}$", fontsize=9,
                ha="left", va="bottom", color=color)
# -----------------------------
# 4) Plotting
# -----------------------------
def add_time_vlines(ax, markers):
    if not markers:
        return
    xmin, xmax = ax.get_xlim()
    for x, lab in markers:
        if not np.isfinite(x):
            continue
        if x < xmin:
            ax.annotate(lab + " ←", xy=(xmin, 0.85), xycoords=ax.get_xaxis_transform(),
                        ha="left", va="top", fontsize=9, clip_on=True, annotation_clip=True)
        elif x > xmax:
            ax.annotate("→ " + lab, xy=(xmax, 0.85), xycoords=ax.get_xaxis_transform(),
                        ha="right", va="top", fontsize=9, clip_on=True, annotation_clip=True)
        else:
            ax.axvline(x, color="gray", ls="--", lw=0.8, clip_on=True)
            ax.text(x, 0.85, lab, rotation=90, va="top", fontsize=9,
                    transform=ax.get_xaxis_transform(), clip_on=True)

def add_k_vlines(ax, markers):
    if not markers:
        return
    xmin, xmax = ax.get_xlim()
    for x, lab in markers:
        if not np.isfinite(x):
            continue
        if x < xmin:
            ax.annotate(lab + " ←", xy=(xmin, 0.85), xycoords=ax.get_xaxis_transform(),
                        ha="left", va="top", fontsize=9, clip_on=True, annotation_clip=True)
        elif x > xmax:
            ax.annotate("→ " + lab, xy=(xmax, 0.85), xycoords=ax.get_xaxis_transform(),
                        ha="right", va="top", fontsize=9, clip_on=True, annotation_clip=True)
        else:
            ax.axvline(x, color="gray", ls="--", lw=0.8, clip_on=True)
            ax.text(x, 0.85, lab, rotation=90, va="top", fontsize=9,
                    transform=ax.get_xaxis_transform(), clip_on=True)

def run_fig_dir(fig_root: str, run: str) -> str:
    d = os.path.join(fig_root, run)
    os.makedirs(d, exist_ok=True)
    return d

def fig_path(fig_root: str, run: str, stem: str) -> str:
    """
    Save figures under: <FIG_DIR>/<run>/<stem>.pdf
    """
    d = run_fig_dir(fig_root, run)
    return os.path.join(d, f"{stem}.pdf")

def plot_ts_mu5_S5(ts, p, cfg: Config, run: str) -> None:
    t = np.asarray(ts.t) - cfg.t_offset
    JBm = ts.jbm -cfg.t_offset
    mu5 = np.asarray(ts.mu5m)
    lam = getattr(p,"lambda5")
    eta = getattr(p,"eta")
    gamma = getattr(p,"gammaf5")
    S5_over_G = np.asarray(ts.srce5m) / gamma



    fig, ax = plt.subplots(figsize=(6, 4))
    # Add slope guide lines
    # Example: line with slope 1 (∝ t) and slope 2 (∝ t²)
    x0, x1 = t[10], t[50]    # choose points within your t range
    y0 = 1e2                 # adjust y0 for vertical placement
    y1 = 1e0
    fig, ax = plt.subplots(figsize=(6, 4))
    # slope 1: y ∝ t
    ax.plot([x0, x1], [y0, y0 * (x1/x0)], 'k--', lw=1)

    # slope 2: y ∝ t²
    ax.plot([x0, x1], [y1, y1 * (x1/x0)**2], 'k-.', lw=1)

    # Optional: annotate them
    ax.text(x1*1.1, y0 * (x1/x0), r"$\propto t$", fontsize=9, va="bottom")
    ax.text(x1*1.1, y1 * (x1/x0)**2, r"$\propto t^2$", fontsize=9, va="bottom")
    ax.loglog(t, np.abs(mu5), "-x", label=r"$\tilde{\mu}_{5} ~~ [l_{*}^{-1}]$")
    ax.loglog(t, np.abs(S5_over_G), "--", label=r"$\tilde{S}_5 / \Gamma_{5} ~~ [l_{*}^{-1}]$")
    ax.loglog(t, np.abs(lam * eta * (-JBm))/gamma, label = r"$\eta\lambda J\cdot B/\Gamma_{5}$" )

    add_time_vlines(ax, compute_time_markers(ts, p))
    ax.set_xlabel("conformal time :~~ $t~~ [t_*]$",fontsize =12)
    ax.set_ylabel("chiral chemical potential",fontsize =12)
    ax.set_title(run)
    ax.legend()
    fig.savefig(fig_path(cfg.FIG_DIR, run, "ts"), dpi=150)
    plt.close(fig)

def plot_ts_brms_and_hel(ts,p, cfg: Config, run: str) -> None:
    t = np.asarray(ts.t) - cfg.t_offset
    brms = np.asarray(ts.brms)
    hel = np.asarray(ts.abm)
    lam = getattr(p,"lambda5")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(t, np.abs(brms), "o-", label=r"$B_{\mathrm{rms}} ~~ [E_{*}^{1/2} l_*^{-3/2}]$")
    ax.loglog(t, np.abs(hel), "x-",label=r"$|h_M| ~~ [E_*l_*^{-2}] $")
    #ax.loglog(t, np.sqrt(lam) * np.abs(ts.abm), 'o-', label=r"$\sqrt{\lambda} \int d^3x A\cdot B ~~ [E_*^{1/2}l_*^{-3/2}]$") # 10**8 lmabda
    ax.loglog(t, np.abs(ts.abm), 'o-', label=r"$h_M ~~ [E_*l_*^{-2}] $")
    add_time_vlines(ax,compute_time_markers(ts,p))
    ax.set_xlabel(r"conformal time :~~ $t~~ [t_*]$",fontsize =12)
    ax.set_ylabel(r"magnetic field")
    ax.set_title(run)
    ax.legend()
    fig.savefig(fig_path(cfg.FIG_DIR, run, "Brms"), dpi=150)
    plt.close(fig)

def _time_colormap(t: np.ndarray, log_t: bool, cmap: str):
    tt = np.copy(t)
    # ensure strictly positive when log-scaling
    if log_t:
        min_pos = np.nanmin(tt[tt > 0]) if (tt > 0).any() else 1.0
        tt = np.where(tt > 0, tt, min_pos)
        tt = np.log10(tt)
    norm = mpl.colors.Normalize(vmin=np.nanmin(tt), vmax=np.nanmax(tt))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return tt, sm

def plot_alltimes_spectrum(
    pw, cfg: Config,k_scale, run: str, field: str,
    k_markers: Optional[List[Tuple[float, str]]] = None,
    slope_exponents: Optional[List[float]] = None 
) -> None:
    if not hasattr(pw, field):
        print(f"[{run}] no field '{field}' in power.dat")
        return

    k = np.asarray(pw.krms) * k_scale
    arr = np.asarray(getattr(pw, field))  # (nt, nk)
    t = np.asarray(pw.t) - cfg.t_offset

    tt_for_cmap, sm = _time_colormap(t, cfg.log_time_cmap, cfg.color_map)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(arr.shape[0]):
        spec = np.nan_to_num(np.abs(arr[i]), nan=0.0, posinf=0.0, neginf=0.0)
        #valid = (k > 0) & np.isfinite(k) & np.isfinite(spec)
        #if valid.sum() < 2:
        #    continue
        ax.loglog(k, k*spec, lw=0.9, color=sm.to_rgba(tt_for_cmap[i]))
    if k_markers:
        add_k_vlines(ax, k_markers)

     # Optional: slope guides (anchor to last-time spectrum by default)
    if slope_exponents:
        ref = np.nan_to_num(np.abs(arr[0]), nan=0.0, posinf=0.0, neginf=0.0)
        add_slope_guides(ax, k, ref, exponents=slope_exponents,
                         color="0.45")
    ylabel = r"$\mathrm{d}\rho_B/\mathrm{d}\ln k  ~~ [E_* l_*^{-3}]$" if field == "mag" else (r"$|\mathrm{d}h_M/\mathrm{d}\ln k  ~~ [E_* l_*^{-2}]|$" if field == "hel_mag" else field)
    ax.set_xlabel(r"$k~~[l_*^{-1}]$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{run}:  (all times)")
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10} (t/t_*)$" if cfg.log_time_cmap else r"$t$")
    fig.savefig(fig_path(cfg.FIG_DIR, run, f"{field}_alltimes"), dpi=160)
    plt.close(fig)

def plot_final_spectra_with_bound(pw, cfg: Config, k_scale, run: str, k_markers: Optional[List[Tuple[float, str]]] = None) -> None:
    try:
        k = np.asarray(pw.krms) * k_scale
        Em = np.asarray(pw.mag)[-1]
        Hm = np.asarray(pw.hel_mag)[-1] if hasattr(pw, "hel_mag") else None

        fig, ax = plt.subplots(figsize=(6, 4))
        validE = (k > 0) & np.isfinite(k) & np.isfinite(Em)
        ax.loglog(k[validE], np.abs(Em[validE]), label=r"$E_{\rm mag}(k)$")

        if Hm is not None:
            validH = validE & np.isfinite(Hm)
            if validH.sum() >= 2:
                ax.loglog(k[validH], k[validH]*np.abs(Hm[validH]), "--", label=r"$|H_{\rm mag}(k)|$")
                bound = 2.0 * k[validE]* np.abs(Em[validH]) / k[validH]
                ax.loglog(k[validH], bound, ":", label=r"$2E_{\rm mag}(k)/k$")

        if k_markers:
            add_k_vlines(ax, k_markers)

        ax.set_xlabel(r"$k~~[l_*^{-1}]$")
        ax.set_ylabel("spectral density")
        ax.set_title(f"{run}: final spectra")
        ax.legend()
        fig.savefig(fig_path(cfg.FIG_DIR, run, "spectra_final"), dpi=160)
        plt.close(fig)

    except:
        print(f"Can't read spectra properly")

def plot_helicity_fraction_alltimes(pw, cfg: Config, k_scale, run: str, k_markers: Optional[List[Tuple[float, str]]] = None) -> None:
    if not hasattr(pw, "hel_mag"):
        print(f"[{run}] no hel_mag in power.dat")
        return

    k = np.asarray(pw.krms) * k_scale
    EmA = np.asarray(pw.mag)      # (nt, nk)
    HmA = np.asarray(pw.hel_mag)  # (nt, nk)
    t = np.asarray(pw.t) - cfg.t_offset

    tt_for_cmap, sm = _time_colormap(t, cfg.log_time_cmap, cfg.color_map)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(EmA.shape[0]):
        Em = np.nan_to_num(EmA[i], nan=0.0, posinf=0.0, neginf=0.0)
        Hm = np.nan_to_num(HmA[i], nan=0.0, posinf=0.0, neginf=0.0)
        kv, hfrac = helicity_fraction(k, np.abs(Em), Hm, eps=cfg.eps)
        if kv.size < 3:
            continue
        ax.semilogx(kv, hfrac, lw=0.8, color=sm.to_rgba(tt_for_cmap[i]))

    if k_markers:
        add_k_vlines(ax, k_markers)

    ax.axhline(+1.0, ls="--", lw=0.8, c="gray")
    ax.axhline(-1.0, ls="--", lw=0.8, c="gray")
    ax.set_xlabel(r"$k~~[l_*^{-1}]$")
    ax.set_ylabel(r"helicity fraction $kh_M/2\rho_B$")
    ax.set_title(f"{run}: helicity fraction (all times)")
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10} (t/t_*)$" if cfg.log_time_cmap else r"$t$")
    fig.savefig(fig_path(cfg.FIG_DIR, run, "helicity_fraction_alltimes"), dpi=160)
    plt.close(fig)

# -----------------------------
# 5) Animations 
# -----------------------------

# -----------------------------
# 6) Pipeline driver
# -----------------------------
def run_pipeline(cfg: Config, sims_override=None) -> None:
    setup_style(cfg)
    sims = sims_override if sims_override is not None else discover_sims(cfg.ROOT)
    print("Detected runs:", [s.name for s in sims])
    pow_all, params_all,params_all1, ts_all = read_everything(sims)
    # Optional: accumulate a small summary table
    summary_rows = []

    for sim in sims:
        name = sim.name
        pw  = pow_all.get(name)
        ts  = ts_all.get(name)
        par = params_all.get(name)
        par1 = params_all1.get(name)
        print(f"___________{par1.wav1}_______________________")
        if ts is None or par is None:
            print(f"[{name}] skipping (missing ts/params).")
            continue
        print(f"reading sim {name}:")
        # Time series figures
        if cfg.make_ts:
            plot_ts_mu5_S5(ts, par, cfg, name)
        if cfg.make_brms:
            plot_ts_brms_and_hel(ts, par, cfg, name)

        # k markers for spectra
        k_mks = compute_k_markers(par) if pw is not None else []

        # Spectra-style figures
        if pw is not None and cfg.make_mag_alltimes:
            plot_alltimes_spectrum(pw, cfg,par1.wav1, name, field="mag", k_markers=k_mks,slope_exponents=[2,3,4])

        if pw is not None and hasattr(pw, "hel_mag") and cfg.make_hel_alltimes:
            plot_alltimes_spectrum(pw, cfg, par1.wav1,name, field="hel_mag",k_markers=k_mks,slope_exponents=[2,3,4])

        if pw is not None and cfg.make_helicity_fraction:
            plot_helicity_fraction_alltimes(pw, cfg,par1.wav1, name, k_markers=k_mks)

        if pw is not None and cfg.make_final_spectra:
            plot_final_spectra_with_bound(pw, cfg,par1.wav1, name, k_markers=k_mks)

        # Summary metrics: final B_rms, final H_int, final B^2 ξ_M
        try:
            brms_f = float(ts.brms[-1])
            brms_max = np.max(ts.brms)
        except Exception:
            brms_f = np.nan

        try:
            tH, Hts = compute_helicity_timeseries(pw) if pw is not None else (np.array([]), np.array([]))
            H_f = float(Hts[-1]) if Hts.size else np.nan
            H_max = np.max(Hts) if Hts.size else np.nan
        except Exception:
            H_f = np.nan

        try:
            tB2L, B2L, B2, Xi = compute_B2_lcoh_weighted(pw, k_scale=par1.wav1) if pw is not None else (np.array([]),)*4
            B2L_f = float(B2L[-1]) if B2L.size else np.nan
            Xi_f  = float(Xi[-1]) if Xi.size else np.nan
            B2L_max = np.max(B2L) if B2L.size else np.nan
            Xi_max  = np.max(Xi) if Xi.size else np.nan
        except Exception:
            B2L_f, Xi_f = np.nan, np.nan

        lam   = get_param_val(par, "lambda5")
        S5    = get_param_val(par, "source5")
        eta   = get_param_val(par, "eta")
        nu    = get_param_val(par, "nu")
        Diff5 = get_param_val(par, "diffmu5")  
        gam   = get_param_val(par, "gammaf5")

        summary_rows.append((
            name, lam, S5, eta, nu, Diff5, gam,
            brms_f, H_f, B2L_f, Xi_f,
            brms_max, H_max, B2L_max, Xi_max,
        ))

    # Write a simple CSV summary
    if summary_rows:
        import csv
        csv_path = os.path.join(cfg.FIG_DIR, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                                "run",
                                "lambda5", "source5", "eta", "nu", "diffmu5", "gammaf5",
                                "Brms_final", "Hint_final", "B2_xi_final", "xi_final",
                                "Brms_max", "Hint_max", "B2_xi_max", "xi_max"
                            ])
            writer.writerows(summary_rows)
        print("Wrote summary:", csv_path)


# -----------------------------
# 6) Main
# -----------------------------
if __name__ == "__main__":

    cfg = Config()  # tweak defaults here if needed

    ap = argparse.ArgumentParser(description="CME pipeline")
    ap.add_argument("--run", action="append",
                    help="Run name to include (can repeat). Default: all runs.")
    ap.add_argument("--like", help="Substring filter for run names.")
    ap.add_argument("--regex", help="Regex filter for run names.")
    ap.add_argument("--list", action="store_true",
                    help="List detected runs and exit.")
    args = ap.parse_args()

    # Discover once, then optional list/filters
    sims_all = discover_sims(cfg.ROOT)

    if args.list:
        print("Detected runs:", [s.name for s in sims_all])
        sys.exit(0)

    # Apply filters (default: keep all)
    sims_sel = _filter_sims(sims_all,
                            only_runs=args.run,
                            like=args.like,
                            regex=args.regex)

    if not sims_sel:
        print("No runs matched your filter.")
        print("Available:", [s.name for s in sims_all])
        sys.exit(1)

    # Run pipeline only on selected sims
    run_pipeline(cfg, sims_override=sims_sel)