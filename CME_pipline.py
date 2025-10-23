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
from matplotlib import transforms as mtransforms
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
                             "programming",  "murman", "CME", "3D")

    FIG_DIR: str = os.path.join(ROOT, "figs_cmi")
    VID_DIR: str = os.path.join(ROOT, "Video")
    # Plotting & behavior
    use_tex: bool = True
    dpi: int = 200
    #k_scale: float = 1000.0   # multiply pow.krms by this (wav1=1000)
    log_time_cmap: bool = True
    t_offset: float = 0.0     # subtract from times if your data needs t-1 etc.
    color_map: str = "plasma"
    mu5_ymin, mu5_ymax = 10 **(-16), 10**(5)
    # Which figures to make
    make_ts: bool = True
    make_brms: bool = True
    make_mag_alltimes: bool = True
    make_hel_alltimes: bool = True
    make_helicity_fraction: bool = True
    make_final_spectra: bool = True


    # animations
    make_anim_mag: bool = True
    make_anim_hel: bool = True
    make_anim_hel_frac: bool = True
    anim_fps: int = 10
    anim_stride: int = 1      # use every Nth time slice
    anim_y: str = "kE"        # "E" for Em(k), "kE" for k*Em(k), "kH" for k*|H(k)|
    anim_xmin, anim_xmax = 10 ** (-1), 10 ** (3)
    anim_ymin, anim_ymax = 10 ** (-29), 10 ** (5)

    # Safety
    eps: float = 1e-40

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
def discover_sims(root: str) -> List[pc.sim.simulation]: # type: ignore
    """Return only valid Pencil sims under root (skip figs, Video, etc.)."""
    sims: List[pc.sim.simulation] = [] # type: ignore
    for name in sorted(os.listdir(root)):
        if name.startswith(".") or name in {"CVS", "figs_cmi", "Video"}:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue

        # Quietly probe; pc.get_sim() returns falsy if not a sim
        sim = pc.get_sim(path, quiet=True)
        if not sim:
            # optional: print(f"? Skipping {name}: not a Pencil sim")
            continue

        # Make sure basic data exists (avoid later dim.dat errors)
        dimdat = os.path.join(sim.datadir, "dim.dat")
        if not os.path.exists(dimdat):
            print(f"? WARNING: Skipping {name}: missing {dimdat}")
            continue

        sims.append(sim)

    if not sims:
        print(f"? WARNING: No simulations found in {root}")
    else:
        print("Detected runs:", [s.name for s in sims]) # type: ignore
    return sims


def read_everything(sims: List[pc.sim.simulation]): # type: ignore
    pow_all: Dict[str, object] = {}
    params_all: Dict[str, object] = {}
    params_all1: Dict[str, object] = {}
    ts_all: Dict[str, object] = {}

    for sim in sims:
        name = sim.name # type: ignore
        try:
            pow_all[name] = read.power(datadir=sim.datadir) # type: ignore
        except Exception as e:
            print(f"[{name}] power reading failed: {e}")

        try:
            params_all[name] = read.param(datadir=sim.datadir, param2=True) # type: ignore
        except Exception as e:
            print(f"[{name}] params reading failed: {e}")

        try:
            params_all1[name] = read.param(datadir=sim.datadir, param1=True) # type: ignore
        except Exception as e:
            print(f"[{name}] params1 reading failed: {e}")

        try:
            ts_all[name] = read.ts(datadir=sim.datadir, file_name="time_series.dat") # type: ignore
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
    return np.trapz(y, x) # type: ignore

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
    """Filter already-validated sims; never assume elements have .name."""
    names = [s.name for s in sims if hasattr(s, "name")]
    keep = set(names)

    if only_runs:
        keep &= set(only_runs)

    if like:
        keep &= {n for n in names if like in n}

    if regex:
        pat = re.compile(regex)
        keep &= {n for n in names if pat.search(n)}

    sel = [s for s in sims if getattr(s, "name", None) in keep]
    if not sel:
        print("No runs matched your filter.")
        print("Available:", names)
    return sel


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

def kcpi_series(ts, k_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t, k_cpi(t)); k_cpi = |mu5|(t)."""
    if ts is None or not hasattr(ts, "mu5m"):
        return np.array([]), np.array([])
    t = np.asarray(ts.t)
    mu = np.asarray(ts.mu5m)
    kc = np.abs(mu) 
    return t, kc

def kpeak_series(pw, k_scale: float, ymode: str = "kE") -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t, k_peak(t)) where peak is argmax over k of:
      ymode="kE" -> k*|E(k)|
      ymode="kH" -> k*|H(k)|
    """
    if pw is None or not hasattr(pw, "t"):
        return np.array([]), np.array([])

    t = np.asarray(pw.t)
    k = np.asarray(pw.krms) * k_scale
    use_hel = (ymode == "kH") and hasattr(pw, "hel_mag")

    arr = np.asarray(pw.hel_mag if use_hel else pw.mag)  # (nt, nk)
    ksafe = np.where(k <= 0, np.nan, k)                  # avoid non-positive k

    peaks = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        spec = np.abs(arr[i])
        y = ksafe * spec  
        if np.all(~np.isfinite(y)):
            continue
        idx = np.nanargmax(y)
        peaks[i] = k[idx] if np.isfinite(y[idx]) else np.nan
    return t, peaks

# --- shared helper (put near other helpers) -------------------------------
def _spectrum_y(arr_i: np.ndarray, k: np.ndarray, field: str, ymode: str) -> np.ndarray:
    """
    Match animate_spectrum(): for 'mag' -> k*|E| if ymode='kE';
    for 'hel_mag' -> k*|H| if ymode='kH'; otherwise plot |spec|.
    """
    s = np.abs(arr_i)
    if field == "hel_mag" and ymode == "kE":
        ymode = "kH"  # same sensible default as in animate_spectrum
    if ymode in ("kE", "kH"):
        return k * s
    return s

def _ylabel_for(field: str, ymode: str) -> str:
    if field == "mag":
        return r"$\mathrm{d}\rho_B/\mathrm{d}\ln k ~~ [E_*\,l_*^{-3}]$" if ymode == "kE" else r"$E_{\rm mag}(k)$"
    if field == "hel_mag":
        return r"$|\mathrm{d}h_M/\mathrm{d}\ln k| ~~ [E_*\,l_*^{-2}]$" if ymode == "kH" else r"$|H_{\rm mag}(k)|$"
    return field


# --- Slides export helpers ----------------------------------------------------
def export_for_slides(fig, base_no_ext, slide_inches=(10.0, 5.625), png_dpi=300):
    """
    Export a figure for Google Slides (SVG + PNG) and a PDF for print.
    - SVG: mathtext (no usetex) so text remains vector/selectable in Slides
    - PNG: high-DPI fallback if your Slides workspace rasterizes SVG
    - PDF: with your current cfg.use_tex setting (great for print)
    """
    import matplotlib as mpl

    # Size to match a 16:9 slide; avoids internal rescaling in Slides
    fig.set_size_inches(*slide_inches, forward=True)

    # Always make a PDF with current rcParams (often usetex=True for print)
    fig.savefig(base_no_ext + ".pdf", bbox_inches="tight")

    # Re-render just for Slides assets with TeX off and vector-friendly fonts
    with mpl.rc_context({
        "text.usetex": False,      # use mathtext so SVG keeps text as text
        "svg.fonttype": "none",    # do not convert fonts to paths
        "pdf.fonttype": 42,        # (no effect on SVG, but fine to keep)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    }):
        fig.savefig(base_no_ext + ".svg")
        fig.savefig(base_no_ext + ".png", dpi=png_dpi)


def export_video_mp4(anim, out_mp4, fps=24, width_px=1920, height_px=1080, bitrate=8000):
    """
    Save animation as H.264 MP4 sized to the slide (16:9). If your figure hasn't
    been sized yet, set it before creating the FuncAnimation:
        fig.set_size_inches(10.0, 5.625)  # 1920x1080 at dpi=192
    """
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=fps, bitrate=bitrate, codec="libx264")
    # dpi chosen so inches * dpi ≈ target pixels
    # (e.g., 10in * 192 dpi = 1920 px; 5.625in * 192 dpi = 1080 px)
    dpi = int(width_px / 10.0)
    anim.save(out_mp4, writer=writer, dpi=dpi)


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
   # x0, x1 = t[1], t[10]    # choose points within your t range
   # y0 = 1e2                 # adjust y0 for vertical placement
   # y1 = 1e0
   # fig, ax = plt.subplots(figsize=(6, 4))
   # # slope 1: y ∝ t
   # ax.plot([x0, x1], [y0, y0 * (x1/x0)], 'k--', lw=1)

   # # slope 2: y ∝ t²
   # ax.plot([x0, x1], [y1, y1 * (x1/x0)**2], 'k-.', lw=1)

   # # Optional: annotate them
   # ax.text(x1*1.1, y0 * (x1/x0), r"$\propto t$", fontsize=9, va="bottom")
   # ax.text(x1*1.1, y1 * (x1/x0)**2, r"$\propto t^2$", fontsize=9, va="bottom")
    ax.loglog(t, np.abs(mu5), "-x", label=r"$\tilde{\mu}_{5} ~~ [l_{*}^{-1}]$")
    ax.loglog(t, np.abs(S5_over_G), "--", label=r"$\tilde{S}_5 / \Gamma_{5} ~~ [l_{*}^{-1}]$")
    ax.loglog(t, np.abs(lam * eta * (-JBm))/gamma, label = r"$\eta\lambda J\cdot B/\Gamma_{5}$" )
    
    ax.set_ylim(cfg.mu5_ymin, cfg.mu5_ymax)

    add_time_vlines(ax, compute_time_markers(ts, p))
    ax.set_xlabel("conformal time :~~ $t~~ [t_*]$",fontsize =12)
    ax.set_ylabel("chiral chemical potential",fontsize =12)
    ax.set_title(run)
    ax.legend()
    export_for_slides(fig, fig_path(cfg.FIG_DIR, run, "ts")[:-4])  # strip ".pdf"

 
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
    #ax.loglog(t, np.abs(ts.abm), 'o-', label=r"$h_M ~~ [E_*l_*^{-2}] $")
    add_time_vlines(ax,compute_time_markers(ts,p))
    ax.set_xlabel(r"conformal time :~~ $t~~ [t_*]$",fontsize =12)
    ax.set_ylabel(r"magnetic field")
    ax.set_title(run)
    ax.legend()
    export_for_slides(fig, fig_path(cfg.FIG_DIR, run, "Brms")[:-4])  # strip ".pdf"

    plt.close(fig)

def _time_colormap(t: np.ndarray, log_t: bool, cmap: str):
    tt = np.copy(t)
    # ensure strictly positive when log-scaling
    if log_t:
        min_pos = np.nanmin(tt[tt > 0]) if (tt > 0).any() else 1.0
        tt = np.where(tt > 0, tt, min_pos)
        tt = np.log10(tt)
    norm = mpl.colors.Normalize(vmin=np.nanmin(tt), vmax=np.nanmax(tt)) # type: ignore
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # type: ignore
    return tt, sm

def plot_alltimes_spectrum(
    pw, cfg: Config, k_scale: float, run: str, field: str, step: int = 1,
    k_markers: Optional[List[Tuple[float, str]]] = None,
    slope_exponents: Optional[List[float]] = None
) -> None:
    """Static overlay of spectra at all times, aligned with video settings."""
    if not hasattr(pw, field):
        print(f"[{run}] no field '{field}' in power.dat")
        return

    # data
    k = np.asarray(pw.krms) * k_scale
    arr = np.asarray(getattr(pw, field))  # (nt, nk)
    t = np.asarray(pw.t) - cfg.t_offset

    # colormap in log_t if requested – same as video’s legend
    tt_for_cmap, sm = _time_colormap(t, cfg.log_time_cmap, cfg.color_map)

    # choose ymode to mirror animate_spectrum default
    ymode = cfg.anim_y
    if field == "hel_mag" and ymode == "kE":
        ymode = "kH"

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(0, arr.shape[0], 1):
        y = _spectrum_y(arr[i], k, field, ymode)
        # be robust to NaNs/negatives
        valid = (k > 0) & np.isfinite(k) & np.isfinite(y) & (y > 0)
        if valid.sum() < 2:
            continue
        ax.loglog(k[valid][::step], y[valid][::step], lw=0.9, color=sm.to_rgba(tt_for_cmap[i]))

    if k_markers:
        add_k_vlines(ax, k_markers)

    # optional slope guides: anchor to first valid spectrum
    if slope_exponents:
        ref = np.nan_to_num(np.abs(arr[0]), nan=0.0, posinf=0.0, neginf=0.0)
        # add_slope_guides(ax, k, ref, exponents=slope_exponents, color="0.45")

    ax.set_xlabel(r"$k~~[l_*^{-1}]$")
    ax.set_ylabel(_ylabel_for(field, ymode))
    # <<< match video ranges exactly >>>
    ax.set_xlim(cfg.anim_xmin * k_scale, cfg.anim_xmax * k_scale)
    ax.set_ylim(cfg.anim_ymin, cfg.anim_ymax)

    ax.set_title(f"{run}: {field} (all times)")
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(t/t_*)$" if cfg.log_time_cmap else r"$t$")

    export_for_slides(fig, fig_path(cfg.FIG_DIR, run, f"{field}_alltimes")[:-4])
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
        export_for_slides(fig, fig_path(cfg.FIG_DIR, run, "spectra_final")[:-4])  # strip ".pdf"

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
    export_for_slides(fig, fig_path(cfg.FIG_DIR, run, "helicity_fraction_alltimes")[:-4])  # strip ".pdf"

    plt.close(fig)

# -----------------------------
# 5) Animations 
# -----------------------------
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def animate_spectrum(
    pw, ts, p2, cfg: Config, run: str, field: str, k_scale: float, out_basename: str
) -> None:
    """
    Animate P(k,t) with correctly-tracked moving markers+labels for
    k_CPI(t)=|mu5|(t) and k_peak(t). Labels use a blended transform so X
    follows data coords while Y hugs the top of the axes.
    """
    if pw is None or not hasattr(pw, field):
        print(f"[{run}] no field '{field}' in power.dat for animation")
        return

    # ---------------- Data ----------------
    k   = np.asarray(pw.krms) * k_scale       # (nk,)
    arr = np.asarray(getattr(pw, field))      # (nt, nk)
    t_pw = np.asarray(pw.t)                   # (nt,)

    ymode = cfg.anim_y
    # sensible default for helicity
    if field == "hel_mag" and ymode == "kE":
        ymode = "kH"

    def spec_y(i):
        s = np.abs(arr[i])
        if ymode in ("kE", "kH"):
            return k * s
        return s

    # Time subsampling
    it = np.arange(0, arr.shape[0], max(1, cfg.anim_stride))
    if it.size == 0:
        print(f"[{run}] nothing to animate for {field}")
        return

    # Fixed theory markers
    k_fixed = compute_k_markers(p2) if p2 is not None else []

    # Moving markers
    tt_kc, kc = kcpi_series(ts, k_scale)               # CPI = |mu5|
    tt_kp, kp = kpeak_series(pw, k_scale, ymode=ymode) # spectral peak

    # Safe linear interpolants
    def _interp_builder(tt, yy):
        tt = np.asarray(tt); yy = np.asarray(yy)
        mask = np.isfinite(tt) & np.isfinite(yy)
        if mask.sum() < 2:
            return lambda t: np.nan
        order = np.argsort(tt[mask])
        x = tt[mask][order]; y = yy[mask][order]
        def f(tnow):
            if not np.isfinite(tnow) or tnow < x[0] or tnow > x[-1]:
                return np.nan
            return float(np.interp(tnow, x, y))
        return f

    kc_at = _interp_builder(tt_kc, kc)
    kp_at = _interp_builder(tt_kp, kp)

    # Only discrete modes that ever have finite data
    finite_cols_any_time = np.any(np.isfinite(arr), axis=0)
    valid_k_grid = (k > 0) & np.isfinite(k) & finite_cols_any_time
    if not np.any(valid_k_grid):
        print(f"[{run}] no valid k grid for {field}")
        return

    # ---------------- Figure ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    line, = ax.loglog([], [], lw=1.5, color="C0",
                      marker='x', markersize=4,
                      markerfacecolor='none', markeredgewidth=0.9)
    title = ax.set_title(f"{run}: {field} spectrum")

    # Fixed vlines
    fixed_artists = []
    for x, lab in (k_fixed or []):
        if np.isfinite(x):
            fixed_artists += [
                ax.axvline(x, color="0.65", ls="--", lw=0.8),
                ax.text(x, 0.9, lab, rotation=90, va="top",
                        transform=ax.get_xaxis_transform(), fontsize=8)
            ]

    # Moving vlines
    v_kc = ax.axvline(np.nan, color="C3", ls="-.", lw=1.0, alpha=0.9)
    v_kp = ax.axvline(np.nan, color="C2", ls=":",  lw=1.2, alpha=0.9)

    # Labels that track vlines: X in data coords, Y in axes coords
    data_x_axes_y = mtransforms.blended_transform_factory(ax.transData, ax.get_xaxis_transform())
    # tiny rightward pixel offset to avoid overlap with the vline
    offset = mtransforms.ScaledTranslation(4/72, 0, fig.dpi_scale_trans)
    bbox_kw = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6)

    txt_kc = ax.text(np.nan, 0.93, r"$k_{\rm CPI}$",
                     transform=data_x_axes_y + offset,
                     fontsize=9, color="C3", ha="left", va="bottom",
                     clip_on=True, bbox=bbox_kw)
    txt_kp = ax.text(np.nan, 0.93, r"$k_{\rm peak}$",
                     transform=data_x_axes_y + offset,
                     fontsize=9, color="C2", ha="left", va="bottom",
                     clip_on=True, bbox=bbox_kw)

    ax.set_xlabel(r"$k\ [l_*^{-1}]$")

    ylabels = {"kE": r"$\mathrm{d}\rho_B/\mathrm{d}\ln k  ~~ [E_* l_*^{-3}]$",
               "kH" : r"$|\mathrm{d}h_M/\mathrm{d}\ln k  ~~ [E_* l_*^{-2}]|$"}
               
    ax.set_xlabel(xlabel=r"$k~~[l_*^{-1}]$") 
    ax.set_ylabel(ylabels[ymode] )
    ax.grid(alpha=0.25)

    ax.set_xlim(cfg.anim_xmin * k_scale, cfg.anim_xmax * k_scale)
    ax.set_ylim(cfg.anim_ymin, cfg.anim_ymax)

    def _visible_on_axes(x):
        xmin, xmax = ax.get_xlim()
        return np.isfinite(x) and (xmin < x < xmax)

    def _maybe_flip_alignment(txt, xdata):
        # Keep label inside if near the right edge
        xmin, xmax = ax.get_xlim()
        if np.isfinite(xdata) and xdata > 0.95 * xmax:
            txt.set_ha("right")
            txt.set_transform(data_x_axes_y + mtransforms.ScaledTranslation(-4/72, 0, fig.dpi_scale_trans))
        else:
            txt.set_ha("left")
            txt.set_transform(data_x_axes_y + offset)

    def init():
        line.set_data([], [])
        v_kc.set_xdata([np.nan, np.nan]); v_kp.set_xdata([np.nan, np.nan])
        txt_kc.set_position((np.nan, 0.93)); txt_kc.set_visible(False)
        txt_kp.set_position((np.nan, 0.93)); txt_kp.set_visible(False)
        return (line, v_kc, v_kp, txt_kc, txt_kp, title, *fixed_artists)

    def update(frame_idx):
        i = it[frame_idx]
        y_full = spec_y(i)

        valid_i = valid_k_grid & np.isfinite(y_full) & (y_full > 0)
        line.set_data(k[valid_i], y_full[valid_i])

        t_now = t_pw[i]

        # CPI marker + label
        kc_now = kc_at(t_now)
        if _visible_on_axes(kc_now):
            v_kc.set_xdata([kc_now, kc_now])
            txt_kc.set_position((kc_now, 0.93))
            _maybe_flip_alignment(txt_kc, kc_now)
            txt_kc.set_visible(True)
        else:
            v_kc.set_xdata([np.nan, np.nan])
            txt_kc.set_visible(False)

        # Peak marker + label
        kp_now = kp_at(t_now)
        if _visible_on_axes(kp_now):
            v_kp.set_xdata([kp_now, kp_now])
            txt_kp.set_position((kp_now, 0.93))
            _maybe_flip_alignment(txt_kp, kp_now)
            txt_kp.set_visible(True)
        else:
            v_kp.set_xdata([np.nan, np.nan])
            txt_kp.set_visible(False)

        title.set_text(f"{run}: {field}  (t = {t_now:.3g})")
        return (line, v_kc, v_kp, txt_kc, txt_kp, title, *fixed_artists)

    anim = FuncAnimation(fig, update, init_func=init, frames=len(it),
                         interval=1000/cfg.anim_fps, blit=False)

    # ---------------- Save ----------------
    os.makedirs(cfg.VID_DIR, exist_ok=True)
    mp4_path = os.path.join(cfg.VID_DIR, f"{out_basename}.mp4")
    gif_path = os.path.join(cfg.VID_DIR, f"{out_basename}.gif")




    try:
        # ensure the figure is 16:9 for crisp playback in Slides
        fig.set_size_inches(10.0, 5.625, forward=True)
        export_video_mp4(anim, mp4_path, fps=cfg.anim_fps, width_px=1920, height_px=1080, bitrate=8000)
        print("Saved:", mp4_path)
        #writer = FFMpegWriter(fps=cfg.anim_fps, metadata={"artist": "CME pipeline"})
        #anim.save(mp4_path, writer=writer, dpi=180)
        #print("Saved:", mp4_path)
    except Exception as e:
        print(f"[{run}] ffmpeg unavailable ({e}); falling back to GIF.")
        writer = PillowWriter(fps=cfg.anim_fps)
        anim.save(gif_path, writer=writer, dpi=150)
        print("Saved:", gif_path)

    plt.close(fig)

def animate_helicity_fraction(
    pw, ts, p2, cfg: Config, run: str, k_scale: float, out_basename: str
) -> None:
    """
    Animate helicity fraction hfrac(k,t) = (k H)/(2E) with:
    """
    # ----- data presence checks -----
    if pw is None or not (hasattr(pw, "mag") and hasattr(pw, "hel_mag")):
        print(f"[{run}] need mag & hel_mag in power.dat for helicity-fraction animation")
        return

    # ----- spectra & grids -----
    k     = np.asarray(pw.krms) * k_scale     # (nk,)
    EmAll = np.asarray(pw.mag)                # (nt, nk)
    HmAll = np.asarray(pw.hel_mag)            # (nt, nk)
    t_pw  = np.asarray(pw.t)                  # (nt,)

    # time subsampling
    it = np.arange(0, EmAll.shape[0], max(1, cfg.anim_stride))
    if it.size == 0:
        print(f"[{run}] nothing to animate for helicity fraction")
        return

    finite_cols_any_time = np.any(np.isfinite(EmAll) | np.isfinite(HmAll), axis=0)
    
    valid_k_grid = (k > 0) & np.isfinite(k) & finite_cols_any_time
    if not np.any(valid_k_grid):
        print(f"[{run}] no valid k grid for helicity fraction")
        return

    # lock the x-grid for the entire animation
    k_fixed_x = k[valid_k_grid]
    k_grid_mark = k_fixed_x  # keep your tick markers consistent too



    if not np.any(valid_k_grid):
        print(f"[{run}] no valid k grid for helicity fraction")
        return
    k_grid_mark = k[valid_k_grid]

    # ----- fixed theory markers -----
    k_fixed = compute_k_markers(p2) if p2 is not None else []

    # ----- moving markers (no interpolation) -----
    # k_peak is already aligned with pw.t
    _, kp_series = kpeak_series(pw, k_scale, ymode="kE")
    # k_CPI comes from ts.t — map to pw.t by nearest neighbor (no interp)
    tt_kc, kc_raw = kcpi_series(ts, k_scale)
    if tt_kc.size >= 1 and kc_raw.size == tt_kc.size:
        # precompute nearest-neighbor map from each t_pw to an index in tt_kc
        idx_nn = np.searchsorted(tt_kc, t_pw).astype(int)
        idx_nn = np.clip(idx_nn, 0, tt_kc.size - 1)
        # choose closer of (idx_nn) vs (idx_nn-1) when applicable
        idx_nn = np.where(
            (idx_nn > 0) &
            (np.abs(tt_kc[idx_nn] - t_pw) > np.abs(tt_kc[idx_nn - 1] - t_pw)),
            idx_nn - 1,
            idx_nn,
        )
        kc_series_on_pw = kc_raw[idx_nn]
    else:
        kc_series_on_pw = np.full_like(t_pw, np.nan, dtype=float)

    # ----- global y-lims so nothing gets hidden -----
    def _hfrac_from_arrays(Em, Hm, kvec, eps):
        valid = (kvec > 0) & np.isfinite(kvec) & np.isfinite(Em) & np.isfinite(Hm) & (np.abs(Em) > 0)
        kv = kvec[valid]
        hfrac = 0.5 * kv * (Hm[valid]) / (Em[valid] + eps)  # NO CLIP
        return kv, hfrac

    ymin, ymax = -1.5, 1.5
   
    # ----- figure -----
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # show discrete k points with markers
    line, = ax.semilogx([], [], lw=1.2, color="C0", marker="o",
                        markersize=3.0, markerfacecolor="none", markeredgewidth=0.8)
    title = ax.set_title(f"{run}: helicity fraction")

    # fixed vlines
    fixed_artists = []
    for x, lab in (k_fixed or []):
        if np.isfinite(x):
            fixed_artists += [
                ax.axvline(x, color="0.65", ls="--", lw=0.8),
                ax.text(x, 0.92, lab, rotation=90, va="top",
                        transform=ax.get_xaxis_transform(), fontsize=8)
            ]

    # moving vlines (no time interpolation)
    v_kc = ax.axvline(np.nan, color="C3", ls="-.", lw=1.0, alpha=0.9)
    v_kp = ax.axvline(np.nan, color="C2", ls=":",  lw=1.2, alpha=0.9)

    # labels that track vlines: X=data coords, Y=axes coords
    data_x_axes_y = mtransforms.blended_transform_factory(ax.transData, ax.get_xaxis_transform())
    offset = mtransforms.ScaledTranslation(4/72, 0, fig.dpi_scale_trans)
    bbox_kw = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6)

    txt_kc = ax.text(np.nan, 0.93, r"$k_{\rm CPI}$",
                     transform=data_x_axes_y + offset,
                     fontsize=9, color="C3", ha="left", va="bottom",
                     clip_on=True, bbox=bbox_kw)
    txt_kp = ax.text(np.nan, 0.93, r"$k_{\rm peak}$",
                     transform=data_x_axes_y + offset,
                     fontsize=9, color="C2", ha="left", va="bottom",
                     clip_on=True, bbox=bbox_kw)

    ## tiny tick marks at every available k (in axes coords near the bottom)
    #k_ticks_tf = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    #for xk in k_grid_mark:
    #    ax.plot([xk, xk], [0.02, 0.06], color="0.8", lw=0.6,
    #            transform=k_ticks_tf, solid_capstyle="butt", zorder=0)

    ax.set_xlabel(r"$k~~[l_*^{-1}]$")
    ax.set_ylabel(r"helicity fraction $kh_M/2\rho_B$")
    ax.set_xlim(cfg.anim_xmin * k_scale, cfg.anim_xmax * k_scale)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)

    def _visible_on_axes(x):
        xmin, xmax = ax.get_xlim()
        return np.isfinite(x) and (xmin < x < xmax)

    def _maybe_flip_alignment(txt, xdata):
        xmin, xmax = ax.get_xlim()
        if np.isfinite(xdata) and xdata > 0.95 * xmax:
            txt.set_ha("right")
            txt.set_transform(data_x_axes_y + mtransforms.ScaledTranslation(-4/72, 0, fig.dpi_scale_trans))
        else:
            txt.set_ha("left")
            txt.set_transform(data_x_axes_y + offset)

    def init():
        line.set_data([], [])
        v_kc.set_xdata([np.nan, np.nan]); v_kp.set_xdata([np.nan, np.nan])
        txt_kc.set_position((np.nan, 0.93)); txt_kc.set_visible(False)
        txt_kp.set_position((np.nan, 0.93)); txt_kp.set_visible(False)
        return (line, v_kc, v_kp, txt_kc, txt_kp, title, *fixed_artists)

    def update(frame_idx):
        i = it[frame_idx]
        Em = np.asarray(EmAll[i], dtype=float)
        Hm = np.asarray(HmAll[i], dtype=float)

        Em = np.asarray(EmAll[i], dtype=float)
        Hm = np.asarray(HmAll[i], dtype=float)

        # slice on the same k’s every frame, coerce bad values to 0 (no drop)
        Em_i = np.nan_to_num(Em[valid_k_grid], nan=0.0, posinf=0.0, neginf=0.0)
        Hm_i = np.nan_to_num(Hm[valid_k_grid], nan=0.0, posinf=0.0, neginf=0.0)

        # constant-size fraction vector
        hfrac = 0.5 * k_fixed_x * Hm_i / (Em_i + cfg.eps)

        # sanity check (will raise immediately if something goes off the rails)
        assert hfrac.shape == k_fixed_x.shape, (
            f"[{run}] hfrac size {hfrac.shape} != k size {k_fixed_x.shape} at frame {i}"
        )

        line.set_data(k_fixed_x, hfrac) 


        # vertical markers at this exact frame (no interpolation)
        kc_now = kc_series_on_pw[i] if i < kc_series_on_pw.size else np.nan
        if _visible_on_axes(kc_now):
            v_kc.set_xdata([kc_now, kc_now]) # type: ignore
            txt_kc.set_position((kc_now, 0.93)) # type: ignore
            _maybe_flip_alignment(txt_kc, kc_now)
            txt_kc.set_visible(True)
        else:
            v_kc.set_xdata([np.nan, np.nan])
            txt_kc.set_visible(False)

        kp_now = kp_series[i] if i < kp_series.size else np.nan
        if _visible_on_axes(kp_now):
            v_kp.set_xdata([kp_now, kp_now]) # type: ignore
            txt_kp.set_position((kp_now, 0.93)) # type: ignore
            _maybe_flip_alignment(txt_kp, kp_now)
            txt_kp.set_visible(True)
        else:
            v_kp.set_xdata([np.nan, np.nan])
            txt_kp.set_visible(False)

        title.set_text(f"{run}: helicity fraction  (t = {t_pw[i]:.3g})")
        return (line, v_kc, v_kp, txt_kc, txt_kp, title, *fixed_artists)

    anim = FuncAnimation(fig, update, init_func=init, frames=len(it),
                         interval=1000/cfg.anim_fps, blit=False)

    # ----- save -----
    os.makedirs(cfg.VID_DIR, exist_ok=True)
    mp4_path = os.path.join(cfg.VID_DIR, f"{out_basename}.mp4")
    gif_path = os.path.join(cfg.VID_DIR, f"{out_basename}.gif")
   
    try:
        fig.set_size_inches(10.0, 5.625, forward=True)
        export_video_mp4(anim, mp4_path, fps=cfg.anim_fps, width_px=1920, height_px=1080, bitrate=8000)
        print("Saved:", mp4_path)

        #writer = FFMpegWriter(fps=cfg.anim_fps, metadata={"artist": "CME pipeline"})
        #anim.save(mp4_path, writer=writer, dpi=180)
        #print("Saved:", mp4_path)
    except Exception as e:
        print(f"[{run}] ffmpeg unavailable ({e}); falling back to GIF.")
        writer = PillowWriter(fps=cfg.anim_fps)
        anim.save(gif_path, writer=writer, dpi=150)
        print("Saved:", gif_path)

    plt.close(fig)



# -----------------------------
# 6) Pipeline driver
# -----------------------------
def run_pipeline(cfg: Config, sims_override=None) -> None:
    setup_style(cfg)
    sims = sims_override if sims_override is not None else discover_sims(cfg.ROOT)
    print("Detected runs:", [s.name for s in sims]) # type: ignore
    pow_all, params_all,params_all1, ts_all = read_everything(sims)
    # Optional: accumulate a small summary table
    summary_rows = []

    for sim in sims:
        name = sim.name # type: ignore
        pw  = pow_all.get(name)
        ts  = ts_all.get(name)
        par = params_all.get(name)
        par1 = params_all1.get(name)
        print(f"___________{par1.wav1}_______________________") # type: ignore
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
            plot_alltimes_spectrum(pw, cfg,par1.wav1, name,step=100, field="mag", k_markers=k_mks,slope_exponents=[2,3,4]) # type: ignore

        if pw is not None and hasattr(pw, "hel_mag") and cfg.make_hel_alltimes:
            plot_alltimes_spectrum(pw, cfg, par1.wav1,name,step=100, field="hel_mag",k_markers=k_mks,slope_exponents=[2,3,4]) # type: ignore

        if pw is not None and cfg.make_helicity_fraction:
            plot_helicity_fraction_alltimes(pw, cfg,par1.wav1, name, k_markers=k_mks) # type: ignore

        if pw is not None and cfg.make_final_spectra:
            plot_final_spectra_with_bound(pw, cfg,par1.wav1, name, k_markers=k_mks) # type: ignore

        # Summary metrics: final B_rms, final H_int, final B^2 ξ_M
        try:
            brms_f = float(ts.brms[-1]) # type: ignore
            brms_max = np.max(ts.brms) # type: ignore
        except Exception:
            brms_f = np.nan

        try:
            tH, Hts = compute_helicity_timeseries(pw) if pw is not None else (np.array([]), np.array([]))
            H_f = float(Hts[-1]) if Hts.size else np.nan
            H_max = np.max(Hts) if Hts.size else np.nan
        except Exception:
            H_f = np.nan

        try:
            tB2L, B2L, B2, Xi = compute_B2_lcoh_weighted(pw, k_scale=par1.wav1) if pw is not None else (np.array([]),)*4 # type: ignore
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
        

            # Animations
        if pw is not None:
            if cfg.make_anim_mag and hasattr(pw, "mag"):
                animate_spectrum(
                    pw, ts, par, cfg, name, field="mag",
                    k_scale=par1.wav1, out_basename=f"{name}_mag" # type: ignore
                )
            if cfg.make_anim_hel and hasattr(pw, "hel_mag"):
                animate_spectrum(
                    pw, ts, par, cfg, name, field="hel_mag",
                    k_scale=par1.wav1, out_basename=f"{name}_hel" # type: ignore
                )
            if cfg.make_anim_hel_frac and hasattr(pw, "hel_mag") and hasattr(pw, "mag"):
                animate_helicity_fraction(
                    pw, ts, par, cfg, name,
                    k_scale=par1.wav1, out_basename=f"{name}_hfrac" # type: ignore
                )


#    # Write a simple CSV summary
#    if summary_rows:
#        import csv
#        csv_path = os.path.join(cfg.FIG_DIR, "summary.csv")
#        with open(csv_path, "w", newline="") as f:
#            writer = csv.writer(f)
#            writer.writerow([
#                                "run",
#                                "lambda5", "source5", "eta", "nu", "diffmu5", "gammaf5",
#                                "Brms_final", "Hint_final", "B2_xi_final", "xi_final",
#                                "Brms_max", "Hint_max", "B2_xi_max", "xi_max"
#                            ])
#            writer.writerows(summary_rows)
#        print("Wrote summary:", csv_path)


# -----------------------------
# 7) Main
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
        print("Detected runs:", [s.name for s in sims_all]) # type: ignore
        sys.exit(0)

    # Apply filters (default: keep all)
    sims_sel = _filter_sims(sims_all,
                            only_runs=args.run,
                            like=args.like,
                            regex=args.regex)

    if not sims_sel:
        print("No runs matched your filter.")
        print("Available:", [s.name for s in sims_all]) # type: ignore
        sys.exit(1)

    # Run pipeline only on selected sims
    run_pipeline(cfg, sims_override=sims_sel)