#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import mplhep as hep
import numpy as np
import matplotlib.pyplot as plt


def load_limits(json_path):
    with open(json_path, "r") as f:
        raw = json.load(f)

    masses = sorted(float(m) for m in raw.keys())

    def get(k):
        vals = []
        for m in masses:
            vals.append(raw[str(m)].get(k, np.nan))
        return np.array(vals, dtype=float)

    return {
        "mass": np.array(masses, dtype=float),
        "exp0": get("exp0"),
        "exp+1": get("exp+1"),
        "exp-1": get("exp-1"),
        "exp+2": get("exp+2"),
        "exp-2": get("exp-2"),
        "obs":  get("obs"),
    }


def densify(x, y, points_per_segment=50):
    """Return (x_dense, y_dense) using linear interpolation with many points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Build dense x between each consecutive pair
    xd = []
    for i in range(len(x) - 1):
        seg = np.linspace(x[i], x[i+1], points_per_segment, endpoint=False)
        xd.append(seg)
    xd.append(np.array([x[-1]]))
    x_dense = np.concatenate(xd)

    # Interpolate (ignore NaNs by masking contiguous finite sections)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        y_dense = np.interp(x_dense, x[mask], y[mask])
    else:
        y_dense = np.full_like(x_dense, np.nan, dtype=float)
    return x_dense, y_dense


def plot_limits(data, args):
    m = data["mass"]
    exp0 = data["exp0"]
    exp_p1, exp_m1 = data["exp+1"], data["exp-1"]
    exp_p2, exp_m2 = data["exp+2"], data["exp-2"]
    obs = data["obs"]

    # Densify for smooth bands/lines
    x_e, e0 = densify(m, exp0, args.smooth)
    _, e_p1 = densify(m, exp_p1, args.smooth)
    _, e_m1 = densify(m, exp_m1, args.smooth)
    _, e_p2 = densify(m, exp_p2, args.smooth)
    _, e_m2 = densify(m, exp_m2, args.smooth)
    x_o, o  = densify(m, obs,   args.smooth)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=140)
    hep.style.use("CMS")
    hep.cms.label("Preliminary",data=True,lumi=1.1, year=2018,fontsize=12)

    # 95% band (±2σ) — yellow
    if np.isfinite(e_p2).any() and np.isfinite(e_m2).any():
        ax.fill_between(x_e, e_m2, e_p2, alpha=0.7, label="±2σ Expected",
                        facecolor="#FFD54F", edgecolor="none")

    # 68% band (±1σ) — green
    if np.isfinite(e_p1).any() and np.isfinite(e_m1).any():
        ax.fill_between(x_e, e_m1, e_p1, alpha=0.9, label="±1σ Expected",
                        facecolor="#33cc33", edgecolor="none")

    # Expected — red line
    ax.plot(x_e, e0, lw=2.0, color="red", label="Expected")

    # Observed — black with markers
    if np.isfinite(o).any():
        ax.plot(x_o, o, lw=2.0, color="black", label="Observed")
        ax.plot(m, obs, ls="none", marker="o", ms=4.5, color="black")

    # Axes & style
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_xlim([300.0,3000.0])
    #ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="best",fontsize=12)
    ax.tick_params(direction="in", top=True, right=True)

    if args.logy:
        ax.set_yscale("log")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description="Make a smooth limits plot from JSON.")
    p.add_argument("json", help="Path to limits JSON.")
    p.add_argument("-o", "--output", default="limits.png", help="Output image path.")
    p.add_argument("--xlabel", default=r"$m$ (GeV)", help="X-axis label.")
    p.add_argument("--ylabel", default=r"95% CL upper limit", help="Y-axis label.")
    p.add_argument("--title", default="Limits", help="Plot title.")
    p.add_argument("--logy", action="store_true", help="Use log scale on Y.")
    p.add_argument("--smooth", type=int, default=60,
                   help="Points per mass interval for smooth bands (default: 60).")
    args = p.parse_args()

    data = load_limits(args.json)
    plot_limits(data, args)


if __name__ == "__main__":
    main()
