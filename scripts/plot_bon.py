"""Best-of-N success vs N per BC dataset size, from scripts/summarize_bon.py's summary.json.

Built with the dataviz skill's method:
- Top: success rate vs N, one panel per dataset (PH, MH), one line per BC dataset size. Size is
  ordinal, so the lines take one hue (the reference blue ramp, steps 250..650; ordinal checks:
  monotone OKLab L, min adjacent dL 0.093 >= 0.06, light-end contrast 2.06:1 >= 2:1). Five series:
  legend, no end labels (the low MH lines end 0.03 apart and would collide).
- Bottom: gain over N=1 as small multiples (the five gain curves converge, so one panel each),
  paired per policy seed (same 128 start states for every N). One series per panel in slot-1 blue,
  solid zero baseline, only the peak labeled.
Error bars: std over the 3 policy seeds. Lines 2px, markers >= 8px with a 2px surface ring, solid
hairline grid, text in ink tokens only. The table view is summary.json / the markdown table.

    python scripts/plot_bon.py <summary.json> <out.png>
"""

import json
import sys

import numpy as np

RAMP = {10: "#86b6ef", 20: "#5598e7", 50: "#2a78d6", 100: "#1c5cab", 200: "#104281"}
SLOT1 = "#2a78d6"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SIZES = (10, 20, 50, 100, 200)
DPI = 150
PX = 72 / DPI  # points per pixel at this dpi


def series(s, ds, n, Ns):
    base = dict(zip(s[f"{ds}-{n}/bon1"]["seeds"], s[f"{ds}-{n}/bon1"]["success"]))
    out = {"mean": [], "std": [], "gain": [], "gain_std": []}
    for N in Ns:
        e = s[f"{ds}-{n}/bon{N}"]
        v = np.array(e["success"])
        g = np.array([x - base[sd] for sd, x in zip(e["seeds"], e["success"])])
        out["mean"].append(v.mean()); out["std"].append(v.std())
        out["gain"].append(g.mean()); out["gain_std"].append(g.std())
    return {k: np.array(v) for k, v in out.items()}


def style(ax, Ns):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns); ax.set_xticklabels([str(N) for N in Ns]); ax.minorticks_off()
    ax.grid(True, axis="y", color=GRID, lw=1 * PX, ls="-"); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(1 * PX)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def main():
    src, out = sys.argv[1], sys.argv[2]
    with open(src) as f:
        s = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = sorted({int(k.split("/bon")[1]) for k in s})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2})
    line = dict(lw=2 * PX * 1.5, marker="o", ms=10 * PX * 1.2, mec=SURFACE, mew=2 * PX * 1.5,
                capsize=2.5, elinewidth=1.2 * PX * 1.5, solid_capstyle="round", solid_joinstyle="round")

    fig = plt.figure(figsize=(15, 11.5), facecolor=SURFACE)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1], hspace=0.04)
    top.set_facecolor(SURFACE); bottom.set_facecolor(SURFACE)

    # --- success vs N
    axs = top.subplots(1, 2, sharey=True)
    for ax, ds in zip(axs, ("ph", "mh")):
        for n in SIZES:
            v = series(s, ds, n, Ns)
            ax.errorbar(Ns, v["mean"], yerr=v["std"], color=RAMP[n], ecolor=RAMP[n],
                        label=f"{n} demos", **line)
        style(ax, Ns)
        ax.set_ylim(0, 1); ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(f"Square {ds.upper()}", loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("N (chunks sampled per decision)", fontsize=9)
    axs[0].set_ylabel("Success rate", fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=RAMP[n], lw=line["lw"], marker="o", ms=line["ms"], mec=SURFACE,
                mew=line["mew"]) for n in SIZES]
    l = [f"{n} demos" for n in SIZES]
    top.legend(h, l, loc="upper right", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.98, 1.0),
               handlelength=1.6, title="BC dataset size", title_fontsize=9)
    top.suptitle("Success rate vs N", x=0.02, ha="left", y=1.0, fontsize=12, color=INK, weight="bold")

    # --- gain over N=1, small multiples
    bxs = bottom.subplots(2, 5, sharex=True, sharey=True)
    for r, ds in enumerate(("ph", "mh")):
        for c, n in enumerate(SIZES):
            ax = bxs[r, c]
            v = series(s, ds, n, Ns)
            ax.axhline(0, color=MUTED, lw=1.5 * PX * 1.5, zorder=1)
            ax.errorbar(Ns, v["gain"], yerr=v["gain_std"], color=SLOT1, ecolor=SLOT1, zorder=3, **line)
            k = int(np.argmax(v["gain"]))
            ax.annotate(f"+{v['gain'][k]:.2f}", (Ns[k], v["gain"][k] + v["gain_std"][k]),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=INK)
            style(ax, Ns)
            ax.set_ylim(-0.1, 0.2); ax.set_yticks([-0.1, 0, 0.1, 0.2])
            ax.set_title(f"{ds.upper()}, {n} demos", loc="left", fontsize=9.5, color=INK, pad=4)
            if r == 1:
                ax.set_xlabel("N", fontsize=9)
        bxs[r, 0].set_ylabel("Gain over N=1", fontsize=9)
    bottom.suptitle("Gain over N=1, paired on the same 128 start states (peak labeled)", x=0.02,
                    ha="left", y=1.02, fontsize=12, color=INK, weight="bold")

    fig.text(0.02, -0.01, "Best-of-N with the no-memory MC critic: N SDE chunks per decision, scored by "
             "the critic's ensemble mean, argmax executed. 128 episodes per policy, 3 policy seeds per "
             "point; error bars = std over seeds.", fontsize=8.5, color=INK2, ha="left")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
