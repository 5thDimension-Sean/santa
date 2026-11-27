"""Hero dashboard for the 'Source-Aware Frontier' notebook: the public race, where the community's savings landed,
and one real rewrite. Uses style / dag_of / rewrite_panel and the palette from viz.py."""
import datetime as _dt

def dashboard2(P, base, now, ours, fam, ladder, pub_hist, ours_hist, showcase, lb, n_out, n_src, base_label="20 Nov"):
    """P: puzzles; base: public frontier at the baseline snapshot; now: merged public frontier (outputs + sources);
    ours: our published programs; ladder: [(notebook, total)] single public notebooks; pub_hist: [(datetime, total)];
    ours_hist: [(datetime, lb)]; n_out / n_src: notebooks merged from output files / from source + datasets."""
    style()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(17, 18.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.1], hspace=0.45, wspace=0.65, top=0.88, bottom=0.04)
    b_tot = sum(len(base[k]) for k in P); n_tot = sum(len(now[k]) for k in P); o_tot = sum(len(ours[k]) for k in P)
    best_name, best_val = min(ladder, key=lambda x: x[1]) if ladder else ("-", n_tot)
    drop = {k: len(base[k]) - len(now[k]) for k in P}
    fig.text(0.125, 0.985, "Santa 2026 · The public race for the shortest Elf programs — merged from every source",
             fontsize=18, fontweight="bold", va="top")
    tiles = [(f"{lb:,}", "Kaggle LB score of this notebook"),
             (f"−{best_val - lb:,}", f"vs best single public notebook ({best_val:,})"),
             (f"−{b_tot - n_tot:,}", f"public frontier since {base_label} ({b_tot:,})"),
             (f"{n_out + n_src}", f"public notebooks merged ({n_src} via source / datasets)")]
    for i, (big, small) in enumerate(tiles):
        x = 0.125 + i * 0.205
        fig.text(x, 0.958, big, fontsize=30, fontweight="bold", color=BLUE if i in (0, 1) else INK, va="top")
        fig.text(x, 0.915, small, fontsize=10.5, color=INK2, va="top")

    # 1. the race over time
    ax = fig.add_subplot(gs[0, :3])
    if pub_hist:
        px = [t for t, _ in pub_hist]; py = [v for _, v in pub_hist]
        ax.step(px, py, where="post", color=MUTED, lw=1.6, ls="--", label="best public frontier (per-puzzle minimum)")
    if ours_hist:
        hx = [t for t, _ in ours_hist]; hy = [v for _, v in ours_hist]
        ax.step(hx, hy, where="post", color=BLUE, lw=2.0, label="our public notebooks (LB of each version)")
        ax.scatter(hx, hy, color=BLUE, s=9, zorder=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_title("The race: total instructions over time (KST)"); ax.set_ylabel("total instructions"); _comma(ax)
    ax.legend(loc="lower left")

    # 2. single-notebook ladder (zoomed)
    ax = fig.add_subplot(gs[0, 3:])
    lad = sorted(ladder, key=lambda x: x[1])[:9]
    names = [n for n, _ in lad] + ["merged public frontier", "this notebook"]
    vals = [v for _, v in lad] + [n_tot, lb]
    order = np.argsort(vals)[::-1]; names = [names[i] for i in order]; vals = [vals[i] for i in order]
    cols = [BLUE if n == "this notebook" else (AQUA if n == "merged public frontier" else "#b9cde6") for n in names]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.7)
    lo = min(vals) - 0.6 * (max(vals) - min(vals) + 20); hi = max(vals) + 0.25 * (max(vals) - lo)
    ax.set_xlim(lo, hi)
    ax.set_yticks(y); ax.set_yticklabels([n if len(n) < 44 else n[:42] + "…" for n in names], fontsize=8.6)
    for yi, v, n in zip(y, vals, names):
        ax.text(v + 0.01 * (hi - lo), yi, f"{v:,}", va="center", fontsize=8.8,
                color=INK if n == "this notebook" else INK2, fontweight="bold" if n == "this notebook" else None)
    ax.set_title("Single public notebooks vs the merge (zoomed)")
    ax.set_xlabel("total instructions over 1,000 puzzles (lower is better)"); ax.grid(axis="y", visible=False); _comma(ax, "x")

    # 3. where the community's savings landed
    ax = fig.add_subplot(gs[1, :4])
    ids = sorted(P); W = 50
    grid = np.full((len(ids) // W, W), np.nan)
    for i, k in enumerate(ids): grid[i // W, i % W] = drop[k]
    cm = ListedColormap(["#efeeea"] + SEQ[1:6])
    bounds = [-0.5, 0.5, 1.5, 2.5, 4.5, 9.5, max(10.5, np.nanmax(grid) + 0.5)]
    ax.imshow(grid, cmap=cm, norm=BoundaryNorm(bounds, cm.N), aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks(range(0, len(ids) // W, 5)); ax.set_yticklabels([f"{r * W + 1:04d}" for r in range(0, len(ids) // W, 5)], fontsize=8)
    ax.grid(False); ax.set_ylabel("puzzle id (rows of 50)")
    ax.set_title(f"The 1,000 puzzles — instructions removed by the community since {base_label}")
    labels = ["unchanged", "−1", "−2", "−3–4", "−5–9", "−10+"]
    ax.legend(handles=[mpatches.Patch(color=cm(i), label=l) for i, l in enumerate(labels)],
              loc="upper left", bbox_to_anchor=(0, -0.02), ncol=6, fontsize=9)

    # 4. by family
    ax = fig.add_subplot(gs[1, 4:])
    fams = ["independent", "mixed", "shared", "multiplicative", "tiny"]
    cnt = [sum(1 for k in P if fam[k] == f) for f in fams]
    sv = [sum(drop[k] for k in P if fam[k] == f) for f in fams]
    pct = [100 * sum(drop[k] for k in P if fam[k] == f) / max(1, sum(len(base[k]) for k in P if fam[k] == f)) for f in fams]
    ax.barh(range(len(fams))[::-1], sv, color=BLUE, height=0.6)
    for i, (v, c, q) in enumerate(zip(sv, cnt, pct)):
        ax.text(v, len(fams) - 1 - i, f"  −{v:,}  ({q:.1f}% of {c} puzzles)", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(fams))[::-1]); ax.set_yticklabels(fams)
    ax.set_xlim(0, max(sv + [1]) * 2.0)
    ax.set_title(f"Removed since {base_label}, by puzzle family*"); ax.grid(axis="y", visible=False)

    # 5. one real rewrite: the baseline public program vs today's frontier
    p = P[showcase]; d0, d1 = dag_of(p, base[showcase]), dag_of(p, now[showcase])
    ax = fig.add_subplot(gs[2, :3])
    rewrite_panel(ax, p, d0, d1, "old", f"{showcase} · public program on {base_label}: replaced part ({len(base[showcase])} instr.)")
    ax = fig.add_subplot(gs[2, 3:])
    rewrite_panel(ax, p, d0, d1, "new", f"{showcase} · today's frontier: the replacement ({len(now[showcase])} instr.)")
    handles = [mpatches.Patch(color="#eeede8", label="value kept from the program"), mpatches.Patch(color=BLUE, label="target"),
               mpatches.Patch(color=RED, label="removed instruction (op → value)"), mpatches.Patch(color=AQUA, label="new instruction (op → value)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(-0.1, -0.02), ncol=4, fontsize=9.5)
    fig.text(0.0, 0.005, "*family inferred from practice-set fingerprints (start-value size, target relations). "
             "Every program counted here was replayed by an exact referee.", fontsize=8.5, color=INK2)
    return fig
