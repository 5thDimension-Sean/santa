"""Hero dashboard for the 'Fresh-Program Re-Search' notebook: which search passes still find savings on a converged
frontier, the last 30 hours of the public race, where those savings landed, and one real rewrite.
Uses style / dag_of / rewrite_panel / _comma and the palette from viz.py."""

def dashboard3(P, base, now, ours, fam, stats, race, showcase, lb, best_single, base_label="22 Nov 09:00"):
    """P: puzzles; base: public frontier at the baseline; now: merged public frontier; ours: our programs;
    stats: [(pass kind, gain, hours, runs)]; race: [(datetime, frontier, best_single, ours)]; best_single: (name, total)."""
    style()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(17, 18.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.1], hspace=0.45, wspace=0.65, top=0.88, bottom=0.04)
    b_tot = sum(len(base[k]) for k in P); n_tot = sum(len(now[k]) for k in P)
    drop = {k: len(base[k]) - len(now[k]) for k in P}
    pub = [r for r in stats if "ours" not in r[0]]   # public engines only
    agg = lambda pre, rows: sum(g for k, g, h, _ in rows if k.startswith(pre)) / max(1e-9, sum(h for k, g, h, _ in rows if k.startswith(pre)))
    fresh, conv = agg("fresh", pub), agg(("converged", "hot"), pub)   # totals over all fresh passes vs all converged / hot-list passes
    own = agg("annealing", stats)
    fig.text(0.125, 0.985, "Santa 2026 · What still saves instructions on a converged frontier",
             fontsize=18, fontweight="bold", va="top")
    tiles = [(f"{lb:,}", "Kaggle LB score of this notebook"),
             (f"−{best_single[1] - lb:,}", f"vs best single public notebook ({best_single[1]:,})"),
             (f"−{own:.0f}/h" if own > 0 else f"×{fresh / max(conv, 1e-9):.0f}", f"our annealing (public: {fresh:.0f}/h fresh, {conv:.1f}/h rest)" if own > 0 else "per search-hour: fresh vs converged"),
             (f"−{b_tot - n_tot:,}", f"public frontier since {base_label} ({b_tot:,})")]
    for i, (big, small) in enumerate(tiles):
        x = 0.125 + i * 0.205
        fig.text(x, 0.958, big, fontsize=30, fontweight="bold", color=BLUE if i in (0, 2) else INK, va="top")
        fig.text(x, 0.915, small, fontsize=10.5, color=INK2, va="top")

    # 1. savings per hour by search pass
    ax = fig.add_subplot(gs[0, :3])
    st = sorted(stats, key=lambda r: r[1] / max(r[2], 1e-9))
    y = np.arange(len(st))
    vals = [g / max(h, 1e-9) for _, g, h, _ in st]
    ax.barh(y, vals, color=[BLUE if "ours" in k else AQUA if k.startswith("fresh") else "#b9cde6" for k, *_ in st], height=0.65)
    ax.set_yticks(y); ax.set_yticklabels([k.replace(" · ", "\n") for k, *_ in st], fontsize=8.6)
    for yi, (k, g, h, r) in zip(y, st):
        ax.text(g / max(h, 1e-9), yi, f"  {g / max(h, 1e-9):.1f}/h  (−{int(g)} in {h:.1f} h, {int(r)} runs)", va="center", fontsize=8.6, color=INK2)
    ax.set_xlim(0, max(vals + [1]) * 1.9)
    ax.set_title("Instructions saved per hour of search, by pass type"); ax.set_xlabel("instructions saved per wall-clock hour (20 cores)")
    ax.grid(axis="y", visible=False)

    # 2. the race over the last 30 hours
    ax = fig.add_subplot(gs[0, 3:])
    if race:
        t = [r[0] for r in race]
        ax.step(t, [r[1] for r in race], where="post", color=MUTED, lw=1.4, ls="--", label="merged public frontier")
        ax.step(t, [r[2] for r in race], where="post", color=ORANGE, lw=1.6, label="best single public notebook")
        ax.step(t, [r[3] for r in race], where="post", color=BLUE, lw=2.0, label="our notebook (LB)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_title(f"The race since {base_label} (KST)"); ax.set_ylabel("total instructions"); _comma(ax)
    ax.legend(loc="upper right")

    # 3. where the savings of the last 30 hours landed
    ax = fig.add_subplot(gs[1, :4])
    ids = sorted(P); W = 50
    grid = np.full((len(ids) // W, W), np.nan)
    for i, k in enumerate(ids): grid[i // W, i % W] = drop[k]
    cm = ListedColormap(["#efeeea"] + SEQ[1:6])
    bounds = [-0.5, 0.5, 1.5, 2.5, 4.5, 9.5, max(10.5, np.nanmax(grid) + 0.5)]
    ax.imshow(grid, cmap=cm, norm=BoundaryNorm(bounds, cm.N), aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks(range(0, len(ids) // W, 5)); ax.set_yticklabels([f"{r * W + 1:04d}" for r in range(0, len(ids) // W, 5)], fontsize=8)
    ax.grid(False); ax.set_ylabel("puzzle id (rows of 50)")
    ax.set_title(f"The 1,000 puzzles — instructions removed by everyone since {base_label}")
    labels = ["unchanged", "−1", "−2", "−3–4", "−5–9", "−10+"]
    ax.legend(handles=[mpatches.Patch(color=cm(i), label=l) for i, l in enumerate(labels)], loc="upper left", bbox_to_anchor=(0, -0.02), ncol=6, fontsize=9)

    # 4. concentration: a few puzzles carry most of the savings
    ax = fig.add_subplot(gs[1, 4:])
    d = np.array(sorted((v for v in drop.values() if v > 0), reverse=True))
    if len(d):
        cx = np.arange(1, len(d) + 1) / len(P) * 100; cy = np.cumsum(d) / d.sum() * 100
        ax.plot(cx, cy, color=BLUE, lw=2.2)
        for q in (50, 80):
            j = int(np.searchsorted(cy, q)); ax.scatter([cx[j]], [cy[j]], color=BLUE, s=22, zorder=3)
            ax.text(cx[j] + 1.5, cy[j] - 4, f"{q}% of the savings\nin {cx[j]:.0f}% of the puzzles", fontsize=8.8, color=INK2)
        ax.set_xlim(0, max(cx) * 1.25); ax.set_ylim(0, 105)
    ax.set_title(f"Savings since {base_label} are concentrated"); ax.set_xlabel("% of all 1,000 puzzles (largest drops first)")
    ax.set_ylabel("% of all instructions removed")

    # 5. one real rewrite
    p = P[showcase]; ref = now if any(len(ours[k]) < len(now[k]) for k in [showcase]) else base
    newer = ours if ref is now else now
    d0, d1 = dag_of(p, ref[showcase]), dag_of(p, newer[showcase])
    lab0 = "best public program" if ref is now else f"public program on {base_label}"
    lab1 = "our re-search" if ref is now else "today's frontier"
    ax = fig.add_subplot(gs[2, :3])
    rewrite_panel(ax, p, d0, d1, "old", f"{showcase} · {lab0}: replaced part ({len(ref[showcase])} instr.)")
    ax = fig.add_subplot(gs[2, 3:])
    rewrite_panel(ax, p, d0, d1, "new", f"{showcase} · {lab1}: the replacement ({len(newer[showcase])} instr.)")
    handles = [mpatches.Patch(color="#eeede8", label="value kept from the program"), mpatches.Patch(color=BLUE, label="target"),
               mpatches.Patch(color=RED, label="removed instruction (op → value)"), mpatches.Patch(color=AQUA, label="new instruction (op → value)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(-0.1, -0.02), ncol=4, fontsize=9.5)
    fig.text(0.0, 0.005, "Pass statistics come from our search logs (wall-clock hours on 20 shared CPU cores). "
             "Every program counted here was replayed by an exact referee.", fontsize=8.5, color=INK2)
    return fig
