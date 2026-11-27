"""Two figures for the 'Fresh-Program Re-Search' notebook: how triwalk sees a program (value set + triples), and the live annealing demo."""

def prog_values(p, prog):
    regs = list(p.starts) + [None] * (p.registers - len(p.starts)); out = []
    for op, i, j, d in prog:
        v = OPF[op](regs[i], regs[j]); regs[d] = v; out.append(v)
    return out

def value_triples(V):
    V = sorted(V); H = set(V); out = []
    for i, a in enumerate(V):
        for b in V[i:]:
            if a + b in H: out.append((a, b, a + b, "+"))
            if a > 1 and a * b in H: out.append((a, b, a * b, "×"))
    return out

def triple_layout(nodes, tris, seed=7, iters=2500):
    """Left-right position follows the value's size (log scale); a small force layout (values and triples repel each other) spreads the rest."""
    rng = np.random.default_rng(seed); n, m = len(nodes), len(tris); ix = {v: i for i, v in enumerate(nodes)}
    lg = np.log10(np.array(nodes, float)); x0 = (lg - lg.min()) / max(1e-9, lg.max() - lg.min()) * 15
    X = np.c_[x0 + rng.normal(0, .3, n), rng.uniform(-4, 4, n)]
    E = np.array([[ix[t[0]], ix[t[1]], ix[t[2]]] for t in tris]); M = X[E].mean(1) + rng.normal(0, .2, (m, 2))
    R = np.r_[np.full(n, 0.85), np.full(m, 0.42)]
    for it in range(iters):
        Z = np.r_[X, M]; D = Z[:, None] - Z[None]; D[..., 0] *= 0.75; d = np.sqrt((D ** 2).sum(-1)) + 1e-6
        F = (np.clip(R[:, None] + R[None] + 0.35 - d, 0, None)[..., None] * D / d[..., None]).sum(1) * 0.55
        for j, e in enumerate(E):
            F[n + j] += (X[e].mean(0) - M[j]) * 0.5
            F[e] += (M[j] - X[e]) * 0.016
        F[:n, 0] += (x0 - X[:, 0]) * 0.16; F[:n, 1] -= X[:, 1] * 0.004
        Z = Z + F * 0.35 * (1 - it / iters) ** 0.4; X, M = Z[:n], Z[n:]
        X[:, 1] = np.clip(X[:, 1], -4.6, 4.6)
    return {v: X[ix[v]] for v in nodes}, {t: M[j] for j, t in enumerate(tris)}

def triple_figure(p, before, after, pid):
    """One real improvement, drawn the way triwalk sees it."""
    style()
    B, A = set(prog_values(p, before)), set(prog_values(p, after)); S = set(p.starts); T = set(p.targets)
    nodes = sorted(S | B | A); pos, tpos = triple_layout(nodes, value_triples(S | B | A))
    xs = [q[0] for q in pos.values()]; ys = [q[1] for q in pos.values()]
    fig, axs = plt.subplots(1, 2, figsize=(17, 6.8))
    for ax, V, title, mark, col in ((axs[0], B, f"Before: {len(before)} instructions", B - A, RED), (axs[1], A, f"After: {len(after)} instructions", A - B, GREEN)):
        tr = value_triples(S | V)
        for a, b, c, op in tr:
            m = tpos[(a, b, c, op)]; hot = bool({a, b, c} & mark); ec = col if hot else (BLUE if op == "+" else ORANGE)
            for v in {a, b, c}: ax.plot([m[0], pos[v][0]], [m[1], pos[v][1]], color=ec, lw=1.6 if hot else 1.0, alpha=0.9 if hot else 0.45, zorder=1, solid_capstyle="round")
            ax.scatter(*m, s=120, color=SURF, edgecolor=ec, linewidth=1.4, zorder=2); ax.text(*m, op, ha="center", va="center", fontsize=9.5, color=INK, zorder=3)
        for v in sorted(S | V):
            if v in S: fc, tc, ec = "#e6e5e0", INK, "#b9b8b2"
            elif v in mark: fc, tc, ec = col, "white", col
            elif v in T: fc, tc, ec = BLUE, "white", BLUE
            else: fc, tc, ec = "white", INK, "#b9b8b2"
            ax.text(*pos[v], f"{v:,}", ha="center", va="center", fontsize=9.5, color=tc, zorder=4, fontweight="bold" if v in mark else None,
                    bbox=dict(boxstyle="round,pad=0.32", fc=fc, ec=ec, lw=1.2))
        ax.set_title(f"{pid} · {title}   ({len(S | V)} values, {len(tr)} triples)"); ax.axis("off")
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2); ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    L = [mpatches.Patch(color="#e6e5e0", label="starting value"), mpatches.Patch(color=BLUE, label="target"), mpatches.Patch(fc="white", ec="#b9b8b2", label="helper value"),
         mpatches.Patch(color=RED, label="dropped"), mpatches.Patch(color=GREEN, label="added"),
         Line2D([], [], color=BLUE, marker="$+$", markersize=8, label="sum triple a + b = c"), Line2D([], [], color=ORANGE, marker="$×$", markersize=8, label="product triple a × b = c")]
    fig.legend(handles=L, loc="lower center", ncol=7, fontsize=10, bbox_to_anchor=(0.5, -0.01)); plt.tight_layout(rect=(0, 0.04, 1, 1))
    return fig

def demo_figure(runs, secs, machine):
    """runs: [dict(pid, start, public, trace=[(t, cost, best)], final)]"""
    style(); n = len(runs); rows = (n + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(17, 4.7 * rows), squeeze=False)
    for ax in axs.flat[n:]: ax.axis("off")
    for ax, r in zip(axs.flat, runs):
        t = [x[0] for x in r["trace"]] or [0]; cost = [x[1] for x in r["trace"]] or [r["start"]]; best = [x[2] for x in r["trace"]] or [r["start"]]
        t, cost, best = t + [secs], cost + [min(cost[-1], r["final"]) if r["final"] < best[-1] else cost[-1]], best + [min(best[-1], r["final"])]
        lo, hi = min(min(best), r["public"], r["final"]) - 2, max(max(cost), r["start"]) + 2
        ax.imshow(np.linspace(1, 0, 256)[None], extent=(0, secs, lo, hi), aspect="auto", cmap="Oranges", alpha=0.22, zorder=0, vmin=0, vmax=1.6)
        ax.plot(t, cost, color="#9ec5f4", lw=1.6, zorder=2, label="current state")
        ax.step(t, best, where="post", color=BLUE, lw=2.6, zorder=3, label="best valid program so far")
        ax.axhline(r["start"], color=MUTED, lw=1.3, ls=":", zorder=1); ax.axhline(r["public"], color=ORANGE, lw=1.4, ls="--", zorder=1)
        bb = dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85)
        ax.text(secs * 0.99, r["start"], f"22 Nov public: {r['start']}", ha="right", va="center", fontsize=9, color=INK2, bbox=bb, zorder=4)
        ax.text(secs * 0.99, r["public"], f"best public today: {r['public']}", ha="right", va="center", fontsize=9, color=INK2, bbox=bb, zorder=4)
        ax.set_xlim(0, secs); ax.set_ylim(lo, hi); ax.grid(False)
        d = r["start"] - r["final"]
        ax.set_title(f"{r['pid']}: {r['start']} → {r['final']} instructions (−{d})" if d else f"{r['pid']}: no shorter program this time"); ax.set_ylabel("instructions")
        ax.text(0.012, 0.03, "hot", transform=ax.transAxes, fontsize=9, color=INK2); ax.text(0.988, 0.03, "cold", transform=ax.transAxes, fontsize=9, color=INK2, ha="right")
    for ax in axs[-1]: ax.set_xlabel("seconds")
    axs.flat[0].legend(loc="upper center", ncol=2, fontsize=9)
    s0, s1 = sum(r["start"] for r in runs), sum(r["final"] for r in runs)
    fig.suptitle(f"Live on this machine ({machine}): {s0} → {s1} instructions in {secs} seconds per puzzle", x=0.01, ha="left", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
