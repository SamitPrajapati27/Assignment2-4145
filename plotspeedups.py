#!/usr/bin/env python3

import csv
import os
import matplotlib.pyplot as plt

os.makedirs("data/plots", exist_ok=True)


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "part": r["part"],
                "pr": int(r["pr"]),
                "pc": int(r["pc"]),
                "p": int(r["p"]),
                "b": int(r["b"]),
                "M": int(r["M"]),
                "K": int(r["K"]),
                "N": int(r["N"]),
                "verified": r["verified"],
                "min_s": float(r["min_s"]),
                "median_s": float(r["median_s"]),
                "avg_s": float(r["avg_s"]),
                "max_s": float(r["max_s"]),
                "flops_per_s": float(r["flops_per_s"]),
            })
    return rows


def get_row(rows, pr, pc, b, M, K=None, N=None):
    K = M if K is None else K
    N = M if N is None else N
    for r in rows:
        if (r["pr"] == pr and r["pc"] == pc and r["b"] == b
                and r["M"] == M and r["K"] == K and r["N"] == N):
            return r
    raise KeyError(f"Missing row: pr={pr}, pc={pc}, b={b}, M={M}, K={K}, N={N}")


def plot_1d(rows, part_label, b_star, out_path):
    ps = [1, 2, 4, 8, 12, 18, 36]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    styles = [
        (1024, 0, "o--", "1024, b=0"),
        (1024, b_star, "o-", f"1024, b={b_star}"),
        (2048, 0, "s--", "2048, b=0"),
        (2048, b_star, "s-", f"2048, b={b_star}"),
    ]

    for sz, b, fmt, label in styles:
        t_seq = get_row(rows, 0, 0, 0, sz)["median_s"]
        speedups = [t_seq / get_row(rows, p, 1, b, sz)["median_s"] for p in ps]
        ax.plot(ps, speedups, fmt, label=label, linewidth=2, markersize=6)

    ax.plot(ps, ps, "k:", label="Ideal (S = p)", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks(ps)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Threads (p)")
    ax.set_ylabel("Speedup S(p) = T_seq / T(p)")
    ax.set_title(f"{part_label}: 1D Partition Speedup")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_2d(rows, part_label, b_star, out_path):
    ps_1d = [1, 2, 4, 8, 12, 18, 36]
    grids_2d = [(1, 2), (2, 2), (2, 4), (3, 4), (3, 6), (6, 6)]
    ps_2d = [pr * pc for pr, pc in grids_2d]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, sz in zip(axes, [1024, 2048]):
        t_seq = get_row(rows, 0, 0, 0, sz)["median_s"]
        s_1d = [t_seq / get_row(rows, p, 1, b_star, sz)["median_s"] for p in ps_1d]
        s_2d = [t_seq / get_row(rows, pr, pc, b_star, sz)["median_s"] for pr, pc in grids_2d]

        ax.plot(ps_1d, s_1d, "o-", label=f"1D (b={b_star})", linewidth=2, markersize=6)
        ax.plot(ps_2d, s_2d, "s-", label=f"2D Square Grid (b={b_star})", linewidth=2, markersize=6)
        ax.plot(ps_1d, ps_1d, "k:", label="Ideal (S = p)", linewidth=1.5)

        ax.set_xscale("log")
        ax.set_xticks(ps_1d)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Threads (p)")
        ax.set_ylabel("Speedup S(p) = T_seq / T(p)")
        ax.set_title(f"{part_label}: 1D vs 2D Speedup (N={sz})")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def format_table_rows(rows, t_seq_map):
    lines = []
    for r in rows:
        t_seq = t_seq_map.get(r["M"], None)
        sp = f"{t_seq / r['median_s']:.2f}x" if (t_seq and r["p"] > 0) else "—"
        grid = "seq" if r["p"] == 0 else f"{r['pr']}x{r['pc']}"
        gflops = r["flops_per_s"] / 1e9
        lines.append(
            f"| {r['M']}x{r['K']}x{r['N']} | {grid} | {r['p']} | {r['b']} | "
            f"{r['min_s']:.4f} | {r['median_s']:.4f} | {r['avg_s']:.4f} | {r['max_s']:.4f} | "
            f"{sp} | {r['flops_per_s']:.3e} ({gflops:.2f} GF/s) | {r['verified']} |"
        )
    return "\n".join(lines)


def main():
    sweep = load_csv("data/partA_sweep.csv")
    partA = load_csv("data/partA_results.csv")
    partB = load_csv("data/partB_results.csv")

    with open("data/b_star.txt") as f:
        b_star = int(f.read().strip().split("=")[-1].strip())

    # Generate all 4 required plots
    plot_1d(partA, "Part A (pthreads)", b_star, "data/plots/partA_1d_speedup.png")
    plot_2d(partA, "Part A (pthreads)", b_star, "data/plots/partA_2d_speedup.png")
    plot_1d(partB, "Part B (OpenMP)", b_star, "data/plots/partB_1d_speedup.png")
    plot_2d(partB, "Part B (OpenMP)", b_star, "data/plots/partB_2d_speedup.png")

    # Compute key metrics for written answers
    tseq_A_1024 = get_row(partA, 0, 0, 0, 1024)["median_s"]
    tseq_A_2048 = get_row(partA, 0, 0, 0, 2048)["median_s"]
    tseq_B_1024 = get_row(partB, 0, 0, 0, 1024)["median_s"]
    tseq_B_2048 = get_row(partB, 0, 0, 0, 2048)["median_s"]

    t1_A_2048_bstar = get_row(partA, 1, 1, b_star, 2048)["median_s"]
    t36_A_1d_b0 = get_row(partA, 36, 1, 0, 2048)["median_s"]
    t36_A_1d_bs = get_row(partA, 36, 1, b_star, 2048)["median_s"]
    t36_A_6x6 = get_row(partA, 6, 6, b_star, 2048)["median_s"]
    t36_A_1x36 = get_row(partA, 1, 36, b_star, 2048)["median_s"]

    t36_B_1d_bs = get_row(partB, 36, 1, b_star, 2048)["median_s"]
    t36_B_6x6 = get_row(partB, 6, 6, b_star, 2048)["median_s"]

    diff_1d_pct = 100.0 * (t36_B_1d_bs - t36_A_1d_bs) / t36_A_1d_bs
    diff_2d_pct = 100.0 * (t36_B_6x6 - t36_A_6x6) / t36_A_6x6

    tbl_hdr = (
        "| Size (MxKxN) | Grid | p | b | Min (s) | Median (s) | Avg (s) | Max (s) | Speedup | Rate (FLOP/s) | Verified |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )

    out = []
    out.append(f"# Auto-Generated Tables & Key Metrics (b* = {b_star})\n")
    out.append("## Key Numbers for Written Answers\n")
    out.append(f"- **Optimal tile size $b^*$**: `{b_star}`")
    out.append(f"- **Part A $T_{{seq}}$ (1024, b=0)**: `{tseq_A_1024:.6f} s`")
    out.append(f"- **Part A $T_{{seq}}$ (2048, b=0)**: `{tseq_A_2048:.6f} s`")
    out.append(f"- **Part A 5(b) $T(1)/T_{{seq}}$ (2048, b={b_star})**: `{t1_A_2048_bstar:.6f} / {tseq_A_2048:.6f} = {t1_A_2048_bstar / tseq_A_2048:.4f}`")
    out.append(f"- **Part A 5(c) $T(36)$ 1D (2048)**: `b=0: {t36_A_1d_b0:.6f} s` vs `b={b_star}: {t36_A_1d_bs:.6f} s`")
    out.append(f"- **Part A 5(d) $T(36)$ Grids (2048, b={b_star})**: `36x1: {t36_A_1d_bs:.6f} s`, `6x6: {t36_A_6x6:.6f} s`, `1x36: {t36_A_1x36:.6f} s`")
    out.append(f"- **Part B 5(c) OpenMP vs pthreads $T(36)$ (2048, b={b_star})**:")
    out.append(f"  - 1D (36x1): Part A = `{t36_A_1d_bs:.6f} s`, Part B = `{t36_B_1d_bs:.6f} s` (`{diff_1d_pct:+.2f}%`)")
    out.append(f"  - 2D (6x6): Part A = `{t36_A_6x6:.6f} s`, Part B = `{t36_B_6x6:.6f} s` (`{diff_2d_pct:+.2f}%`)\n")

    out.append("## Part A: Sequential Tile Sweep\n")
    out.append(tbl_hdr)
    out.append(format_table_rows(sweep, {1024: tseq_A_1024, 2048: tseq_A_2048}))
    out.append("\n## Part A: All Results (Baselines, 1D, 2D, Non-Divisible)\n")
    out.append(tbl_hdr)
    out.append(format_table_rows(partA, {1024: tseq_A_1024, 2048: tseq_A_2048}))
    out.append("\n## Part B: All Results (Baselines, 1D, 2D, Non-Divisible)\n")
    out.append(tbl_hdr)
    out.append(format_table_rows(partB, {1024: tseq_B_1024, 2048: tseq_B_2048}))

    with open("data/tables_and_summary.md", "w") as f:
        f.write("\n".join(out) + "\n")

    print("Generated plots in data/plots/ and tables in data/tables_and_summary.md")


if __name__ == "__main__":
    main()