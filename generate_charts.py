"""
Generate publication-quality charts for TFM experiment results.
Outputs PNGs for LaTeX inclusion.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS_DIR = Path("data/results")
OUTPUT_DIR = Path("latex/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

UNIR_BLUE = "#0098CD"
COLORS = ["#0098CD", "#FF6B35", "#2ECC71", "#9B59B6"]


def load_summary(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def fig1_phase1_metrics_bar():
    """Grouped bar chart: Phase 1 metrics by chunking strategy."""
    data = load_summary("phase1_summary.json")
    configs = data["configs"]
    labels = ["C1\nTamaño fijo", "C2\nRecursivo", "C3\nSemántico"]
    metrics = ["rouge1_f1", "rougeL_f1", "bertscore_f1"]
    metric_labels = ["ROUGE-1 F1", "ROUGE-L F1", "BERTScore F1"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        values = [c["metrics"][metric] for c in configs]
        bars = ax.bar(labels, values, color=COLORS[:3], edgecolor="white", width=0.6)
        ax.set_title(mlabel, fontweight="bold")
        if metric == "bertscore_f1":
            ax.set_ylim(0.822, 0.835)
        else:
            ax.set_ylim(0, max(values) * 1.25)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("#333333")
        bars[best_idx].set_linewidth(2)

    fig.suptitle("Fase 1: Métricas automáticas por estrategia de segmentación", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_fase1_metricas.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_fase1_metricas.png'}")


def fig2_phase2_metrics_bar():
    """Grouped bar chart: Phase 2 metrics by embedding model."""
    data = load_summary("phase2_summary.json")
    configs = data["configs"]
    labels = ["BGE\n(Gen.)", "MiniLM\n(Gen.)", "BioLORD\n(Bio.)", "PubMedBERT\n(Bio.)"]
    metrics = ["rouge1_f1", "rougeL_f1", "bertscore_f1"]
    metric_labels = ["ROUGE-1 F1", "ROUGE-L F1", "BERTScore F1"]

    # Reorder: BGE, MiniLM, BioLORD, PubMedBERT
    order = [1, 0, 3, 2]  # E2_BGE, E1_MiniLM, E4_BioLORD, E3_PubMedBERT
    configs_ordered = [configs[i] for i in order]
    colors_bar = [COLORS[0], COLORS[0], COLORS[1], COLORS[1]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        values = [c["metrics"][metric] for c in configs_ordered]
        bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", width=0.6)
        ax.set_title(mlabel, fontweight="bold")

        if metric == "bertscore_f1":
            ax.set_ylim(0.80, 0.84)
        else:
            ax.set_ylim(0, max(values) * 1.25)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("#333333")
        bars[best_idx].set_linewidth(2)

    fig.suptitle("Fase 2: Métricas automáticas por modelo de embedding", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_fase2_metricas.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_fase2_metricas.png'}")


def fig3_heatmap_phase1():
    """Heatmap: ROUGE-1 by question type x chunking strategy."""
    data = load_summary("phase1_summary.json")
    configs = data["configs"]

    matrix = np.array([
        [c["metrics_factual"]["rouge1_f1"] for c in configs],
        [c["metrics_explanatory"]["rouge1_f1"] for c in configs],
        [c["metrics_comparative"]["rouge1_f1"] for c in configs],
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.13, vmax=0.25)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["C1 Fijo", "C2 Recursivo", "C3 Semántico"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Factual", "Explicativa", "Comparativa"])

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > 0.21 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title("Fase 1: ROUGE-1 F1 por tipo de pregunta y estrategia", fontweight="bold")
    fig.colorbar(im, ax=ax, label="ROUGE-1 F1", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_fase1_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_fase1_heatmap.png'}")


def fig4_heatmap_phase2():
    """Heatmap: ROUGE-1 by question type x embedding model."""
    data = load_summary("phase2_summary.json")
    configs = data["configs"]
    # Order: BGE, MiniLM, BioLORD, PubMedBERT
    order = [1, 0, 3, 2]
    configs_ordered = [configs[i] for i in order]

    matrix = np.array([
        [c["metrics_factual"]["rouge1_f1"] for c in configs_ordered],
        [c["metrics_explanatory"]["rouge1_f1"] for c in configs_ordered],
        [c["metrics_comparative"]["rouge1_f1"] for c in configs_ordered],
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.10, vmax=0.27)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["BGE\n(Gen.)", "MiniLM\n(Gen.)", "BioLORD\n(Bio.)", "PubMedBERT\n(Bio.)"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Factual", "Explicativa", "Comparativa"])

    for i in range(3):
        for j in range(4):
            val = matrix[i, j]
            color = "white" if val > 0.20 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title("Fase 2: ROUGE-1 F1 por tipo de pregunta y modelo de embedding", fontweight="bold")
    fig.colorbar(im, ax=ax, label="ROUGE-1 F1", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_fase2_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_fase2_heatmap.png'}")


def fig5_gen_vs_bio():
    """Grouped bar chart: generalista vs biomédico comparison."""
    gen_metrics = {"rouge1_f1": 0.2213, "rougeL_f1": 0.1539, "bertscore_f1": 0.8331}
    bio_metrics = {"rouge1_f1": 0.1600, "rougeL_f1": 0.1130, "bertscore_f1": 0.8184}

    metrics = ["ROUGE-1 F1", "ROUGE-L F1", "BERTScore F1"]
    gen_vals = list(gen_metrics.values())
    bio_vals = list(bio_metrics.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, gen_vals, width, label="Generalistas", color=COLORS[0], edgecolor="white")
    bars2 = ax.bar(x + width / 2, bio_vals, width, label="Biomédicos", color=COLORS[1], edgecolor="white")

    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparación agregada: Generalistas vs. Biomédicos", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add significance markers
    for i in range(3):
        y_max = max(gen_vals[i], bio_vals[i])
        ax.annotate("***", xy=(i, y_max + 0.008), ha="center", fontsize=14, fontweight="bold", color="#333333")

    for bars, vals in [(bars1, gen_vals), (bars2, bio_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.015,
                    f"{val:.4f}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

    ax.set_ylim(0, 0.92)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_gen_vs_bio.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_gen_vs_bio.png'}")


def fig6_radar_optimal():
    """Radar chart comparing all 7 configs across 3 metrics."""
    p1 = load_summary("phase1_summary.json")
    p2 = load_summary("phase2_summary.json")

    labels_metrics = ["ROUGE-1", "ROUGE-L", "BERTScore"]
    configs_all = [
        ("C1 Fijo", p1["configs"][0]["metrics"]),
        ("C2 Recursivo", p1["configs"][1]["metrics"]),
        ("C3 Semántico", p1["configs"][2]["metrics"]),
        ("BGE", p2["configs"][1]["metrics"]),
        ("MiniLM", p2["configs"][0]["metrics"]),
        ("BioLORD", p2["configs"][3]["metrics"]),
        ("PubMedBERT", p2["configs"][2]["metrics"]),
    ]

    # Normalize to 0-1 range for radar
    all_r1 = [c[1]["rouge1_f1"] for c in configs_all]
    all_rl = [c[1]["rougeL_f1"] for c in configs_all]
    all_bs = [c[1]["bertscore_f1"] for c in configs_all]

    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

    nr1, nrl, nbs = norm(all_r1), norm(all_rl), norm(all_bs)

    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    phase1_colors = ["#AAAAAA", UNIR_BLUE, "#CCCCCC"]
    phase2_colors = [COLORS[0], "#88CCEE", COLORS[1], "#CC6666"]
    all_colors = phase1_colors + phase2_colors

    for i, (name, _) in enumerate(configs_all):
        vals = [nr1[i], nrl[i], nbs[i]]
        vals += vals[:1]
        lw = 3 if name in ("C2 Recursivo", "BGE") else 1.5
        ls = "-" if name in ("C2 Recursivo", "BGE") else "--"
        ax.plot(angles, vals, linewidth=lw, linestyle=ls, label=name, color=all_colors[i])
        if name in ("C2 Recursivo", "BGE"):
            ax.fill(angles, vals, alpha=0.1, color=all_colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_metrics, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Comparación normalizada de todas las configuraciones", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_radar_configs.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_radar_configs.png'}")


def fig7_boxplot_phase2():
    """Box plots of per-question ROUGE-1 distributions for Phase 2."""
    files = {
        "BGE\n(Gen.)": RESULTS_DIR / "phase2_E2_BGE.json",
        "MiniLM\n(Gen.)": RESULTS_DIR / "phase2_E1_MiniLM.json",
        "BioLORD\n(Bio.)": RESULTS_DIR / "phase2_E4_BioLORD.json",
        "PubMedBERT\n(Bio.)": RESULTS_DIR / "phase2_E3_PubMedBERT.json",
    }

    data_box = []
    labels_box = []
    for label, fpath in files.items():
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        scores = [r["rouge1_f1"] for r in d["results"]]
        data_box.append(scores)
        labels_box.append(label)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))

    colors_box = [COLORS[0], COLORS[0], COLORS[1], COLORS[1]]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("ROUGE-1 F1")
    ax.set_title("Distribución de ROUGE-1 F1 por modelo de embedding (Fase 2)", fontweight="bold")

    # Add mean markers
    means = [np.mean(d) for d in data_box]
    ax.scatter(range(1, 5), means, color="red", marker="D", s=50, zorder=5, label="Media")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_fase2_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_fase2_boxplot.png'}")


if __name__ == "__main__":
    fig1_phase1_metrics_bar()
    fig2_phase2_metrics_bar()
    fig3_heatmap_phase1()
    fig4_heatmap_phase2()
    fig5_gen_vs_bio()
    fig6_radar_optimal()
    fig7_boxplot_phase2()
    print("\nAll charts generated successfully!")
