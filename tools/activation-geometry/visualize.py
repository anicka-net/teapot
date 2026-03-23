#!/usr/bin/env python3
"""Visualize cross-model compassion and safety geometry results.

Generates 5 figures comparing 8 models across architecture families.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Model order: grouped by alignment architecture (integrated → bolted-on)
MODELS = [
    "apertus-8b",
    "yi-1.5-9b",
    "deepseek-r1-distill",
    "mistral-7b",
    "ke-v9",
    "llama-3.1-8b",
    "phi-4",
    "gemma-2-9b",
]

MODEL_DISPLAY = {
    "apertus-8b": "Apertus 8B",
    "deepseek-r1-distill": "DeepSeek R1-Distill",
    "gemma-2-9b": "Gemma 2 9B",
    "ke-v9": "Karma Electric v9",
    "llama-3.1-8b": "Llama 3.1 8B",
    "mistral-7b": "Mistral 7B",
    "phi-4": "Phi-4",
    "yi-1.5-9b": "Yi 1.5 9B",
}

# Framework display names (secular, publication-safe)
FW_DISPLAY = {
    "chenrezig": "Buddhist\n(Chenrezig)",
    "tara": "Buddhist\n(Tara)",
    "agape": "Christian\n(Agape)",
    "rahma": "Islamic\n(Rahma)",
    "secular": "Secular",
}

FW_SHORT = {
    "chenrezig": "Chenrezig",
    "tara": "Tara",
    "agape": "Agape",
    "rahma": "Rahma",
    "secular": "Secular",
}

FRAMEWORKS = ["chenrezig", "tara", "agape", "rahma", "secular"]

# Model colors — grouped by alignment architecture
MODEL_COLORS = {
    "apertus-8b": "#2E86C1",      # integrated: blue family
    "yi-1.5-9b": "#5DADE2",
    "deepseek-r1-distill": "#85C1E9",
    "mistral-7b": "#F5B041",      # mild constraint: amber
    "ke-v9": "#E67E22",           # intensified: saffron
    "llama-3.1-8b": "#E74C3C",    # bolted-on: red family
    "phi-4": "#CB4335",
    "gemma-2-9b": "#922B21",
}


def load_model_data(model):
    """Load all JSON results for a model."""
    path = os.path.join(RESULTS_DIR, model)
    data = {}
    for fname in os.listdir(path):
        if fname.endswith(".json"):
            key = fname.replace(".json", "")
            with open(os.path.join(path, fname)) as f:
                data[key] = json.load(f)
    return data


def get_identity_layer(data):
    """Get the identity (final analysis) layer index as string."""
    cfg = data["experiment_config"]
    layers = cfg.get("analysis_layers", list(range(cfg["n_layers"])))
    return str(max(layers))


def get_cosine_at_layer(data, fw1, fw2, layer_str):
    """Extract cosine similarity handling both data formats."""
    cos = data["cosine_similarity"]
    first_key = list(cos.keys())[0]

    if first_key.isdigit():
        # Apertus format: layer → fw1 → fw2
        return cos.get(layer_str, {}).get(fw1, {}).get(fw2, np.nan)
    else:
        # Pair format: "fw1_vs_fw2" → layer → value
        for pair_name in [f"{fw1}_vs_{fw2}", f"{fw2}_vs_{fw1}"]:
            if pair_name in cos:
                return cos[pair_name].get(layer_str, np.nan)
        return np.nan


def get_cosine_across_layers(data, fw1, fw2):
    """Get cosine values across all analysis layers, returns (layers, values)."""
    cfg = data["experiment_config"]
    layers = cfg.get("analysis_layers", list(range(cfg["n_layers"])))
    values = []
    for l in layers:
        values.append(get_cosine_at_layer(data, fw1, fw2, str(l)))
    return layers, values


def build_similarity_matrix(data, layer_str):
    """Build 5x5 similarity matrix at given layer."""
    matrix = np.zeros((len(FRAMEWORKS), len(FRAMEWORKS)))
    for i, fw1 in enumerate(FRAMEWORKS):
        for j, fw2 in enumerate(FRAMEWORKS):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = get_cosine_at_layer(data, fw1, fw2, layer_str)
    return matrix


def fig_heatmap_grid(all_data):
    """Per-model compassion heatmaps at identity layer (2x4 grid)."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    labels = [FW_SHORT[fw] for fw in FRAMEWORKS]

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 4, idx % 4]
        data = all_data[model]
        identity = get_identity_layer(data)
        matrix = build_similarity_matrix(data, identity)

        sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels if idx % 4 == 0 else False,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cmap="RdYlBu_r",
            vmin=0.0,
            vmax=1.0,
            center=0.5,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        n_layers = data["experiment_config"]["n_layers"]
        ax.set_title(
            f"{MODEL_DISPLAY[model]}\n(L{identity}/{n_layers})",
            fontsize=11,
            fontweight="bold",
        )
        ax.tick_params(labelsize=8)
        if idx % 4 == 0:
            ax.set_yticklabels(labels, rotation=0, fontsize=8)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle(
        "Contemplative Framework Similarity at Identity Layer",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(FIGURES_DIR, "1_heatmap_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_safety_comparison(all_data):
    """Safety-compassion alignment: the three architectures."""
    models_sorted = [
        "apertus-8b", "yi-1.5-9b", "deepseek-r1-distill",
        "mistral-7b", "ke-v9",
        "llama-3.1-8b", "phi-4", "gemma-2-9b",
    ]

    safety_chenrezig = []
    safety_empty = []
    for m in models_sorted:
        sg = all_data[m]["safety_geometry"]
        identity = get_identity_layer(all_data[m])
        safety_chenrezig.append(sg["safety_vs_chenrezig"][identity])
        safety_empty.append(sg["safety_vs_empty"][identity])

    labels = [MODEL_DISPLAY[m] for m in models_sorted]
    y_pos = np.arange(len(models_sorted))
    colors = [MODEL_COLORS[m] for m in models_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: safety ↔ chenrezig
    bars1 = ax1.barh(y_pos, safety_chenrezig, color=colors, edgecolor="white", height=0.7)
    ax1.set_xlabel("Cosine Similarity", fontsize=11)
    ax1.set_title("Safety ↔ Compassion", fontsize=13, fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlim(0, 0.6)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, safety_chenrezig):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

    # Right: safety ↔ empty (baseline)
    bars2 = ax2.barh(y_pos, safety_empty, color=colors, edgecolor="white", height=0.7)
    ax2.set_xlabel("Cosine Similarity", fontsize=11)
    ax2.set_title("Safety ↔ Baseline", fontsize=13, fontweight="bold")
    ax2.set_xlim(-0.55, 0.7)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars2, safety_empty):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax2.text(val + offset, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.2f}", va="center", ha=ha, fontsize=9)

    # Architecture annotations on right panel
    ax2.annotate("Integrated", xy=(0.55, 1.0), fontsize=9, color="#2E86C1",
                 fontweight="bold", ha="center")
    ax2.annotate("Bolted-on", xy=(-0.35, 7.0), fontsize=9, color="#922B21",
                 fontweight="bold", ha="center")

    fig.suptitle(
        "Safety–Compassion Alignment Across Models",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "2_safety_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_cluster_centers(all_data):
    """What sits at each model's geometric center — mean cosine per framework."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(MODELS))
    width = 0.15
    fw_colors = {
        "chenrezig": "#E67E22",
        "tara": "#F39C12",
        "agape": "#2E86C1",
        "rahma": "#27AE60",
        "secular": "#7F8C8D",
    }

    for i, fw in enumerate(FRAMEWORKS):
        means = []
        for model in MODELS:
            data = all_data[model]
            identity = get_identity_layer(data)
            # Mean cosine of this framework with all OTHER frameworks
            vals = []
            for other in FRAMEWORKS:
                if other != fw:
                    vals.append(get_cosine_at_layer(data, fw, other, identity))
            means.append(np.nanmean(vals))

        offset = (i - 2) * width
        bars = ax.bar(x + offset, means, width, label=FW_SHORT[fw],
                      color=fw_colors[fw], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Cosine with Other Frameworks", fontsize=12)
    ax.set_title(
        "Geometric Center: Which Tradition Is Most Central?",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], rotation=25, ha="right",
                        fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=10, ncol=5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "3_cluster_centers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_axis_norms(all_data):
    """Axis magnitude at identity layer — how much each model separates frameworks."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(FRAMEWORKS))
    width = 0.09
    n_models = len(MODELS)

    for i, model in enumerate(MODELS):
        data = all_data[model]
        identity = get_identity_layer(data)
        norms = []
        for fw in FRAMEWORKS:
            # Try axis_norms first, then safety_axis_norms
            val = data.get("axis_norms", {}).get(fw, {}).get(identity, np.nan)
            if np.isnan(val):
                val = data.get("safety_axis_norms", {}).get(fw, {}).get(identity, np.nan)
            norms.append(val)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, norms, width, label=MODEL_DISPLAY[model],
               color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Framework", fontsize=12)
    ax.set_ylabel("Axis Norm (log scale)", fontsize=12)
    ax.set_title(
        "Compassion Axis Magnitude at Identity Layer",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([FW_SHORT[fw] for fw in FRAMEWORKS], fontsize=11)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "4_axis_norms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_convergence(all_data):
    """Buddhist pair convergence across normalized layer depth."""
    pairs = {
        "Buddhist pair\n(Chenrezig ↔ Tara)": ("chenrezig", "tara"),
        "Buddhist–Christian\n(Chenrezig ↔ Agape)": ("chenrezig", "agape"),
        "Abrahamic pair\n(Agape ↔ Rahma)": ("agape", "rahma"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (pair_label, (fw1, fw2)) in zip(axes, pairs.items()):
        for model in MODELS:
            data = all_data[model]
            cfg = data["experiment_config"]
            n_layers = cfg["n_layers"]
            layers, values = get_cosine_across_layers(data, fw1, fw2)

            # Normalize to proportional depth (0.0 = first layer, 1.0 = last)
            depth = [l / (n_layers - 1) for l in layers]

            ax.plot(
                depth, values,
                color=MODEL_COLORS[model],
                linewidth=2,
                label=MODEL_DISPLAY[model],
                marker="o",
                markersize=3,
                alpha=0.85,
            )

        ax.set_xlabel("Proportional Depth", fontsize=11)
        ax.set_title(pair_label, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlim(0.45, 1.02)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    axes[0].set_ylabel("Cosine Similarity", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Cross-Tradition Convergence in Upper Layers (Normalized Depth)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(FIGURES_DIR, "5_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data for all models...")
    all_data = {}
    for model in MODELS:
        all_data[model] = load_model_data(model)
        cfg = all_data[model]["experiment_config"]
        print(f"  {MODEL_DISPLAY[model]}: {cfg['n_layers']} layers")

    print("\nGenerating figures:")
    fig_heatmap_grid(all_data)
    fig_safety_comparison(all_data)
    fig_cluster_centers(all_data)
    fig_axis_norms(all_data)
    fig_convergence(all_data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
