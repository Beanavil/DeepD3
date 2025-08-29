import re
import os
import numpy as np
import surface_distance
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, jaccard_score

# Science plots setup
import scienceplots

plt.style.use(["science", "ieee", "muted"])  # high-contrast, high-vis

# Accommodate plot style to be Nature's compliant
plt.rcParams.update(
    {"font.size": 6, "font.family": "sans-serif", "font.sans-serif": "Helvetica"}
)

# Style settings, assuming 4 model scalings (8, 16, 32, 64) and one reference (vanilla u-net) model
key_idx = {
    # DeepD3 variants
    "64f": 0,
    "32f": 1,
    "16f": 2,
    "8f": 3,
    # Vanilla U-Net
    "vanilla": 4,
}

densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
plot_linestyles = ["solid", "dashed", "dashdot", "dotted", densely_dashdotdotted]
plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plot_markerstyles = ["X", "s", "^", "o", "+"]


# General plots utilities
def truncate(f, n):
    factor = 10.0**n
    return int(f * factor) / factor


def mm_to_inches(mm):
    return mm / 25.4


def m_to_um(res_in_m):
    return truncate(res_in_m * 1e6, 4)


def extract_filters(model_name):
    match = re.search(r"f(\d+)", str(model_name))
    if match:
        return f"{int(match.group(1))}f"
    return None


def extract_model_and_filters(model_name):
    match = re.match(r"([a-zA-Z0-9]*)_f([\d]+)", str(model_name))
    if match:
        name = match.group(1)
        variant = f" {int(match.group(2))}f" if match.group(1) == "DeepD3" else ""
        return name + variant
    return None


# Training plots utilities
def common_train_plot_setup(y_label):
    """Common setup for the training and validation plots, which are side-by-side plots."""
    fig, ax = plt.subplots(1, 2)
    titles = ["Training", "Validation"]
    for i in range(len(ax)):
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(y_label)
        ax[i].set_xlabel("epoch")
    return fig, ax


def common_train_plot_aftersetup(fig, ax, y_min, y_max):
    """Common after (plotting) setup for the training and validation plots."""
    # x and y ticks and labels, for consistency
    x = range(1, 31)
    y_range_size = float(y_max) - float(y_min)
    y = [
        round(y_min + i * 0.05, 2) for i in range(int(np.ceil(y_range_size / 0.05)) + 1)
    ]
    x_labels = [str(x_val) if x_val == 0 or x_val == len(x) else "" for x_val in x]
    y_labels = [str(0) if y_val == 0.0 else str(y_val) for y_val in y]
    for i in range(len(ax)):
        ax[i].set_xticks(ticks=x, labels=[label for label in x_labels])
        ax[i].set_yticks(ticks=y, labels=[label for label in y_labels])
    # Position of the legend box fixed outside
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.16, 0.90), frameon=True)


def plot_train_loss(models_dfs, y_min, y_max, out_folder):
    """Plots training and validation loss for DeepD3 versus vanilla U-Net."""
    y_label = "total loss [au]"
    fig, ax = common_train_plot_setup(y_label)
    for model_name in models_dfs.keys():
        label = f"{model_name}"
        for filters in models_dfs[model_name].keys():
            df = models_dfs[model_name][filters]
            epochs = df["epoch"]
            if model_name == "DeepD3":
                label = f"{model_name} {filters}"
            idx = key_idx[filters]
            color = plot_colors[idx]
            linestyle = plot_linestyles[idx]
            ax[0].plot(
                epochs,
                df[f"loss"],
                label=label,
                color=color,
                linestyle=linestyle,
            )
            ax[1].plot(
                epochs,
                df[f"val_loss"],
                label=label,
                color=color,
                linestyle=linestyle,
            )
    common_train_plot_aftersetup(fig, ax, y_min, y_max)
    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"train_loss_deepd3_vanilla.png"), dpi=300)
    plt.savefig(
        os.path.join(out_folder, f"train_loss_deepd3_vanilla.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close()


def plot_train_iou(models_dfs, y_min, y_max, out_folder):
    """Plots training and validation loss and IoU for DeepD3 versus vanilla U-Net."""
    y_label = "IoU score"
    fig, ax = common_train_plot_setup(y_label)
    for element_str in ["dendrites", "spines"]:
        for model_name in models_dfs.keys():
            label = f"{model_name}"
            for filters in models_dfs[model_name].keys():
                df = models_dfs[model_name][filters]
                epochs = df["epoch"]
                if element_str == "spines":
                    label = "__nolabel__"
                elif model_name == "DeepD3":
                    label = f"{model_name} {filters}"
                idx = key_idx[filters]
                color = plot_colors[idx]
                linestyle = plot_linestyles[idx]
                ax[0].plot(
                    epochs,
                    df[f"{element_str}_iou_score"],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                )
                ax[1].plot(
                    epochs,
                    df[f"val_{element_str}_iou_score"],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                )
    # Plot divinding line
    ax[0].hlines(y=0.25, xmin=13, xmax=17, linewidth=0.5, color="black")
    ax[1].hlines(y=0.25, xmin=13, xmax=17, linewidth=0.5, color="black")
    ax[0].text(15, 0.255, "dendrite", fontsize=5, ha="center", va="bottom")
    ax[0].text(15, 0.245, "spines", fontsize=5, ha="center", va="top")
    ax[1].text(15, 0.255, "dendrite", fontsize=5, ha="center", va="bottom")
    ax[1].text(15, 0.245, "spines", fontsize=5, ha="center", va="top")
    common_train_plot_aftersetup(fig, ax, y_min, y_max)
    # Save
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"train_iou_score_deepd3_vanilla.png"), dpi=300
    )
    plt.savefig(
        os.path.join(out_folder, f"train_iou_score_deepd3_vanilla.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close()


# Inference plots utilities
def compute_infer_metrics(ground_truth, prediction, metrics):
    """Computes the metrics for the inference benchmarking data.

    Those are recall for segmentation quality and IoU and NSD for segmentation agreement.
    """
    results = {}
    if "recall" in metrics:
        results["recall"] = recall_score(
            ground_truth.flatten(), prediction.flatten(), zero_division=0
        )
    if "IoU" in metrics:
        results["IoU"] = jaccard_score(
            ground_truth.flatten(), prediction.flatten(), zero_division=0
        )
    if "NSD" in metrics:
        z_res_mm = 0.3e-3
        xy_res_mm = 0.2e-4
        tolerance = z_res_mm
        surface_distances = surface_distance.compute_surface_distances(
            ground_truth, prediction, spacing_mm=(z_res_mm, xy_res_mm, xy_res_mm)
        )
        results["NSD"] = surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, tolerance
        )
    return results


def plot_infer_metrics(metrics_df, x_max, out_plots_folder, suffix="deepd3"):
    """Create plots for metrics values across benchmarks"""
    x = range(x_max)
    x_labels = [str(x_val) if x_val == 0 or x_val == len(x) - 1 else "" for x_val in x]
    metrics = metrics_df.columns.values.tolist()[1:]
    for metric in metrics:
        y_max = 5 if metric == "IoU" else 10
        y = [y / 10.0 for y in range(0, y_max + 1, 1)]
        y_labels = [
            str(1) if y_val == 1.0 else str(0) if y_val == 0.0 else str(y_val)
            for y_val in y
        ]
        _, ax = plt.subplots()
        for variant in metrics_df["DeepD3 scaling variant"].unique():
            variant_rows = metrics_df[metrics_df["DeepD3 scaling variant"] == variant]
            idx = key_idx[variant]
            linestyle = plot_linestyles[idx]
            marker = plot_markerstyles[idx]
            plt.plot(
                x,
                variant_rows[metric],
                label=variant,
                linestyle=linestyle,
                marker=marker,
                markersize=1,
            )
        ax.set_title(f"{metric} across benchmarks")
        ax.set_ylabel(metric)
        ax.set_xticks(ticks=x, labels=[label for label in x_labels])
        ax.set_yticks(ticks=y, labels=[label for label in y_labels])
        # ax.legend(bbox_to_anchor=(1.17, 1), loc=1, frameon=True)
        ax.legend(loc=0, frameon=True)
        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(out_plots_folder, f"{metric}_{suffix}.png"))
        plt.savefig(
            os.path.join(out_plots_folder, f"{metric}_{suffix}.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.close()


def create_mean_std_table(
    metrics_df,
    out_tables_folder,
    caption,
    label,
    suffix="deepd3",
    column_format="cc",
    index=False,
):
    """Plots a summary of the metrics values (mean +- stddev) for the inference benchmarking data."""
    metrics = metrics_df.columns.values.tolist()[1:]
    summary = metrics_df.groupby("DeepD3 scaling variant").agg(["mean", "std"])
    summary.columns = ["_".join(x) for x in summary.columns.ravel()]
    summary = summary.reset_index()
    for metric in metrics:
        summary[f"{metric}"] = (
            "$"
            + summary[f"{metric}_mean"].round(3).astype(str)
            + " \\pm "
            + summary[f"{metric}_std"].round(3).astype(str)
            + "$"
        )
    summary = summary[["DeepD3 scaling variant"] + metrics].sort_values(
        by="DeepD3 scaling variant",
        key=lambda col: col.map(key_idx),
        ascending=False,
    )
    if not index:
        summary.drop("DeepD3 scaling variant", axis=1, inplace=True)
    summary_table = summary.to_latex(
        index=False,
        position="h",
        column_format=column_format,
        escape=False,
        float_format="{:0.3f}".format,
        caption=caption,
        label=label,
    )
    metrics_str = "_".join(metrics_df.columns.values.tolist()[1:])
    # Save
    with open(
        os.path.join(out_tables_folder, f"summary_{metrics_str}_{suffix}.tex"), "w"
    ) as f:
        f.write(summary_table)
