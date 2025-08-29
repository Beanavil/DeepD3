# Local imports
import scripts.utils.plot_utils as pu
from scripts.utils.generic_utils import variant_ordering_key

# Others
import os
import glob
import logging
import argparse
import matplotlib
import numpy as np
import pandas as pd
import rich.logging


def generate_plots(models_dfs, loss_minmax, iou_minmax, out_folder):
    pu.plot_train_loss(models_dfs, loss_minmax[0], loss_minmax[1], out_folder)
    pu.plot_train_iou(models_dfs, iou_minmax[0], iou_minmax[1], out_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_training_analysis",
        description="Generate plots from loss and metrics values from model training for BIMAP P4",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-fh",
        "--fig-height",
        type=float,
        default=50,
        help="Figure height for plots, in millimeters. Default to 50.",
    )
    parser.add_argument(
        "-fw",
        "--fig-width",
        type=float,
        default=110,
        help="Figure width for plots, in millimeters. Default to 110.",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        default="models",
        help="Path to the folder containing the model and its loss/metrics data, relative to args.path. Default to 'models' (that is, args.path/models).",
    )
    parser.add_argument(
        "-o",
        "--out-plots-folder",
        default="plots",
        help="Path to the folder to contain the generated plots, relative to args.path. Default to 'plots' (that is, args.path/plots).",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=f"{current_folder}/images",
        help="Path to the folder containing the raw data. Default to 'images' on the current folder.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set plot style to be Nature's compliant
    matplotlib.rcParams["figure.figsize"] = pu.mm_to_inches(
        args.fig_width
    ), pu.mm_to_inches(args.fig_height)

    # Add a logger
    log_level = logging.INFO
    logging.basicConfig(
        format="%(message)s",
        handlers=[
            rich.logging.RichHandler(console=None, rich_tracebacks=True, markup=True),
        ],
        level=log_level,
    )
    log = logging.getLogger("rich")

    # Validate input folder(s) contents
    data_folder = args.path
    model_folder = f"{data_folder}/{args.model_folder}"

    if not os.path.exists(data_folder):
        parser.error(f"Folder {data_folder} does not exist")
        exit()
    if not os.path.exists(model_folder):
        parser.error(
            f"Folder {model_folder} does not exist. The model must be trained and placed into {model_folder}"
        )
        exit()

    out_folder = f"{data_folder}/{args.out_plots_folder}"
    os.makedirs(out_folder, exist_ok=True)
    models_data = sorted(
        glob.glob(f"{model_folder}/*metrics*.csv"), key=variant_ordering_key
    )

    if args.verbose:
        log.info(
            f"Found {len(models_data)} trained models. Generating plots for them and saving in {out_folder}."
        )

    models_dfs = {}
    all_losses = []
    all_ious = []
    for model_data in models_data:
        model_df = pd.read_csv(model_data)
        basename = os.path.splitext(os.path.basename(model_data))[0]
        model_name = basename.split("_")[0]
        filters = pu.extract_filters(basename)
        if not model_name in models_dfs:
            models_dfs[model_name] = {}
        if model_name == "DeepD3":
            models_dfs[model_name][filters] = model_df
        else:
            # For the vanilla U-Net we only have one variant
            models_dfs[model_name]["vanilla"] = model_df
        all_losses.extend(model_df["loss"].tolist())
        all_losses.extend(model_df["val_loss"].tolist())
        for element_str in ["dendrites", "spines"]:
            all_ious.extend(model_df[f"{element_str}_iou_score"].tolist())
            all_ious.extend(model_df[f"val_{element_str}_iou_score"].tolist())

    # Compute minimum and maximum values of the metrics rounded down and up, respectively, by 0.05.
    loss_min, iou_min = (
        np.floor(min(all_losses) * 20) / 20,
        np.floor(min(all_ious) * 20) / 20,
    )
    loss_max, iou_max = (
        np.ceil(max(all_losses) * 20) / 20,
        np.ceil(max(all_ious) * 20) / 20,
    )

    generate_plots(
        models_dfs=models_dfs,
        loss_minmax=[loss_min, loss_max],
        iou_minmax=[iou_min, iou_max],
        out_folder=out_folder,
    )

    if args.verbose:
        log.info(f"Finished generating plots.")
