# Local imports
from scripts.utils.plot_utils import mm_to_inches

# Others
import os
import glob
import logging
import argparse
import matplotlib
import rich.logging
import pandas as pd
import matplotlib.pyplot as plt


def generate_plot(df, out_name, out_folder, height, width):
    epochs = df["epoch"]

    # Plots for loss
    _, ax = plt.subplots(1, 2, figsize=(width, height))
    ax[0].plot(
        epochs, df["loss"], label="Training loss", color="blue"
    )  # linestyle='--', marker='o')
    ax[0].set_title("Training loss")
    ax[0].set_ylabel("total loss [au]")
    ax[0].set_xlabel("epoch")

    ax[1].plot(epochs, df["val_loss"], label="Validation loss", color="blue")
    ax[1].set_title("Validation loss")
    ax[1].set_ylabel("total loss [au]")
    ax[1].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{out_name}_loss.pdf"), format="pdf", dpi=300)
    plt.close()

    # Plots for dendrites' iou
    _, ax = plt.subplots(1, 2, figsize=(width, height))
    ax[0].plot(epochs, df["dendrites_iou_score"], label="Training", color="blue")
    ax[0].set_title("Dendrites - Training")
    ax[0].set_ylabel("IoU Score")
    ax[0].set_xlabel("epoch")

    ax[1].plot(epochs, df["val_dendrites_iou_score"], label="Validation", color="blue")
    ax[1].set_title("Dendrites - Validation")
    ax[1].set_ylabel("IoU Score")
    ax[1].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"{out_name}_dendrites_iou.pdf"), format="pdf", dpi=300
    )
    plt.close()

    # Plots for spines' iou
    _, ax = plt.subplots(1, 2, figsize=(width, height))
    ax[0].plot(epochs, df["spines_iou_score"], label="Training", color="blue")
    ax[0].set_title("Spines  Training")
    ax[0].set_ylabel("IoU Score")
    ax[0].set_xlabel("epoch")

    ax[1].plot(epochs, df["val_spines_iou_score"], label="Validation", color="blue")
    ax[1].set_title("Spines - Validation")
    ax[1].set_ylabel("IoU Score")
    ax[1].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"{out_name}_spines_iou.pdf"), format="pdf", dpi=300
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_trainng_loss_metrics",
        description="Generate plots from loss and metrics values from model training for BIMAP P4",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-fh",
        "--fig-height",
        type=float,
        default=60,
        help="Figure height for plots, in millimeters. Default to 120.",
    )
    parser.add_argument(
        "-fw",
        "--fig-width",
        type=float,
        default=180,
        help="Figure width for plots, in millimeters. Default to 180.",
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
    parser.add_argument(
        "-s",
        "--fontsize",
        type=int,
        default=8,
        help="Font size to be used in the plots. Default to 8.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set fontsize
    matplotlib.rcParams.update({"font.size": args.fontsize})

    # Add a logger and a log file.
    log_level = logging.INFO
    log_file = os.path.join(args.path, "plotting_metrics_log.txt")
    logging.basicConfig(
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            rich.logging.RichHandler(console=None, rich_tracebacks=True, markup=True),
        ],
        level=log_level,
    )
    log = logging.getLogger("rich")

    # Validate input folder(s) contents
    data_folder = args.path
    model_folder = f"{data_folder}/{args.model_folder}"

    # Validate existence of folder
    if not os.path.exists(data_folder):
        parser.error(f"Folder {data_folder} does not exist")
    if not os.path.exists(model_folder):
        parser.error(
            f"Folder {model_folder} does not exist. The model must be trained and placed into {model_folder}"
        )

    out_folder = f"{data_folder}/{args.out_plots_folder}"
    os.makedirs(out_folder, exist_ok=True)
    models_data = glob.glob(f"{model_folder}/*.csv")

    if args.verbose:
        log.info(
            f"Found {len(models_data)} trained models. Generating plots for them and saving in {out_folder}."
        )

    for model_data in models_data:
        df = pd.read_csv(model_data)
        folder_name = os.path.basename(os.path.dirname(model_data))
        file_name = os.path.splitext(os.path.basename(model_data))[0]
        out_name = f"{folder_name}_{file_name}"
        generate_plot(
            df=df,
            out_name=out_name,
            out_folder=out_folder,
            height=mm_to_inches(args.fig_height),
            width=mm_to_inches(args.fig_width),
        )

    if args.verbose:
        log.info(f"Finished generating plots.")
