# Local imports
from scripts.utils.generic_utils import mm_to_inches, m_to_um

# Others
import os
import glob
import json
import logging
import pathlib
import argparse
import rich.logging
from itertools import groupby
import pandas as pd
import matplotlib.pyplot as plt


def plot_res_distribution(in_folder: str, out_folder: str, animal: str):
    # Get all metadata files.
    meta_files = glob.glob(f"{in_folder}/*_meta.json")

    # Get resolutions. Group them by Z resolution.
    res_data = []
    for meta_file in meta_files:
        with open(meta_file) as jf:
            meta = json.load(jf)
            res_z, res_x, res_y = meta["pixel_sizes"]["sizes"]
            res_z, res_x, res_y = m_to_um(res_z), m_to_um(res_x), m_to_um(res_y)
            res_xy = max(res_x, res_y)
            res_data.append(
                {"res_z": res_z, "res_xy": res_xy, "file": os.path.basename(meta_file)}
            )
            jf.close()
            del jf

    if len(res_data) == 0:
        log.error("No metadata, stopping execution")
        exit(0)
    elif args.verbose:
        log.info(f"Using {len(res_data)} resolutions for generating plots")

    # Group by Z resolution and aggregate XY resolutions.
    df = pd.DataFrame(res_data)
    grouped_res_data = (
        df.groupby("res_z")
        .agg(res_xy=pd.NamedAgg(column="res_xy", aggfunc=lambda x: sorted(set(x))))
        .reset_index()
    )

    # Generate latex table with data.
    tex_file = os.path.join(out_folder, f"{animal}_res_table.tex")
    with open(tex_file, "w") as tf:
        tf.write("\\begin{tabular}{l l}\n")
        tf.write("\\toprule\n")
        tf.write("Z res ($\\mu m$) & XY res ($\\mu m$) \\\\\n")
        tf.write("\\midrule\n")
        for _, row in grouped_res_data.iterrows():
            z = row["res_z"]
            xy = ", ".join(f"{xy:.4f}" for xy in row["res_xy"])
            tf.write(f"{z:.4f} & {xy} \\\\\n")
        tf.write("\\bottomrule\n")
        tf.write("\\end{tabular}\n")

    # Generate vector graphics plot.
    plt.figure(figsize=(mm_to_inches(args.fig_height), mm_to_inches(args.fig_width)))
    for z, group_df in df.groupby("res_z"):
        xy_res = group_df[["res_xy"]].values
        plt.scatter([z] * len(xy_res), xy_res[:, 0], label=f"{z:.4f} um", alpha=0.7)

    plt.xlabel("Z res")
    plt.ylabel("XY res")
    plt.title(f"Resolution distribution for {animal}")
    plt.grid(True)
    plt.tight_layout()

    # Save as vector PDF
    png_path = os.path.join(out_folder, f"{animal}_res_plot.png")
    plt.savefig(png_path, format="png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_res_distribution",
        description="Get resolutions distribution for BIMAP P4 turtle and mice brain images for training/validation",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-fh",
        "--fig-height",
        type=float,
        default=120,
        help="Figure height for resolution distribution plot, in millimeters. Default to 120.",
    )
    parser.add_argument(
        "-m",
        "--out-plots-folder",
        default="plots",
        help="Path to the folder to contain the generated plots and distribution data, relative to args.path. Default to 'plots' (that is, args.path/plots).",
    )
    parser.add_argument(
        "-o",
        "--preproc-out-folder",
        default="processed",
        help="Path to the folder containing the preprocessed data, relative to args.path. Default to 'processed' (that is, args.path/processed).",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=f"{current_folder}/images",
        help="Path to the folder containing the raw data. Default to 'images' on the current folder.",
    )
    parser.add_argument(
        "-fw",
        "--fig-width",
        type=float,
        default=180,
        help="Figure width for resolution distribution plot, in millimeters. Default to 180.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Add a logger and a log file.
    log_level = logging.INFO
    log_file = os.path.join(args.path, "res_distribution_log.txt")
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
    preproc_folder = f"{data_folder}/{args.preproc_out_folder}"

    # Validate existence of folder
    if not os.path.exists(data_folder):
        parser.error(f"Folder {data_folder} does not exist")
    if not os.path.exists(preproc_folder):
        parser.error(
            f"Folder {preproc_folder} does not exist. The raw data must be already preprocessed and placed into {preproc_folder}"
        )

    # Validate existence of subfolders with preprocessed data for each animal
    subfolders = [f.path for f in os.scandir(preproc_folder) if f.is_dir()]
    subfolders.sort()

    def folder_key(f_name):
        return pathlib.PurePath(f_name).name

    animals_data = {
        gr: list(items) for gr, items in groupby(subfolders, key=folder_key)
    }

    animals_data.pop(args.out_plots_folder, None)

    for animal, data_subfolders in animals_data.items():
        if args.verbose:
            log.info(f"Getting resolutions distribution for {animal}")
        out_folder = f"{data_folder}/{args.out_plots_folder}/{animal}"
        os.makedirs(out_folder, exist_ok=True)
        for subfolder in data_subfolders:
            if args.verbose:
                log.info(f"    Processing metadata from {subfolder}")
            plot_res_distribution(
                in_folder=subfolder, out_folder=out_folder, animal=animal
            )
            if args.verbose:
                log.info(f"    Finished processing metadata from {subfolder}")
        if args.verbose:
            log.info(
                f"Finished processing metadata for {animal}. Plots written to {out_folder}"
            )

    if args.verbose:
        log.info(f"Finished processing all data")
