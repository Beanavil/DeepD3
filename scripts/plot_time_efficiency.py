# Local imports
from scripts.utils.generic_utils import variant_ordering_series_key
from scripts.utils.plot_utils import extract_model_and_filters

# Others
import os
import glob
import logging
import argparse
import rich.logging
import pandas as pd


def generate_table(timings_dfs, out_folder):
    df = timings_dfs["train"]
    cpu_df = timings_dfs["inf_cpu"]
    gpu_df = timings_dfs["inf_gpu"]

    # Merge inference dfs and create multicols
    merged_df = pd.merge(df, cpu_df, on="model")
    merged_df = pd.merge(merged_df, gpu_df, on="model", suffixes=("", "_gpu"))
    merged_df = merged_df.rename(
        columns={
            "model": "Model",
            "avg_batch_time_ms": "Training time",
            "avg_inference_time_ms": ("CPU", "Inference time"),
            "avg_inference_time_ms_gpu": ("GPU", "Inference time"),
        }
    )
    merged_df["Model"] = merged_df["Model"].apply(extract_model_and_filters)
    merged_df.columns = pd.MultiIndex.from_tuples(
        [c if isinstance(c, tuple) else ("", c) for c in merged_df.columns]
    )

    # Compute efficiencies
    merged_df.insert(
        3,
        ("CPU", "Efficiency"),
        merged_df[("CPU", "Inference time")] / merged_df[("", "Training time")],
    )
    merged_df.insert(
        5,
        ("GPU", "Efficiency"),
        merged_df[("GPU", "Inference time")] / merged_df[("", "Training time")],
    )

    # Create table
    timings_table = merged_df.to_latex(
        index=False,
        caption=(
            "Training and inference times and time efficiencies for different DeepD3 "
            "scaling variants (8, 16, 32 and 64 filters). Training time is measured "
            "as average milliseconds (ms) for training a batch, and inference time "
            "is measured as average milliseconds (ms) of inference across the benchmarking dataset."
            "dataset. Training was run on an NVIDIA A5000 GPU, as well as the GPU inference, while "
            "the CPU inference was run on an AMD Ryzen 9 6900HS."
        ),
        label="tab:timings",
        position="h",
        column_format="ccccc",
        escape=False,
        float_format="{:0.3f}".format,
    )

    # Save
    with open(os.path.join(out_folder, "timings_table.tex"), "w") as f:
        f.write(timings_table)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_time_efficiency",
        description="Generate tables for timings and time efficiencies from training and inference timings for BIMAP P4",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-i",
        "--inference-folder",
        default="inference",
        help="Path to the folder containing the predicted stacks and their timing files, relative to args.path. Default to 'models' (that is, args.path/models).",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        default="models",
        help="Path to the folder containing the models and the timing file, relative to args.path. Default to 'models' (that is, args.path/models).",
    )
    parser.add_argument(
        "-o",
        "--out-tables-folder",
        default="tables",
        help="Path to the folder to contain the generated tables, relative to args.path. Default to 'tables' (that is, args.path/tables).",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=f"{current_folder}/images",
        help="Path to the folder containing the raw data. Default to 'images' on the current folder.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

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

    # Validate input folder(s) existence
    data_folder = args.path
    inference_cpu_folder = os.path.join(data_folder, f"{args.inference_folder}_cpu")
    inference_gpu_folder = os.path.join(data_folder, f"{args.inference_folder}_gpu")
    model_folder = os.path.join(data_folder, args.model_folder)
    if not os.path.exists(data_folder):
        parser.error(f"Folder {data_folder} does not exist")
        exit()
    if not os.path.exists(inference_cpu_folder):
        parser.error(
            f"Folder {inference_cpu_folder} does not exist. The CPU timings must be placed into {inference_cpu_folder}"
        )
        exit()
    if not os.path.exists(inference_gpu_folder):
        parser.error(
            f"Folder {inference_gpu_folder} does not exist. The GPU timings must be placed into {inference_gpu_folder}"
        )
        exit()
    if not os.path.exists(model_folder):
        parser.error(
            f"Folder {model_folder} does not exist. The models must be trained and placed into {model_folder}"
        )
        exit()

    out_folder = f"{data_folder}/{args.out_tables_folder}"
    os.makedirs(out_folder, exist_ok=True)
    inference_cpu_timings = glob.glob(f"{inference_cpu_folder}/*time*.csv")
    inference_gpu_timings = glob.glob(f"{inference_gpu_folder}/*time*.csv")
    train_timing = glob.glob(f"{model_folder}/*time*.csv")

    if len(train_timing) != 1:
        if args.verbose:
            log.error("Cannot generate plot if training timings are not present")
            exit()
    if len(inference_cpu_timings) != 1 or len(inference_gpu_timings) != 1:
        if args.verbose:
            log.error(
                "Cannot generate plot if inference on CPU or inference on GPU timings are not present"
            )
            exit()

    if args.verbose:
        log.info(
            f"Start generating plots for timings and time efficiency, and saving in {out_folder}."
        )

    timings_dfs = {
        "train": pd.read_csv(train_timing[0])
        .dropna()
        .sort_values("model", key=variant_ordering_series_key),
        "inf_cpu": pd.read_csv(inference_cpu_timings[0])
        .dropna()
        .sort_values("model", key=variant_ordering_series_key),
        "inf_gpu": pd.read_csv(inference_gpu_timings[0])
        .dropna()
        .sort_values("model", key=variant_ordering_series_key),
    }

    generate_table(
        timings_dfs=timings_dfs,
        out_folder=out_folder,
    )

    if args.verbose:
        log.info(f"Finished generating tables.")
