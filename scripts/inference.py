# Local imports
from deepd3.core.analysis import Stack
from deepd3.core.export import ExportPredictions
from scripts.utils.generic_utils import extract_base

# Others
import os
import glob
import logging
import argparse
import tifffile
import numpy as np
import pandas as pd
import rich.logging
import tensorflow as tf
from PyQt5.QtCore import QObject
from timeit import default_timer as timer


class ExportPredictions(QObject):
    def __init__(self, pred_spines, pred_dendrites, max_proj):
        """Class to export ROIs to a folder. Modified from deepd3.core.export.ExportPredictions.

        Args:
            rois (dict): dictionary of ROIs, keys are z-plane, value is list with ROIs in given plane.
        """
        super().__init__()
        self.pred_spines = pred_spines
        self.pred_dendrites = pred_dendrites
        self.max_proj = max_proj

    def export(self, basename, out_folder):
        """Export predictions as tif files.

        Args:
            fn (str): out file basename
            folder (str): target folder
        """
        target_spines = os.path.join(out_folder, basename + "_pred_spines.tif")
        target_dendrites = os.path.join(out_folder, basename + "_pred_dendrites.tif")
        target_proj = os.path.join(out_folder, basename + "_max_proj.tif")

        try:
            tifffile.imwrite(target_spines, (self.pred_spines * 255).astype(np.uint8))
            tifffile.imwrite(
                target_dendrites, (self.pred_dendrites * 255).astype(np.uint8)
            )
            tifffile.imwrite(target_proj, self.max_proj.astype(np.float32))

            return True, target_spines + "\n" + target_dendrites + "\n" + target_proj

        except Exception as e:
            return False, e


def inference(models_folder: str, bench_folder: str, out_folder: str, context: str):
    models = glob.glob(f"{models_folder}/*.h5")
    bench_stacks = glob.glob(f"{bench_folder}/*_stack.tif")

    if args.verbose:
        log.info(f"    Found {len(models)} models")

    timings = []

    for model_fn in models:
        if args.verbose:
            log.info(f"        Benchmarking {model_fn}")

        model_name = os.path.splitext(os.path.basename(model_fn))[0]
        out_model_inference_folder = os.path.join(out_folder, model_name)
        os.makedirs(out_model_inference_folder, exist_ok=True)

        for bench_stack in bench_stacks:
            S = Stack(bench_stack)
            basename = extract_base(bench_stack)
            with tf.device(context):
                time_start = timer()
                S.predictWholeImage(model_fn)
                time_end = timer()
                if S.segmented:
                    # Compute inference time
                    inference_time_ms = (time_end - time_start) * 1000
                    timings.append(
                        {
                            "model": model_name,
                            "stack": basename,
                            "inference_time_ms": inference_time_ms,
                        }
                    )
                    # Export predictions
                    ep = ExportPredictions(
                        S.prediction[..., 1],  # spines pred
                        S.prediction[..., 0],  # dendrite pred
                        S.prediction.max(0),  # max projection
                    )
                    r, e = ep.export(basename, out_model_inference_folder)
                    if args.verbose:
                        if r:
                            log.info(f"Exported! Successfully exported to \n{e}")
                        else:
                            log.error(
                                f"Failed to export! Problems to export the data... \n{e}"
                            )
        # Average inference time over benchmarking stacks per model, and write
        df = pd.DataFrame(timings)
        df_avg = df.groupby("model")["inference_time_ms"].mean().reset_index()
        df_avg.rename(
            columns={"inference_time_ms": "avg_inference_time_ms"}, inplace=True
        )
        timing_file = os.path.join(
            out_folder, f"inference_time_{"gpu" if args.run_gpu else "cpu"}.csv"
        )
        df_avg.to_csv(timing_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="inference",
        description="Benchmark inference of trained models model for BIMAP P4 turtle and mice brain images semantic segmentation, and produce segmentation masks for further analysis",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-a",
        "--animal",
        default="turtle",
        help="Animal for which inference will be done. Default to 'turtle'",
    )
    parser.add_argument(
        "-b",
        "--benchmark-folder",
        default="benchmarking",
        help="Path to the folder that contains the benchmarking stacks, relative to args.path. Default to 'benchmarking' (that is, args.path/benchmarking).",
    )
    parser.add_argument(
        "-m",
        "--models-folder",
        default="models",
        help="Path to the folder containing the trained models, relative to args.path. Default to 'models' (that is, args.path/models).",
    )
    parser.add_argument(
        "-o",
        "--out-folder-prefix",
        default="inference",
        help="Path to the folder to contain the trained models, relative to args.path. Default to 'models' (that is, args.path/models).",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=f"{current_folder}/images",
        help="Path to the folder containing the raw data. Default to 'images' on the current folder.",
    )
    parser.add_argument(
        "-r",
        "--run-gpu",
        action="store_true",
        help="Whether to run inference on the gpu.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Add a logger and a log file.
    log_level = logging.INFO
    log_file = os.path.join(args.path, "inference_log.txt")
    logging.basicConfig(
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            rich.logging.RichHandler(console=None, rich_tracebacks=True, markup=True),
        ],
        level=log_level,
    )
    log = logging.getLogger("rich")

    # Validate input folder(s) existence and create output folder
    data_folder = args.path
    out_folder = os.path.join(data_folder, "inference")
    bench_folder = os.path.join(
        data_folder, os.path.join(args.benchmark_folder, args.animal)
    )
    models_folder = os.path.join(data_folder, args.models_folder)

    if not os.path.exists(data_folder):
        parser.error(f"Folder {data_folder} does not exist")
    if not os.path.exists(bench_folder):
        parser.error(f"Folder {bench_folder} does not exist")
    if not os.path.exists(models_folder):
        parser.error(f"Folder {models_folder} does not exist")
    os.makedirs(out_folder, exist_ok=True)

    # Set device to be used by Tensorflow
    if args.run_gpu:
        context = "/gpu:0"
    else:
        context = "/cpu:0"

    if args.verbose:
        log.info(f"Started inference for models from {models_folder} on {context}")

    inference(
        bench_folder=bench_folder,
        models_folder=models_folder,
        out_folder=out_folder,
        context=context,
    )

    if args.verbose:
        log.info(f"Finished processing all data")
