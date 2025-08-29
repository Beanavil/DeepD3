# Local imports
import scripts.utils.plot_utils as pu
from scripts.utils.generic_utils import variant_ordering_key

# Others
import os
import glob
import logging
import argparse
import tifffile
import matplotlib
import pandas as pd
import rich.logging
from scipy.stats import kruskal, shapiro


def analyze(
    infer_spines,
    gt_spines,
    metrics,
    out_plots_folder,
    metrics_type,
    suffix="deepd3",
    prediction_thres=0,
):
    # We store columns for DeepD3 scaling variant (8f, ...) and metric names
    # So for each scaling variant, we store for each metric its values across benchmarks
    metrics_results = []
    gt_spines = gt_spines
    for b in range(len(gt_spines)):
        bench = gt_spines[b]
        ground_truth = tifffile.imread(bench)
        ground_truth = (ground_truth > 0).astype(bool)
        for variant_inference in infer_spines:
            prediction = tifffile.imread(infer_spines[variant_inference][b])
            prediction = (prediction > prediction_thres).astype(bool)
            # Compute metrics
            results = pu.compute_infer_metrics(ground_truth, prediction, metrics)
            # Save metrics
            metrics_results.append(
                {"DeepD3 scaling variant": variant_inference, **results}
            )
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv(
        os.path.join(out_plots_folder, f"metrics_{metrics_type}_{suffix}.csv"),
        index=False,
    )
    return metrics_df


def analyze_seg_quality(
    infer_spines, gt_spines, out_plots_folder, out_tables_folder, suffix="deepd3"
):
    """Analyze segmentation quality with recall score. Generate plot over benchmarks and summary table."""
    if os.path.exists(os.path.join(out_plots_folder, f"metrics_quality_{suffix}.csv")):
        metrics_df = pd.read_csv(
            os.path.join(out_plots_folder, f"metrics_quality_{suffix}.csv")
        )
    else:
        metrics_df = analyze(
            infer_spines=infer_spines,
            gt_spines=gt_spines,
            metrics=["recall"],
            out_plots_folder=out_plots_folder,
            metrics_type="quality",
            suffix=suffix,
        )
    pu.plot_infer_metrics(
        metrics_df=metrics_df,
        x_max=len(gt_spines),
        out_plots_folder=out_plots_folder,
        suffix=suffix,
    )
    pu.create_mean_std_table(
        metrics_df=metrics_df,
        out_tables_folder=out_tables_folder,
        caption=(
            "Segmentation quality metrics for different DeepD3 scaling variants "
            "(8, 16, 32 and 64 filters)."
        ),
        label=f"tab:seg_quality_metrics_{suffix}",
        suffix=suffix,
        index=True,
    )


def analyze_seg_agreement(
    infer_spines, gt_spines, out_plots_folder, out_tables_folder, suffix="deepd3"
):
    """Analyze segmentation agreement with IoU (overlap-based) and NSD (boundary-based) scores. Generate plot over benchmarks and summary table."""
    if os.path.exists(
        os.path.join(out_plots_folder, f"metrics_agreement_{suffix}.csv")
    ):
        metrics_df = pd.read_csv(
            os.path.join(out_plots_folder, f"metrics_agreement_{suffix}.csv")
        )
    else:
        metrics_df = analyze(
            infer_spines=infer_spines,
            gt_spines=gt_spines,
            metrics=["IoU", "NSD"],
            prediction_thres=80,
            out_plots_folder=out_plots_folder,
            metrics_type="agreement",
            suffix=suffix,
        )
    pu.plot_infer_metrics(
        metrics_df=metrics_df,
        x_max=len(gt_spines),
        out_plots_folder=out_plots_folder,
        suffix=suffix,
    )
    pu.create_mean_std_table(
        metrics_df=metrics_df,
        out_tables_folder=out_tables_folder,
        caption=(
            "Segmentation agreement metrics for different DeepD3 scaling variants "
            "(8, 16, 32 and 64 filters)."
        ),
        label=f"tab:seg_agreement_metrics_{suffix}",
        suffix=suffix,
        index=False,
    )


def print_shapiro_res(res, conf_level=0.01):
    print(
        f"Shapiro Wilk test for normality. Statistic = {res.statistic}, p-value = {res.pvalue} -> {"Not normal" if res.pvalue < conf_level else "Normal"}"
    )


def print_kruskall_res(res, conf_level=0.01):
    print(
        f"Kruskal-Wallis test. Statistic = {res.statistic}, p-value = {res.pvalue} -> {"Diffs significative" if res.pvalue < conf_level else "Diffs not significative"}"
    )


def analyze_diff():
    deepd3_qual_df = pd.read_csv(
        os.path.join(out_plots_folder, f"metrics_quality_deepd3.csv")
    )
    base_qual_df = pd.read_csv(
        os.path.join(out_plots_folder, f"metrics_quality_deepd3_base.csv")
    )
    # Normality tests
    print_shapiro_res(shapiro(deepd3_qual_df["recall"]))
    print_shapiro_res(shapiro(base_qual_df["recall"]))
    # Analysis of variance
    print_kruskall_res(kruskal(deepd3_qual_df["recall"], base_qual_df["recall"]))

    deepd3_agree_df = pd.read_csv(
        os.path.join(out_plots_folder, f"metrics_agreement_deepd3.csv")
    )
    base_agree_df = pd.read_csv(
        os.path.join(out_plots_folder, f"metrics_agreement_deepd3_base.csv")
    )
    # Normality test
    print_shapiro_res(shapiro(deepd3_agree_df["IoU"]))
    print_shapiro_res(shapiro(base_agree_df["IoU"]))
    print_shapiro_res(shapiro(deepd3_agree_df["NSD"]))
    print_shapiro_res(shapiro(base_agree_df["NSD"]))
    # Analysis of variance
    print_kruskall_res(kruskal(deepd3_agree_df["IoU"], base_agree_df["IoU"]))
    print_kruskall_res(kruskal(deepd3_agree_df["NSD"], base_agree_df["NSD"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_performance_metrics",
        description="Generate plots for metrics values from model inference results for BIMAP P4",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-a",
        "--animal",
        default="turtle",
        help="Animal for which inference was be done. Default to 'turtle'",
    )
    parser.add_argument(
        "-b",
        "--benchmark-folder",
        default="benchmarking",
        help="Path to the folder containing the benchmarking dataset for DeepD3, relative to args.path. Default to 'benchmarking' (that is, args.path/benchmarking).",
    )
    parser.add_argument(
        "-ub",
        "--use-base",
        action="store_true",
        help="Whether to use inference data from the model trained without floodfilled masks.",
    )
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
        default=55,
        help="Figure width for plots, in millimeters. Default to 55",
    )
    parser.add_argument(
        "-m",
        "--inference-folder",
        default="inference_gpu",
        help="Path to the folder containing the inference results for DeepD3, relative to args.path. Default to 'inference_gpu' (that is, args.path/inference_gpu).",
    )
    parser.add_argument(
        "-op",
        "--out-plots-folder",
        default="plots",
        help="Path to the folder to contain the generated plots, relative to args.path. Default to 'plots' (that is, args.path/plots).",
    )
    parser.add_argument(
        "-ot",
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
    gt_folder = os.path.join(
        data_folder,
        os.path.join(args.benchmark_folder, args.animal),
    )
    flood_gt_folder = os.path.join(gt_folder, "floodfill")
    infer_folder = os.path.join(
        data_folder, args.inference_folder + ("_base" if args.use_base else "")
    )

    if not os.path.exists(data_folder):
        parser.error(f"Folder {data_folder} does not exist")
        exit()
    if not os.path.exists(gt_folder):
        parser.error(f"Folder {gt_folder} does not exist")
        exit()
    if not os.path.exists(flood_gt_folder):
        parser.error(f"Folder {flood_gt_folder} does not exist")
        exit()
    if not os.path.exists(infer_folder):
        parser.error(f"Folder {infer_folder} does not exist")
        exit()

    out_plots_folder = f"{data_folder}/{args.out_plots_folder}"
    out_tables_folder = f"{data_folder}/{args.out_tables_folder}"
    os.makedirs(out_plots_folder, exist_ok=True)
    os.makedirs(out_tables_folder, exist_ok=True)

    # Organize results indexing by filter count of the model
    deepdd3_infer_folders = [
        f.path
        for f in os.scandir(infer_folder)
        if f.is_dir() and f.name.startswith("DeepD3")
    ]
    deepdd3_infer_folders = sorted(deepdd3_infer_folders, key=variant_ordering_key)

    if args.verbose:
        log.info(
            f"Found {len(deepdd3_infer_folders)} folders for DeepD3 inference data. Generating plots for them and saving in {out_plots_folder}."
        )

    gt_spines = sorted(glob.glob(f"{gt_folder}/*_spines.tif"))
    infer_spines = {}
    for folder in deepdd3_infer_folders:
        basename = os.path.splitext(os.path.basename(folder))[0]
        filters = pu.extract_filters(basename)
        infer_spines[filters] = sorted(glob.glob(f"{folder}/*_spines.tif"))

    # Perform segmentation quality and agreement analysis
    analyze_seg_quality(
        infer_spines=infer_spines,
        gt_spines=gt_spines,
        out_plots_folder=out_plots_folder,
        out_tables_folder=out_tables_folder,
        suffix="deepd3" + ("_base" if args.use_base else ""),
    )

    gt_spines = sorted(glob.glob(f"{flood_gt_folder}/*_spines.tif"))
    analyze_seg_agreement(
        infer_spines=infer_spines,
        gt_spines=gt_spines,
        out_plots_folder=out_plots_folder,
        out_tables_folder=out_tables_folder,
        suffix="deepd3" + ("_base" if args.use_base else ""),
    )

    # Analyze significance of results
    analyze_diff()

    if args.verbose:
        log.info(f"Finished generating plots.")
