# Local imports
from deepd3.model import DeepD3_Model
from deepd3.utils.floodfill import floodfill_stacks
from deepd3.training.tile import TiledDataGenerator, Stack

# Set keras framework
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from test_utils import add_file, schedule

# Others
import glob
import json
import logging
import pathlib
import argparse
import rich.logging
from itertools import groupby
from collections import defaultdict
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

# Fixed training parameters
res = 0.02  # Fixed to 20 nm, can be None for mixed resolution training
sample_shape = (1, 256, 256)  # Bigger tiles since we have more resolution


def train_model(in_folder: str, out_folder: str, animal: str):
    # Get all files from the four types (original stack, spines masks, dendrites masks, metadata).
    stack_files = glob.glob(f"{in_folder}/*_stack.tif")
    spines_files = glob.glob(f"{in_folder}/*_spines.tif")
    dendrite_files = glob.glob(f"{in_folder}/*_dendrite.tif")
    meta_files = glob.glob(f"{in_folder}/*_meta.json")

    # Group files per base name.
    datasets = defaultdict(dict)
    for f in stack_files:
        add_file(f, "stack", datasets)
    for f in spines_files:
        add_file(f, "spines", datasets)
    for f in dendrite_files:
        add_file(f, "dendrite", datasets)
    for f in meta_files:
        add_file(f, "meta", datasets)
    if args.verbose:
        log.info(f"        Found {len(datasets)} datasets")

    stack_list = []

    # Create Stack objects for the tiled generator.
    for base in datasets:
        try:
            stack_path = datasets[base]["stack"]
            spines_path = datasets[base]["spines"]
            dendrite_path = datasets[base]["dendrite"]
            meta_path = datasets[base]["meta"]

            with open(meta_path) as jf:
                meta = json.load(jf)

            stack_list.append(
                Stack(
                    img=stack_path, s_masks=spines_path, d_mask=dendrite_path, meta=meta
                )
            )
        except Exception as e:
            log.warning(f"Skipping {base} due to error: {e}")
            continue

    if len(stack_list) == 0:
        log.error("No train data, stopping execution")
        exit(0)
    elif args.verbose:
        log.info(f"Using {len(stack_list)} stacks for training")

    if args.floodfill:
        # Floodfill all the spines' masks and write back to disk.
        if args.verbose:
            log.info("Floodfilling spines masks")

        floodfill_stacks(fn=stack_list)

        if args.verbose:
            log.info(
                "Floodfilled spines masks. For training model, rerun the script without -flo (--floodfill)"
            )
        return

    # Separate training and test (validation) data.
    # By default, train_test_split will use a random 25% of the available data for validation and
    # the complementary, 75%, for training. We pass and int as random state for reproducible trainings.
    train_stack_list, validate_stack_list =  train_test_split(stack_list, random_state=42)

    # Create the tiled samples from raw images and (for spines, floodfilled) masks.
    if args.verbose:
        log.info("Creating tiled stacks from training and validation data")
    dg_training = TiledDataGenerator(
        fn=train_stack_list,
        batch_size=args.batch_size,
        target_resolution=res,
        size=sample_shape,
    )
    dg_validation = TiledDataGenerator(
        fn=validate_stack_list,
        batch_size=args.batch_size,
        target_resolution=res,
        size=sample_shape,
    )

    # Compile model.
    # Create a naive DeepD3 model with a given base filter count.
    model = DeepD3_Model(filters=args.filters)

    # Set appropriate training settings
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        # Dice loss for dendrite, MSE for spines.
        loss=[sm.losses.dice_loss, "mse"],
        metrics=["acc", sm.metrics.iou_score],
    )
    if args.verbose:
        log.info(model.summary())

    # Train model.
    # Save best model automatically during training.
    mc = ModelCheckpoint(
        f"{out_folder}/DeepD3_trained_model_{animal}.h5", save_best_only=True
    )

    # Save metrics.
    csv = CSVLogger(f"{out_folder}/DeepD3_test_model_{animal}.csv")

    # Adjust learning rate during training to allow for better convergence.
    lrs = LearningRateScheduler(schedule)

    # Actually train the network.
    h = model.fit(
        dg_training,
        batch_size=dg_training.batch_size,
        epochs=args.epochs,
        validation_data=dg_validation,
        callbacks=[mc, csv, lrs],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_model",
        description="Train model for BIMAP P4 turtle and mice brain images semantic segmentation, and produce a model for inference",
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=32,
        help="Images processed at once, best value will depend on the GPU. Default to 32.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="Epochs to be used for training. Default to 30.",
    )
    parser.add_argument(
        "-f",
        "--filters",
        type=int,
        default=8,
        help="Filters to be used for the model. Default to 8.",
    )
    parser.add_argument(
        "-flo",
        "--floodfill",
        action="store_true",
        help="Whether to floodfill masks before training.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate to be used for training. Default to 0.0005.",
    )
    parser.add_argument(
        "-m",
        "--out-models-folder",
        default="models",
        help="Path to the folder to contain the trained models, relative to args.path. Default to 'models' (that is, args.path/models).",
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Add a logger and a log file.
    log_level = logging.INFO
    log_file = os.path.join(args.path, "training_log.txt")
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

    animals_data.pop(args.out_models_folder, None)

    for animal, data_subfolders in animals_data.items():
        if args.verbose:
            log.info(f"Training model for {animal}")
        out_folder = f"{data_folder}/{args.out_models_folder}/{animal}"
        if not args.floodfill: os.makedirs(out_folder, exist_ok=True)
        for subfolder in data_subfolders:
            if args.verbose:
                log.info(f"    Training model on data from {subfolder}")
            train_model(in_folder=subfolder, out_folder=out_folder, animal=animal)
            if args.verbose:
                log.info(f"    Finished training from {subfolder}")
        if args.verbose:
            log.info(
                f"Finished training model for {animal}. Model written to {out_folder}"
            )

    if args.verbose:
        log.info(f"Finished processing all data")
