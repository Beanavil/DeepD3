# Local imports
from scripts.utils.generic_utils import add_file_as_tuple, process_obf, process_mat

# Others
import os
import glob
import logging
import pathlib
import argparse
import rich.logging
from itertools import groupby
from collections import defaultdict


def process_raw_data_subfolder(in_folder: str, out_folder: str):
    """Core of the raw data processing pipeline.

    Processes all the raw data from one subfolder for one animal/species and writes the
    processed data to the specified output folder.
    """
    # Get all files from the three types (stack, masks, skeletons).
    obf_files = glob.glob(f'{in_folder}/*.obf')
    mat_files = glob.glob(f'{in_folder}/*.mat')
    nml_files = glob.glob(f'{in_folder}/*.nml')

    # Group files per dataset. Each dataset has at least one file of each type but some
    # have more than one of each.
    datasets = defaultdict(lambda: {'obf': [], 'mat': [], 'nml': []})
    for f in obf_files:
        add_file_as_tuple(f, 'obf', datasets)
    for f in mat_files:
        add_file_as_tuple(f, 'mat', datasets)
    for f in nml_files:
        add_file_as_tuple(f, 'nml', datasets)
    if args.verbose:
        log.info(f'        Found {len(datasets)} datasets')

    # Process original microscope image (.obf).
    for base in datasets.keys():
        for obf_file in datasets[base]['obf']:
            process_obf(obf_file, base, out_folder, log)
    if args.verbose:
        log.info(f'        Processed obf files')

    # Process masks (.mat).
    # We get the dendrite and spine masks in the same file. What we need to accomplish is to
    # separate the dendrite from the spines masks, as well as to separate each slice of spines'
    # masks (that is, to overlay the masks for the dendrite and the differenct XY slices).
    for base in datasets.keys():
        mat_file_list = datasets[base]['mat']
        if mat_file_list:
            dendrite_idx = process_mat(
                mat_file_list, base, out_folder, args.verbose, log
            )
            log.info(
                f'{base} dataset used {len(mat_file_list)} mat files. stack from {base} dataset using dendrite mask from mask {dendrite_idx}'
            )

    if args.verbose:
        log.info(f'        Processed mat files')

    # Process skelonization files (.nml).
    # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='process_raw_data',
        description='Process BIMAP P4 turtle and mice raw data, and produce usable stacks',
    )

    current_folder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument(
        '-a',
        '--animals',
        default='turtle, mice',
        help='Comma-separated list of animals (\'animal1, animal2, ..., animalX\') for which data is provided, such that <animal_name>_* folder(s) exist in the input data folder. Default to \'turtle, mice\'',
    )
    parser.add_argument(
        '-o',
        '--out-folder',
        default='processed',
        help='Path to the folder to contain the processed data, relative to args.path. Default to \'processed\' (that is, args.path/processed).',
    )
    parser.add_argument(
        '-p',
        '--path',
        default=f'{current_folder}/images',
        help='Path to the folder containing the raw data. Default to \'images\' on the current folder.',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Add a logger and a log file.
    log_level = logging.INFO
    log_file = os.path.join(args.path, 'preprocessing_log.txt')
    logging.basicConfig(format='%(message)s', handlers=[logging.FileHandler(log_file, mode='a'), rich.logging.RichHandler(
        console=None, rich_tracebacks=True, markup=True)], level=log_level)
    log = logging.getLogger('rich')

    # Validate input folder contents.
    data_folder = args.path

    # Validate existence of folder.
    if not os.path.exists(data_folder):
        parser.error(f'Folder {data_folder} does not exist')
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    subfolders.sort()

    # Validate existence of subfolders for each animal.
    def folder_key(f_name):
        return pathlib.PurePath(f_name).name.split('_')[0]

    animals_data = {
        gr: list(items) for gr, items in groupby(subfolders, key=folder_key)
    }
    if len(animals_data) == 0:
        parser.error(
            'There must be at least one data folder for one of the animals (turtel, mice)'
        )

    # Remove unexpected folders.
    valid_animals = [a.strip() for a in args.animals.split(',')]
    for key in list(animals_data.keys()):
        if key not in valid_animals:
            animals_data.pop(key)

    # Process data from the found folders.
    for animal, data_subfolders in animals_data.items():
        if args.verbose:
            log.info(f'Processing data for {animal}')
        out_folder = f'{data_folder}/{args.out_folder}/{animal}'
        os.makedirs(out_folder, exist_ok=True)
        for subfolder in data_subfolders:
            if args.verbose:
                log.info(f'    Processing data from {subfolder}')
            process_raw_data_subfolder(in_folder=subfolder, out_folder=out_folder)
            if args.verbose:
                log.info(f'    Finished processing data from {subfolder}')
        if args.verbose:
            log.info(
                f'Finished processing data for {animal}. Data written to {out_folder}'
            )

    if args.verbose:
        log.info(f'Finished processing all data')
