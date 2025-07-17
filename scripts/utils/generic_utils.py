import os
import sys
import re
import cv2
import json
import h5py
import tensorflow
import numpy as np
import tifffile as tf
from scipy import ndimage
from msr_reader import OBFFile

common_stack_name_re = r"(\d{4}-\d{2}-\d{2}-m\d+)"


def fadvise(fileno):
    """Hint OS to discard file contents."""
    if sys.platform != "win32":
        os.posix_fadvise(fileno, 0, os.fstat(fileno).st_size, os.POSIX_FADV_DONTNEED)
    else:
        # On Windows or if posix_fadvise is not available, do nothing
        pass


def laplacian_var(img):
    """Compute variance of laplacian of an image. That is, sharpness level.
    Higher variance in laplacian means higher variance in intensity changes.
    """
    laplacian = ndimage.laplace(img)
    # cv2.meanStdDev avoids allocating an extra full‑sized array, unlike cv2.Laplacian().
    _, std = cv2.meanStdDev(laplacian)
    return float(std.item() ** 2)


def sharpness(img):
    """Determines the level of sharpness of a 3D stack.

    Averages the sharpness levels of all XY slices.
    """
    sharpness_vals = []
    for i in range(img.shape[0]):
        z_img = img[i]
        lap_var = laplacian_var(z_img)
        sharpness_vals.append(lap_var)
    return sum(sharpness_vals) / len(sharpness_vals)


def stacksize_2_dic(s):
    return {"sizes": s.sizes}


def to_uint8_minmax(img):
    """Converts an image in float32 to uint8 by minmax scaling."""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img - img_min) / (img_max - img_min)
    return (img_norm * 255).astype(np.uint8)


def process_obf(obf_path, base, out_folder, log):
    """Processes an obf file, which corresponds to a group of 3D microscope images,

    It selects the overall sharpest stack and writes it to the 'out_folder'
    as TIF with the naming '<base>_stack.tif'.

    Additionally, it saves the metadata of the stack (such as, image shapes and resolution)
    to a JSON file '<base>_meta.json'. This metadata is used internally by DeepD3 during the
    data generation phase.
    """
    # Save sharpest stack as TIF.
    try:
        with OBFFile(obf_path) as f:
            max_sharpness = 0
            max_sharpness_idx = 0
            sharpest_stack = None
            for idx in range(f.num_stacks):
                img = f.read_stack(idx)
                img_sharpness = sharpness(img)
                if img_sharpness > max_sharpness:
                    max_sharpness = img_sharpness
                    max_sharpness_idx = idx
                    sharpest_stack = img
            out_tif_path = os.path.join(out_folder, f"{base}_stack.tif")
            tf.imwrite(out_tif_path, to_uint8_minmax(sharpest_stack))

            # Save metadata as JSON.
            meta = {
                # Stack shapes (Z, Y, X).
                "stack_shapes": stacksize_2_dic(f.shapes[max_sharpness_idx]),
                # Resolutions (in meters).
                "pixel_sizes": stacksize_2_dic(f.pixel_sizes[max_sharpness_idx]),
            }
            out_json_path = os.path.join(out_folder, f"{base}_meta.json")
            with open(out_json_path, "w") as jf:
                json.dump(meta, jf, indent=2)

            # Prevent OS from caching contents.
            fileno = f.fd.fileno()
            fadvise(fileno)
        del f
    except Exception as e:
        log.warning(f"Skipping {obf_path} due to error: {e}")
        return



def split_labels_by_size(label_stack, z_thresh=1.5):
    """
    Splits merged labels into dendrite (large) and spine (small) using z-score size thresholding.

    Parameters:
        label_stack (ndarray): 3D labeled mask (0 = background)
        z_thresh (float): Z-score threshold to classify large structures (default = 1.5)

    Returns:
        spine_labels (list): Label IDs classified as spines
        dendrite_labels (list): Label IDs classified as dendrites
        threshold (float): Size threshold used for splitting
    """
    labels = np.unique(label_stack)
    labels = labels[labels != 0]  # remove background label

    label_sizes = {label: np.sum(label_stack == label) for label in labels}

    sizes = np.array(list(label_sizes.values()))

    mean = sizes.mean()
    std = sizes.std()
    threshold = mean + z_thresh * std

    dendrite_labels = [
        label for label, size in label_sizes.items() if size >= threshold
    ]
    spine_labels = [label for label in labels if label not in dendrite_labels]

    return sorted(spine_labels), sorted(dendrite_labels), threshold


def label_masks(hdf5_mask):
    """Converts a list of binary 3D masks into a single labeled mask stack.

    The input 'mask' has a shape indicating how many different masks it containes. For instance,
    mask.shape = (36, 1) indicates that there are 36 masked stacks (36 rows x 1 column elements).
    We assume (know), the shape is two-dimensional.

    Each element [i][j] of this mask corresponds to a mask for the whole stack. Those are the ones
    that we want to overlay.
    """
    curr_ref = hdf5_mask[0, 0]
    curr_mask = hdf5_mask.file[curr_ref][()]
    stack = np.zeros(curr_mask.shape, dtype=np.uint8)
    del curr_mask
    # gc.collect()

    label_idx = 0
    for i in range(hdf5_mask.shape[0]):
        for j in range(hdf5_mask.shape[1]):
            curr_ref = hdf5_mask[i][j]
            curr_mask = hdf5_mask.file[curr_ref][()]
            if len(curr_mask.shape) == 3:
                label_idx += 1
                # Put mask into stack with 'idx' where originally there were a 1.
                stack[curr_mask > 0] = label_idx
            del curr_mask
            # gc.collect()
    return stack


def process_mat(mat_paths, base, out_folder, verbose, log, z_thresh=1.5):
    """Process and merge labeled masks from multiple .mat files for a given dataset."""

    merged_stack = None  # Will hold combined labeled mask stack
    label_offset = 0  # Track label offsets to avoid overlaps

    for mat_path in mat_paths:
        file = h5py.File(mat_path, "r")
        if "mask" not in file:
            if verbose:
                log.info(f"        Skipping {mat_path}: no 'mask' key.")
            continue

        h5df_data = file["mask"]
        curr_label_stack = label_masks(h5df_data)

        if merged_stack is None:
            merged_stack = np.zeros_like(curr_label_stack, dtype=np.uint16)

        # Shift labels in current stack to avoid overlap
        curr_label_stack = curr_label_stack.astype(np.uint16)
        curr_label_stack[curr_label_stack > 0] += label_offset

        # Merge nonzero voxels into merged_stack
        nonzero_mask = curr_label_stack > 0
        merged_stack[nonzero_mask] = curr_label_stack[nonzero_mask]

        # Update label_offset for next round
        label_offset = merged_stack.max()

    if merged_stack is None:
        log.warning(f"        No valid masks found for base {base}. Skipping.")
        return None

    # Save merged labeled mask as tif
    tif_save_path = os.path.join(out_folder, f"{base}_merged_labels.tif")
    tf.imwrite(tif_save_path, merged_stack)
    if verbose:
        log.info(f"        Saved merged labeled mask as .mat: {tif_save_path}")

    # Now split into spines and dendrite
    spine_labels, dendrite_labels, threshold = split_labels_by_size(
        merged_stack, z_thresh=z_thresh
    )

    # Create binary masks from labels
    spines = np.isin(merged_stack, spine_labels).astype(np.uint8) * 255
    dendrite = np.isin(merged_stack, dendrite_labels).astype(np.uint8) * 255

    label_dict = {
        "dendrite": dendrite_labels,
        "spine": spine_labels,
        "threshold": threshold,
    }

    # Save spines and dendrite masks as TIF.
    tf.imwrite(os.path.join(out_folder, f"{base}_spines.tif"), spines)
    tf.imwrite(os.path.join(out_folder, f"{base}_dendrite.tif"), dendrite)

    return label_dict


def extract_base(filename):
    """Extract matching base part from filename.
    First match of the regex, default to the part of the filename before first "_".
    """
    base = os.path.basename(filename)
    m = re.match(common_stack_name_re, base)
    return m.group(1) if m else re.split(r"_", base)[0]


def add_file(path: str, key: str, datasets):
    """Group files by base filename."""
    base = extract_base(path)
    datasets[base][key] = path


def add_file_as_tuple(path: str, key: str, datasets):
    """Group files by base filename."""
    base = extract_base(path)
    datasets[base][key] += (path,)


def schedule(epoch, lr):
    if epoch < 15:
        return lr

    else:
        return lr * tensorflow.math.exp(-0.1)
