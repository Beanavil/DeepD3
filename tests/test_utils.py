import os
import sys
import re
import cv2
import json
import h5py
import tensorflow
import scipy
import numpy as np
import tifffile as tf
from scipy import ndimage
from msr_reader import OBFFile

common_stack_name_re = r"(\d{4}-\d{2}-\d{2}-m\d+)"


def fadvise(fileno):
    """ Hint OS to discard file contents."""
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
        # if z_img.dtype != np.uint8:
        #     z_img = cv2.normalize(z_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lap_var = laplacian_var(z_img)
        sharpness_vals.append(lap_var)
    return sum(sharpness_vals) / len(sharpness_vals)


def stacksize_2_dic(s):
    return {"sizes": s.sizes}


def process_obf(obf_path, base, out_folder):
    """Processes an obf file, which corresponds to a group of 3D microscope images,

    It selects the overall sharpest stack and writes it to the 'out_folder'
    as TIF with the naming '<base>_stack.tif'.

    Additionally, it saves the metadata of the stack (such as, image shapes and resolution)
    to a JSON file '<base>_meta.json'. This metadata is used internally by DeepD3 during the
    data generation phase.
    """
    # Save sharpest stack as TIF.
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
        tf.imwrite(out_tif_path, sharpest_stack)

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


def split_masks(stack):
    """Splits masks into dendrites and spines.

    Dendrite mask doesn't necessarily have the same position always. It is detected by assuming
    that it has the highest amount of masked pixels from all the masks.
    """
    max_pixel_count = 0
    dendrite_idx = 0
    for idx in range(1, stack.max() + 1):
        pixel_count = np.sum(stack == idx)
        if pixel_count > max_pixel_count:
            max_pixel_count = pixel_count
            dendrite_idx = idx
    spines = ((stack > 0) & (stack != dendrite_idx)).astype(np.uint8) * 255
    dendrite = (stack == dendrite_idx).astype(np.uint8) * 255
    return spines, dendrite, dendrite_idx

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

def process_mat(mat_paths, base, out_folder, verbose, log):
    """Process and merge labeled masks from multiple .mat files for a given dataset."""
    
    merged_stack = None  # Will hold combined labeled mask stack
    label_offset = 0     # Track label offsets to avoid overlaps

    for mat_path in mat_paths:
        file = h5py.File(mat_path, 'r')
        if 'mask' not in file:
            if verbose:
                log.info(f"        Skipping {mat_path}: no 'mask' key.")
            continue

        h5df_data = file['mask']
        curr_label_stack = label_masks(h5df_data)  # Your function that returns labeled 3D mask

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

    # Optional: save merged labeled mask as .mat
    mat_save_path = os.path.join(out_folder, f"{base}_merged_labels.mat")
    scipy.io.savemat(mat_save_path, {"merged_labels": merged_stack})
    if verbose:
        log.info(f"        Saved merged labeled mask as .mat: {mat_save_path}")

    # Now split into spines and dendrite
    spines, dendrite, dendrite_idx = split_masks(merged_stack)

    # Save spines and dendrite masks as TIF.
    tf.imwrite(os.path.join(out_folder, f"{base}_spines.tif"), spines)
    tf.imwrite(os.path.join(out_folder, f"{base}_dendrite.tif"), dendrite)

    return dendrite_idx


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
