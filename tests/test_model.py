import os
import re
import glob
from collections import defaultdict
from msr_reader import OBFFile
import tifffile as tf
import cv2
import json
import numpy as np
import flammkuchen as fl
import sys
import IPython
from pathlib import Path
from deepd3.training.tile import TiledDataGenerator, Stack
from deepd3.utils.floodfill import floodfill_stacks
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
import tensorflow

current_path = "/".join(
    IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[-5:]
)
deepd3_root_path = Path(current_path).resolve().parent.parent
sys.path.insert(0, f"{deepd3_root_path}")
sys.path.insert(0, f"{deepd3_root_path}/deepd3")

img_folder = "test_images"
animal = "turtle" # or mice
OUT_FOLDER = f"{img_folder}/out/{animal}"
RAW_FOLDER = f"{OUT_FOLDER}_raw"
os.makedirs(OUT_FOLDER, exist_ok=True)

# get all files
obf_files = glob.glob(f"{RAW_FOLDER}/*.obf")
mat_files = glob.glob(f"{RAW_FOLDER}/*.mat")
nml_files = glob.glob(f"{RAW_FOLDER}/*.nml")

# extract matching base (before "_deconv")
def extract_base(filename):
    return re.split(r"_deconv", os.path.basename(filename))[0]

def normalized_float2int8(img):
    return (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)

def laplacian_var(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian.var()

def sharpness(img):
    sharpness_vals = []
    for i in range(img.shape[0]):
        z_img = img[i]
        if z_img.dtype != np.uint8:
            z_img = normalized_float2int8(z_img)
        lap_var = laplacian_var(z_img)
        sharpness_vals.append(lap_var)
    return sum(sharpness_vals) / len(sharpness_vals)

def process_obf(obf_path, out_folder):
    with OBFFile(obf_path) as f:
        max_sharpness = 0
        max_sharpness_idx = 0

        # reading image data
        for idx in range(f.num_stacks):
            img = f.read_stack(idx)    # read stack with index idx into numpy array

            # sharpness
            img_sharpness = sharpness(img)
            if img_sharpness > max_sharpness:
                max_sharpness = img_sharpness
                max_sharpness_idx = idx

        sharpest_stack = f.read_stack(max_sharpness_idx)
        base_name = os.path.splitext(os.path.basename(obf_path))[0]
        out_tif_path = os.path.join(out_folder, f"{base_name}_sharpest.tif")
        tf.imwrite(out_tif_path, sharpest_stack)

        # save metadata as JSON
        metadata = {
            "pixel_sizes": f.pixel_sizes[max_sharpness_idx], # like shapes, but with pixel sizes (unit: meters)
            "stack_shape": f.shapes[max_sharpness_idx] # list of stack shapes, including stack and dimension names
        }
        out_json_path = os.path.join(out_folder, f"{base_name}_metadata.json")
        with open(out_json_path, "w") as jf:
            json.dump(metadata, jf, indent=2)

        return out_tif_path

def split_stack_into_dendrite_and_spines(stack):
    max_pixel_count = 0
    dendrite_idx = 1#stack.max()

    # for i in range(1, stack.max() + 1):
    #     pixel_count = np.sum(stack == i)
    #     if pixel_count > max_pixel_count:
    #         max_pixel_count = pixel_count
    #         dendrite_idx = i
    # print(f"Mask {dendrite_idx} is most likely the dendrite, with {max_pixel_count} active pixels")

    spines = ((stack > 0) & (stack != dendrite_idx)).astype(np.uint8) * 255
    dendrite = (stack == dendrite_idx).astype(np.uint8) * 255
    return spines, dendrite, dendrite_idx


# We want to overlay all the masks in a same stack. We start with the very first mask, and
# put the rest on top of it with different values (i.e. binary masks converted from 0/1 to 0/idx)

# mask has shape (36, 1), meaning that it has 36 rows of 1 element each
def combine_masks_to_stack(mask):
    # Convert list of binary 3D masks into a single labeled mask stack.
    stack = np.zeros(mask[0][0].shape, dtype=np.uint8)
    label_idx = 0  

    for i in range(mask.shape[0]):
        # The shape should be (17, 520, 1630), meaning that it's a 17x520x1630 (OZxOYxOX) 3D binary mask
        if len(mask[i][0].shape) == 3:
            label_idx += 1
            # Put mask into stack with 'idx' where originally there were 1
            stack[mask[i][0] > 0] = label_idx
    return stack

# debug method for quick check
def quick_check(masks):
    for i, m in enumerate(masks):
        print(f"Item {i}: type = {type(m)}")
        print("Keys:", list(m.keys()))
        for k, v in m.items():
            print(f"  {k}: type = {type(v)}, shape = {getattr(v, 'shape', 'N/A')}")

    # Each mask element has:
    # - #refs#
    # - mask
    # Each of those has a type and a shape:
    # - #refs#: type dictionary, no shape
    # - mask: type numpy array, shape (rows, cols)

def schedule(epoch, lr):
    if epoch < 15:
        return lr

    else:
        return lr * tensorflow.math.exp(-0.1)


## Reading in the raw file format (OBF, .sec)
# group files by base
datasets = defaultdict(dict)
for f in obf_files:
    base = extract_base(f)
    datasets[base]['obf'] = f
for f in mat_files:
    base = extract_base(f)
    datasets[base]['mat'] = f
for f in nml_files:
    # loosely match with the nearest base using partial matching
    for base in datasets:
        if base in f:
            datasets[base]['nml'] = f
            break

for base, files in datasets.items():
    for k, v in files.items():
        print(f"  {k}: {os.path.basename(v)}")

for base, files in datasets.items():
    if 'obf' in files:
        process_obf(files['obf'], OUT_FOLDER)

## Semantic segmentation
### Saving the .mat files and spines + dendrites
for mat_path in mat_files:
    data = fl.load(mat_path)
    if 'mask' not in data:
        print(f"Skipping {mat_path}, no 'mask' key.")
        continue

    mask_array = data['mask']

    label_stack = combine_masks_to_stack(mask_array)
    base = os.path.splitext(os.path.basename(mat_path))[0]

    # Save stack (.mat file) as tif
    stack_path = os.path.join(OUT_FOLDER, f"{base}_stack.tif")
    tf.imwrite(stack_path, label_stack.astype(np.uint8))

    # Semantic segmentation data: spines and dendrite
    spines, dendrite, dendrite_idx = split_stack_into_dendrite_and_spines(label_stack)

    # Save spines and dendrite as tif
    tf.imwrite(os.path.join(OUT_FOLDER, f"{base}_spines.tif"), spines)
    tf.imwrite(os.path.join(OUT_FOLDER, f"{base}_dendrite.tif"), dendrite)

    print(f"{base}: stack saved, dendrite={dendrite_idx}, spines + dendrite masks saved.")

## Load training data
# Set model config parameters
batch_size = 32               # Data processed at once, depends on your GPU
res = 0.02                    # Fixed to 20 nm, can be None for mixed resolution training
sample_shape = (1, 256, 256)  # Bigger tiles since we have more resolution

stack_list = []

# Load metadata from JSON files
for ds in datasets:
    try:
        stack_path = os.path.join(OUT_FOLDER, f"{ds}_sharpest.tif")
        dendrite_path = os.path.join(OUT_FOLDER, f"{ds}_dendrite.tif")
        spine_path = os.path.join(OUT_FOLDER, f"{ds}_spines.tif")
        metadata_path = os.path.join(OUT_FOLDER, f"{ds}_metadata.json")

        with open(metadata_path) as jf:
            meta = json.load(jf)

        s = Stack(
            img=stack_path,
            d_mask=dendrite_path,
            s_masks=spine_path,
            meta=meta
        )
        stack_list.append(s)
    except Exception as e:
        print(f"Skipping {ds} due to error: {e}")
        continue

# Floodfill all the spines' masks and write back to disk
floodfill_stacks(fn=stack_list)

# Create the tiled samples from raw images and (for spines, floodfilled) mas
dg_training = TiledDataGenerator(fn=stack_list,
                                 batch_size=batch_size,
                                 target_resolution=res,
                                 size=sample_shape)

dg_validation = dg_training  # Temporarily reuse training data for now

## Compile model
os.environ["SM_FRAMEWORK"] = "tf.keras"

from deepd3.model import DeepD3_Model
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

### ROCm dependencies incantations ###
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
# Force preload
# import ctypes
# ctypes.CDLL("/opt/rocm/lib/librccl.so.1")
######################################
sm.set_framework("tf.keras")


# Create a naive DeepD3 model with a given base filter count
model = DeepD3_Model(filters=8)

# Set appropriate training settings
model.compile(optimizer=Adam(learning_rate=0.0005),  # Optimizer, good default setting, can be tuned
              # Dice loss for dendrite, MSE for spines
              loss=[sm.losses.dice_loss, "mse"],
              metrics=['acc', sm.metrics.iou_score])  # Metrics for monitoring progress

model.summary()

## Training model
EPOCHS = 30

# Save best model automatically during training
mc = ModelCheckpoint("DeepD3_test_model.h5",
                     save_best_only=True)

# Save metrics
csv = CSVLogger("DeepD3_test_model.csv")

# Adjust learning rate during training to allow for better convergence
lrs = LearningRateScheduler(schedule)

# Actually train the network
h = model.fit(dg_training,
              batch_size=dg_training.batch_size,
              epochs=EPOCHS,
              validation_data=dg_validation,
              callbacks=[mc, csv, lrs],
              )
