import os, glob, json
from deepd3.model import DeepD3_Model
from deepd3.training.tile import TiledDataGenerator, Stack
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

# Force TF to grow GPU memory instead of grabbing it all
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Pick just a few datasets
data_root = r"D:\BIMAP-Rumessa\training_validation\turtle\data"
stack_files = glob.glob(os.path.join(data_root, "*_stack.tif"))[:3]  # only 3 for test

stack_list = []
for f in stack_files:
    base = f.replace("_stack.tif", "")
    sp = base + "_spines.tif"
    dp = base + "_dendrite.tif"
    mp = base + "_meta.json"

    with open(mp) as jf:
        meta = json.load(jf)

    stack_list.append(Stack(img=f, s_masks=sp, d_mask=dp, meta=meta))

# Tiny generators
dg_train = TiledDataGenerator(fn=stack_list, batch_size=1, target_resolution=0.02, size=(1,256,256))
dg_val = TiledDataGenerator(fn=stack_list, batch_size=1, target_resolution=0.02, size=(1,256,256))

# Model
model = DeepD3_Model(filters=4)  # very light test model
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=[sm.losses.dice_loss, "mse"],
    metrics=["acc", sm.metrics.iou_score],
)

print(model.summary())

# Train 1 epoch
history = model.fit(
    dg_train,
    validation_data=dg_val,
    epochs=1,
    steps_per_epoch=5,         # only 5 batches
    validation_steps=2
)

print("✅ Quick train finished successfully!")
