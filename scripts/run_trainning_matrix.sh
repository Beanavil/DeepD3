#! /bin/bash

DEEPD3_SRC="$1"
DATA_SRC="$2"
if [[ $# -ne 2 ]]; then
    echo "Error: invalid arguments (found $#)! Usage: $0 <deepd3_src> <data_src>"
    exit 1
fi
TRAIN_SCRIPT="$DEEPD3_SRC/scripts/train_model.py"
PYTHONPATH_="$DEEPD3_SRC/deepd3:$PYTHONPATH"
echo "Using paths:"
echo "  * DEEPD3_SRC = $DEEPD3_SRC"
echo "  * TRAIN_SCRIPT = $TRAIN_SCRIPT"
echo "  * DATA_SRC = $DATA_SRC"
echo "  * PYTHONPATH = $PYTHONPATH_"

# Fixed parameters
BATCH_SIZE=32
EPOCHS=30
SAMPLE_SHAPE="1,256,256"

# Tuned parameters
FILTERS=(8 16 32 64)
DATA_FOLDERS=("training_validation" "training_validation_base")
VANILLA_OPTS=("--vanilla" "")

for v in "${VANILLA_OPTS[@]}"; do
    for folder in "${DATA_FOLDERS[@]}"; do
        if [[ "$folder" == *"base" ]]; then
            MODEL_FOLDER_NAME="models_base"
            NAME_ENDING="base"
        else
            MODEL_FOLDER_NAME="models"
            NAME_ENDING=""
        fi
        for f in "${FILTERS[@]}"; do
            MODEL_NAME="f${f}_bs${BATCH_SIZE}_e${EPOCHS}_s${SAMPLE_SHAPE//,/-}_dice_mse_${NAME_ENDING}"
            MODEL_NAME=${MODEL_NAME%_}

            echo "Running \"PYTHONPATH=$PYTHONPATH_ python $TRAIN_SCRIPT
                --path $DATA_SRC
                --verbose
                --preproc-out-folder $folder
                --filters $f
                --batch-size $BATCH_SIZE
                --epochs $EPOCHS
                --sample-shape $SAMPLE_SHAPE
                --model-name "$MODEL_NAME"
                --out-models-folder "$MODEL_FOLDER_NAME"
                $v\""
            PYTHONPATH=$PYTHONPATH_ python $TRAIN_SCRIPT \
                --path $DATA_SRC \
                --verbose \
                --preproc-out-folder $folder \
                --filters $f \
                --batch-size $BATCH_SIZE \
                --epochs $EPOCHS \
                --sample-shape $SAMPLE_SHAPE \
                --model-name "$MODEL_NAME" \
                --out-models-folder "$MODEL_FOLDER_NAME" \
                $v
        done
    done
done
