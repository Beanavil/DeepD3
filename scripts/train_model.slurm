#!/bin/bash
#SBATCH --job-name=deepd3_train_model  # Job name
#SBATCH --output=%x_%j.out             # Output file (using %x=jobname and %j=jobid)
#SBATCH --error=%x_%j.err              # Error file
#SBATCH --nodes=1                      # Number of nodes to reserve
#SBATCH --gres=gpu:1                   # Generic resources requested (1 GPU in our case)
#SBATCH --mem=0                        # Use all memory on node
#SBATCH --time=0-06:00:00              # Max runtime (D-HH:MM:SS)
#SBATCH --account=<username>           # Account to charge
#SBATCH --no-requeue                   # Do not requeue on failure

# Load modules
module purge
module load python/3.12-conda

# Activate virtual environment
conda activate tensorflow2.18-py3.12

# Move to the root directory of DeepD3. Default is ~/DeepD3. If otherwise, queue the job with
# sbatch --export=DEEPD3_PATH=/path/to/deepd3 ... to set DEEPD3_PATH to the correct path to DeepD3
DEEPD3_PATH="${DEEPD3_PATH:~/DeepD3}"
echo "Using DeepD3 at $DEEPD3_PATH"
cd $DEEPD3_PATH

# Set environment variables
GIT_BRANCH="${GIT_BRANCH:main}"
PYTHONPATH=$DEEPD3_PATH:$DEEPD3_PATH/deepd3:$PYTHONPATH
DATA_PATH="${DATA_PATH:~/data}"
PREPROC_OUT_FOLDER="${PREPROC_OUT_FOLDER:processed}"
OUT_MODELS_FOLDER="${OUT_MODELS_FOLDER:models}"
EXTRA_ARGS="${EXTRA_ARGS:--floodfill --batch-size 32 --filters 8 --learning-rate 0.0005}"

# Checkout to appropriate branch
git checkout $GIT_BRANCH

# Install python dependencies
pip install -r tests/requirements.txt

# Train model
srun python tests/train_model.py --path $DATA_PATH --verbose --preproc-out-folder $PREPROC_OUT_FOLDER --out-models-folder $OUT_MODELS_FOLDER $EXTRA_ARGS
