# Tests

## Data preprocessing

For running the raw data preprocessing, run the [process_raw_data.py](process_raw_data.py) python script. It expects a path to the folder containing the raw data. If no provided, `./images` is used by default.

## Model training

For testing the model training, run the [train_model.py](train_model.py) python script. It expects a path to the folder containing the raw data, and assumes that the preprocessing has been done (and thus that there is subfolder `processed` for the TIF files to be used for training). If no provided, `./images` is used by default.
