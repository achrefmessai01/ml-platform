#!/bin/bash

. venv/bin/activate;

python3 scripts/train.py --config training_configs/combined_dataset.yaml;
python3 scripts/train.py --config training_configs/combined_augmented_dataset.yaml;
python3 scripts/train.py --config training_configs/combined_augmented_dataset_custom.yaml;