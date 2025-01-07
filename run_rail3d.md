pyenv activate Pointcept-env 

# In order to build the cpp_wrappers for KPConvX
pip install numpy==1.23.5 

# Other dependencies
pip install easydict

# Preprocess data
python pointcept/datasets/preprocessing/rail3d/preprocess_rail3d.py --dataset_root data/Rail3D --output_root data/Rail3D-Preprocessed --num-workers

# Run training sript for PTv3
sh scripts/train.sh -p python3.10 -g 1 -d rail3d -c semseg-pt-v3 -n rail3d-ptv3-base

# Run training sript for KPConvX
sh scripts/train.sh -p python3.10 -g 1 -d rail3d -c semseg-kpconvx-base -n rail3d-kpconvx-base

