# Installation
## Create and activate conda environment
```
conda create -n G3Flow python=3.9
conda activate G3Flow
```

## Install FoundationPose
```
# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0

# Install dependencies
cd tools/FoundationPose
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
```

### Build extensions
```
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
```

## Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

## Build extensions
```
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:${conda_path}/G3Flow/include/eigen3"
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

## Install RoboTwin
See [https://github.com/TianxingChen/RoboTwin_private/blob/main/INSTALLATION.md](https://github.com/TianxingChen/RoboTwin_private/blob/main/INSTALLATION.md), skip the `Assert download` part.

## Install Diffusion Policy
```
cd G3FlowDP/dp
pip install -e .
```

## Weights Downloading
Download weights for founcation models:
```
python tools/weights_for_g3flow/download_weights.py
cd ../../RoboTwin_Benchmark
mv ../tools/weights_for_g3flow/robotwin_models.zip ./
unzip robotwin_models.zip
mv robotwin_models/* ./
rm robotwin_models.zip
```

## Install Grounded-SAM
```
pip install dinov2

python -m pip install -e segment_anything

pip install --no-build-isolation -e GroundingDINO

git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
```

