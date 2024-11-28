# Installation
## Create and activate conda environment
```
conda create -n G3Flow python=3.9
conda activate G3Flow
```

# Install Boost
```
# move to project root first
mkdir boost
cd boost
wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2
tar --bzip2 -xf ./boost_1_86_0.tar.bz2
cd boost_1_86_0
./bootstrap.sh --prefix=${PROJECT_ROOT}/boost
./b2 install
```

## Install FoundationPose
Move to FoundationPose directory:`cd tools/FoundationPose`
### Install package
```
# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0

# Install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
```

### Build extensions
Modify mycuda setup.py: `${PROJECT_ROOT}/tools/FoundationPose/bundlesdf/mycuda/setup.py`
```
# line 36, modify the ${conda_path}, as example:
"${conda_path}/envs/G3Flow/include/eigen3" # TODO
->
"/home/tianxingchen/anaconda3/envs/G3Flow/include/eigen3" # TODO
```

Modify mycpp CMakeLists.txt: `${PROJECT_ROOT}/tools/FoundationPose/mycpp/CMakeLists.txt`
```
# line 8, modify the ${PROJECT_ROOT}, as example:
"${PROJECT_ROOT}/boost/include"
->
"/home/tianxingchen/G3Flow/boost/include"
```

Build extensions:
```
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:${conda_path}/envs/G3Flow/include/eigen3"
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:${conda_path}/envs/G3Flow/include/eigen3/Eigen"
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

## Install PyTorch3D
```
cd tools/pytorch3d_simplified
pip install -e .
```

## Install RoboTwin
```
pip install sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic
```

### REMOVE !!!!!!!!!
#### Remove `convex=True`
You can use `pip show mplib` to find where the `mplib` installed.
```
# mplib.planner (mplib/planner.py) line 71
# remove `convex=True`

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
=> 
self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            # convex=True,
            verbose=False,
        )
```

#### Remove `or collide`
```
# mplib.planner (mplib/planner.py) line 848
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}

## Install Diffusion Policy
```
cd G3FlowDP/dp
pip install -e .
```

# Install DP3

## Weights & Assets Downloading
Download weights for founcation models:
```
source tools/weights_for_g3flow/download_assets.sh
```

## Install Grounded-SAM
```
cd tools/Grounded-Segment-Anything

pip install dinov2
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO

git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
```

