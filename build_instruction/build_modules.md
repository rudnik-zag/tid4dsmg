# Instructions for building modules

## Table of Contents
[Build Modules](#BM)
   - [GSPLAT](#GS)
   - [COLMAP](#CM)
   - [Metric3D](#M3D)
   - [Marigold](#MG)


## Build gsplat module <a name="GS"></a>
In order to properly install gsplat as a package in env, the following steps are required:
    - Check the version of gcc and g++ compilers against the installed CUDA version.
    - CUDA 11.8 needs gcc and g++ 11.x version
    - From main directory run bash ./scripts/create_env_gs.sh
    - This will automatically create venv and install gspalt
    - If you use tox to start the gaussin splatting training, it will install additional dependencies by itself, and if not, then you need to manually install the req file intended for GS, which is in the requirements folder.


## Build COLMAP module <a name="CM"></a>
Run ./scripts/create_colmap_venv.sh

## Build Metric3d module <a name="M3D"></a>
In reguirements_v2.txt coment next module:
   - torch == 2.0.1
   - torchvision == 0.15.2
   - numpy #== 1.23.1
   - xformers == 0.0.21

Run ./scripts/create_metric3d_venv.sh

## Build Marigold module <a name="MG"></a>