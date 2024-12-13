# Instructions for building modules

## Table of Contents
[Build Modules](#BM)
   - [GSPLAT](#GS)


## Build gsplat module <a name="GS"></a>
In order to properly install gsplat as a package in env, the following steps are required:
    - Check the version of gcc and g++ compilers against the installed CUDA version.
    - CUDA 11.8 needs gcc and g++ 11.x version
    - From main directory run bash ./scripts/create_env_gs.sh
    - This will automatically create venv and install gspalt
    - If you use tox to start the gaussin splatting training, it will install additional dependencies by itself, and if not, then you need to manually install the req file intended for GS, which is in the requirements folder.