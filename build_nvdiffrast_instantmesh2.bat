@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "PIP_CACHE_DIR=D:\pip_cache"
set "TEMP=D:\temp"
set "TMP=D:\temp"
conda run -n instantmesh2 pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/
