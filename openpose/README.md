# How to run:

$ conda activate py39

$ cd openpose

$ mkdir build && cd build

$ cmake -DBUILD_PYTHON=ON ..

$ make -j$(nproc)

$ python mcr.py examples/media/