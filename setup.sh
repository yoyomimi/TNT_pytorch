python setup.py build_ext --inplace
export PATH=/usr/local/cuda-9.2/bin::$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH && python try.py  