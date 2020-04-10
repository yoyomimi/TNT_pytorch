python setup.py build_ext --inplace
export PATH=/usr/local/cuda-9.0/bin::$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH && python try.py  