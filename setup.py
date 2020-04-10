#!/usr/bin/env python
import os
import subprocess
import time
from setuptools import Extension, dist, find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources):

    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
    else:
        raise EnvironmentError('CUDA is required to compile!')
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


if __name__ == '__main__':
    setup(
        name = "Track_PyTorch",
        cmdclass = {'build_ext': BuildExtension},
        ext_modules = [
            make_cuda_ext(
                name='roi_align_cuda',
                module='extensions.roi_align',
                sources=[
                    'src/roi_align_cuda.cpp',
                    'src/roi_align_kernel.cu',
                    'src/roi_align_kernel_v2.cu',
                ]),
        ]
    )
