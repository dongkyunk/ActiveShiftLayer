from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='opASL',
    ext_modules=[
        CppExtension('opASL', ['opASL.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })