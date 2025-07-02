import os
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "tfcemediation.adjacency",
        ["tfcemediation/adjacency.pyx"],
        language="c++",
        include_dirs=[
            numpy.get_include(),
            "libs"
        ],
        extra_compile_args=["-std=c++17"],
        libraries=[],
        library_dirs=['libs/'], 
    ),
    Extension(
        "tfcemediation.tfce",
        ["tfcemediation/tfce.pyx"],
        language="c++",
        include_dirs=[
            numpy.get_include(),
            "libs"
        ],
        extra_compile_args=["-std=c++17"],  
        libraries=[],
        library_dirs=[],
    ),
    Extension(
        "tfcemediation.cynumstats",
        sources=["tfcemediation/cynumstats.pyx"],
        include_dirs=[numpy.get_include()],
        language="c"
    )
]

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

setup(
    ext_modules=cythonize(extensions, language_level="3")
)