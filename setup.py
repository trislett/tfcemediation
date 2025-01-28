import os
import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

PACKAGE_NAME = "tfce_mediation_slim"

BUILD_REQUIRES = ["numpy", "scipy", "matplotlib", "nibabel", "cython", "scikit-learn", "scikit-image", "joblib", "pandas", "tqdm"]

CLASSIFIERS = ["Development Status :: 4 - Beta",
  "Environment :: Console",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  "Operating System :: OS Independent",
  "Programming Language :: Python",
  "Topic :: Scientific/Engineering :: Medical Science Apps."]

# Define the extension modules
extensions = [
    # First extension: adjacency
    Extension(
        "tfcemediation.adjacency",  # Name of the resulting module
        ["tfcemediation/adjacency.pyx"],  # Source file
        language="c++",  # Use C++ for compilation
        include_dirs=[
            numpy.get_include(),  # NumPy headers
            "lib"  # Path to the 'lib/geodesic' folder
        ],
        extra_compile_args=["-std=c++11"],  # Enable C++11 standard
        libraries=[],  # Specify additional libraries if needed
        library_dirs=['lib/'],  # Specify library directories if needed
    ),
    # Second extension: tfce
    Extension(
        "tfcemediation.tfce",  # Name of the resulting module
        ["tfcemediation/tfce.pyx"],  # Source file
        language="c++",  # Use C++ for compilation
        include_dirs=[
            numpy.get_include(),  # NumPy headers
            "lib"  # Path to the 'lib/geodesic' folder
        ],
        extra_compile_args=["-std=c++11"],  # Enable C++11 standard
        libraries=[],  # Specify additional libraries if needed
        library_dirs=[],  # Specify library directories if needed
    ),
    Extension(
        "tfcemediation.cynumstats",  # Module name
        sources=["tfcemediation/cynumstats.pyx"],  # Cython source
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        language="c"
    )
]

exec(open('tfcemediation/version.py').read())
setup(name = PACKAGE_NAME, version = __version__,
  maintainer = "Tristram Lett",
  maintainer_email = "tristram.lett@charite.de",
  description = "TFCE_mediation",
  long_description = "Fast regression and mediation analysis of vertex or voxel MRI data with TFCE",
  url = "https://github.com/trislett/TFCE_mediation",
  download_url = "",
  platforms=["Linux", "Solaris", "Mac OS-X", "Unix"],
  license = "GNU General Public License v3 or later (GPLv3+)",
  zip_safe=False,
  install_requires=BUILD_REQUIRES,
  packages=find_packages(),
  ext_modules=cythonize(extensions, language_level="3"),
  python_requires=">=3.7",
)
