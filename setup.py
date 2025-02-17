import os
import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

PACKAGE_NAME = "tfcemediation"

BUILD_REQUIRES = ["numpy", "scipy", "matplotlib", "nibabel", "cython", "scikit-learn", "scikit-image", "joblib", "pandas", "tqdm", "statsmodels"]

CLASSIFIERS = ["Development Status :: 4 - Beta",
  "Environment :: Console",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  "Operating System :: OS Independent",
  "Programming Language :: Python",
  "Topic :: Scientific/Engineering :: Medical Science Apps."]

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

exec(open('tfcemediation/version.py').read())
setup(name = PACKAGE_NAME, version = __version__, include_package_data=True,
  maintainer = "Tristram Lett",
  maintainer_email = "tris.lett@gmail.com",
  description = "tfcemediation",
  long_description = "Fast regression and mediation analysis of vertex or voxel MRI data with TFCE",
  url = "https://github.com/trislett/tfce_mediation_slim",
  download_url = "",
  platforms=["Linux", "Solaris", "Mac OS-X", "Unix"],
  license = "GNU General Public License v3 or later (GPLv3+)",
  zip_safe=False,
  install_requires=BUILD_REQUIRES,
  packages=find_packages(),
  package_dir={'tfcemediation': 'tfcemediation'},
  package_data={'tfcemediation': ['static/*', 'libs/*', 'static/aseg-subcortical-Surf/*', 'static/JHU-ICBM-Surf/*', 'static/aseg-other-Surf/']},
  ext_modules=cythonize(extensions, language_level="3"),
  python_requires=">=3.7",
)
