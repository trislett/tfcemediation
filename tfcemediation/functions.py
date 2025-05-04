#!/usr/bin/env python

import os
import sys
import struct
import warnings
import pickle
import gzip
import shutil
import subprocess
import gc
import time
import datetime
import re

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv

from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed, wrap_non_picklable_objects, dump
from joblib import load as jload
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import t as tdist, f as fdist
from scipy.stats import norm, chi2
from scipy.special import erf
from tfcemediation.tfce import CreateAdjSet
from tfcemediation.adjacency import compute
from tfcemediation.cynumstats import cy_lin_lstsqr_mat, fast_se_of_slope
from patsy import dmatrix
from scipy.ndimage import label as scipy_label
from scipy.ndimage import generate_binary_structure, convolve
from skimage.measure import marching_cubes

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

# get static resources
scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
static_directory = os.path.join(scriptwd, "tfcemediation", "static")
static_files = os.listdir(static_directory)

pack_directory = os.path.join(scriptwd, "tfcemediation", "static", "aseg-subcortical-Surf")
aseg_subcortical_files = np.sort(os.listdir(pack_directory))
pack_directory = os.path.join(scriptwd, "tfcemediation", "static", "JHU-ICBM-Surf")
jhu_white_matter_files = np.sort(os.listdir(pack_directory))

def configure_threads(n_threads=1):
	"""
	Set the number of threads so it doesn't interfer with joblib higher-level parallelism. Call this at the very beginning 
	of your script before any imports. Unfortunately, there is no perfect solution due to Python's import system. This 
	function is very important in terms of processing time. It is best to n_threads=1, and then adjust the number of jobs 
	with n_jobs.
	
	Example:
	
	from tfcemediation.functions import configure_threads
	configure_threads(1)
	
	Parameters
	----------
	n_threads : int, default = None
		Sets external parallel processing to use single core. So, n_jobs has complete control on
		the number of threads being used. 
	Returns
	-------
		None
	"""
	import os
	os.environ["OMP_NUM_THREADS"] = str(n_threads)
	os.environ["MKL_NUM_THREADS"] = str(n_threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
	os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)



def generate_seeds(n_seeds, maxint = int(2**32 - 1)):
	"""
	Generates a list of random integer seeds.
	
	This function creates a list of 'n_seeds' random integers within the range [0, maxint],
	which can be used for initializing random number generators.
	
	Parameters
	----------
	n_seeds : int
		The number of random seeds to generate.
	maxint : int, optional
		The upper bound for the random integers (default is 2^32 - 1).
	
	Returns
	-------
	list
		A list of 'n_seeds' randomly generated integers.
	"""
	return([np.random.randint(0, maxint) for i in range(n_seeds)])


def dummy_code(variable, iscontinous = False, demean = True):
	"""
	Dummy codes a variable
	
	Parameters
	----------
	variable : array
		1D array variable of any type 
	Returns
	---------
	dummy_vars : array
		dummy coded array of shape [(# subjects), (unique variables - 1)]
	
	"""
	if iscontinous:
		dummy_vars = variable - np.mean(variable,0)
	else:
		unique_vars = np.unique(variable)
		dummy_vars = []
		for var in unique_vars:
			temp_var = np.zeros((len(variable)))
			temp_var[variable == var] = 1
			dummy_vars.append(temp_var)
		dummy_vars = np.array(dummy_vars)[1:] # remove the first column as reference variable
		dummy_vars = np.squeeze(dummy_vars).astype(int).T
		if demean:
			dummy_vars = dummy_vars - np.mean(dummy_vars,0)
	return(dummy_vars)


def stack_ones(arr):
	"""
	Add a column of ones to an array
	
	Parameters
	----------
	arr : array

	Returns
	---------
	arr : array
		array with a column of ones
	
	"""
	return(np.column_stack([np.ones(len(arr)),arr]))


def sanitize_columns(s):
	"""
	Sanitizes a string to create a valid and consistent column name.
	
	This function performs the following transformations on the input string:
	- Replaces occurrences of '[T.' with '_'
	- Removes all non-alphanumeric characters, excluding spaces and hyphens
	- Replaces spaces with underscores
	- Converts the string to lowercase
	- Strips leading and trailing underscores

	Parameters
	----------
	s : str
		The input string to sanitize.

	Returns
	---------
	str
		The sanitized string.
	"""
	s = s.replace('[T.', '_')
	s = re.sub(r'[^\w\s-]', '', s)
	s = s.replace(' ', '_')
	s = s.lower()
	s = s.strip('_')
	return(s)


def scale_arr(arr, centre = True, scale = True, div_sqrt_nvar = False, axis = 0):
	"""
	Helper function to center and scale data.

	Parameters:
	-----------
	arr : np.ndarray
		Data to be scaled
	centre : bool
		A boolean that specifies whether to center data.
		Default value is True.
	scale : bool
		A boolean that specifies whether to scale data.
		Default value is True.
	div_sqrt_numvar : bool
		A boolean that specifies whether to divide the views by the square root of the number of variables.
		Default value is True.
	axis : int
		An integer that specifies the axis to use when computing the mean and standard deviation.

	Returns:
	--------
	x : np.ndarray
		The scaled data.
	"""
	if arr.ndim == 1:
		arr = arr.reshape(-1,1)
	x = np.array(arr)
	x_mean = np.mean(x, axis = axis)
	x_std = np.std(x, axis = axis)
	if centre:
		x = x - x_mean
	if scale:
		x = np.divide(x, x_std)
	if div_sqrt_nvar:
		x = np.divide(x, np.sqrt(x.shape[1]))
	return(x)


def get_precompiled_freesurfer_adjacency(geodesic_distance = 3):
	"""
	Loads precomputed adjacency matrices from the midthickness FreeSurfer surface for left and right hemispheres.
	
	Parameters
	----------
	geodesic_distance : int, optional
		The geodesic distance of the adjacency (default is 3).
	
	Returns
	-------
	tuple of np.ndarray
		adjacency_lh : numpy array
			Adjacency matrix for the left hemisphere.
		adjacency_rh : numpy array
			Adjacency matrix for the right hemisphere.
	"""
	assert geodesic_distance in np.array([1,2,3]), "Error: the geodesic distance must be 1, 2, or 3 for precompiled matrices"
	adjacency_lh = np.load('%s/lh_adjacency_dist_%d.0_mm.npy' % (static_directory, int(geodesic_distance)), allow_pickle=True)
	adjacency_rh = np.load('%s/rh_adjacency_dist_%d.0_mm.npy' % (static_directory, int(geodesic_distance)), allow_pickle=True)
	return(adjacency_lh, adjacency_rh)


def convert_fslabel(name_fslabel):
	"""
	Reads and parses a FreeSurfer label file.
	
	Parameters
	----------
	name_fslabel : str
		Path to the FreeSurfer label file.
	
	Returns
	-------
	tuple of np.ndarray
		v_id : numpy array
			Vertex indices.
		v_ras : numpy array
			Vertex coordinates (RAS - Right, Anterior, Superior).
		v_value : numpy array
			Associated scalar values for each vertex.
	
	Raises
	------
	ValueError
		If the file header is incorrectly formatted.
	"""
	obj = open(name_fslabel)
	reader = obj.readline().strip().split()
	reader = np.array(obj.readline().strip().split())
	if reader.ndim == 1:
		num_vertex = reader[0].astype(int)
	else:
		print('Error reading header')
	v_id = np.zeros((num_vertex)).astype(int)
	v_ras = np.zeros((num_vertex,3)).astype(np.float32)
	v_value = np.zeros((num_vertex)).astype(np.float32)
	for i in range(num_vertex):
		reader = obj.readline().strip().split()
		v_id[i] = np.array(reader[0]).astype(int)
		v_ras[i] = np.array(reader[1:4]).astype(np.float32)
		v_value[i] = np.array(reader[4]).astype(np.float32)
	return (v_id, v_ras, v_value)


def create_vertex_adjacency_neighbors(vertices, faces):
	"""
	Creates a vertex adjacency list from a given set of vertices and faces.

	Parameters:
	-----------
	vertices : numpy.ndarray
		A 2D array of shape (N, 3) representing the vertex coordinates, where N is the number of vertices.
	faces : numpy.ndarray
		A 2D array of shape (M, 3) representing the face connectivity, where M is the number of faces.
		Each face is defined by three vertex indices.

	Returns:
	--------
	adjacency : list of numpy.ndarray
		A list of arrays where each array contains the indices of vertices adjacent to the corresponding vertex.
	"""
	num_vertices = vertices.shape[0]
	adjacency = [[] for _ in range(num_vertices)]
	edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
	edges = np.unique(np.sort(edges, axis=1), axis=0)  # Remove duplicates and sort
	for v0, v1 in edges:
		adjacency[v0].append(v1)
		adjacency[v1].append(v0)
	adjacency = [np.array(adj, dtype=int) for adj in adjacency]
	return(adjacency)


def create_vertex_adjacency_geodesic(vertices, faces, geodesic_distance=3.0):
	"""
	Creates a vertex adjacency list based on geodesic distance for a given mesh.
	The adjacency matrices for geodesic distances for {1.0,2.0,3.0} are already precompiled (get_precompiled_freesurfer_adjacency) 
	for the freesurfer fsaverage surfaces (midthickness: halfway between white and pial surfaces). Only use this function if you 
	want to create adjacency lists for a new surface or a geodesic distance other than those listed above.

	The function computes adjacency relationships between vertices by determining which vertices
	are within a specified geodesic distance on the mesh surface. It takes approximately an hour to finish.
	Unfortunately, the inner function (tfcemediation.adjacency.compute) is not easily parallelizable.

	Parameters:
	-----------
	vertices : numpy.ndarray
		A 2D array of shape (N, 3) representing the vertex coordinates, where N is the number of vertices.
		Each row corresponds to the (x, y, z) coordinates of a vertex.
	faces : numpy.ndarray
		A 2D array of shape (M, 3) representing the face connectivity, where M is the number of faces.
		Each row contains three vertex indices that define a triangular face.
	geodesic_distance : float, optional
		The maximum geodesic distance (shortest path along the mesh surface) to consider two vertices as adjacent.
		Default is 3.0.

	Returns:
	--------
	adjacency : list of numpy.ndarray
		A list of arrays where each array contains the indices of vertices adjacent to the corresponding vertex.
		The i-th element of the list corresponds to the adjacency list for the i-th vertex.
	"""
	vertices = vertices.astype(np.float32, order="C")
	faces = faces.astype(np.int32, order="C")
	dist_arr = np.array([geodesic_distance], dtype=np.float32)
	adjacency = compute(vertices, faces, dist_arr)
	return(adjacency[0])


def register_freesurfer_surfaces(subjects, surface, subjects_dir = None, num_cores=16, tempdir=None, approximate_fwhm = 3):
	"""
	Registers FreeSurfer surface data to the fsaverage template using spherical registration.

	Parameters:
	-----------
	subjects : list of str
		A list of subject IDs corresponding to directories within the FreeSurfer subjects directory.
	surface : str
		The surface metric to register. Must be either 'area' or 'thickness'.
	subjects_dir : str, optional
		Path to the FreeSurfer subjects directory. If None, defaults to the SUBJECTS_DIR environment variable.
	num_cores : int, optional
		Number of CPU cores to use for parallel processing. Default is 16.
	tempdir : str, optional
		Path to a temporary directory for storing intermediate registration files. If None, a new directory is created.
	approximate_fwhm : int or None, optional
		Approximate full-width at half-maximum (FWHM) in mm for smoothing the registered surface data. 
		If None, no smoothing is applied. Default is 3mm.

	Returns:
	--------
	tempdir : str
		Path to the directory containing the registered and optionally smoothed surface data.

	Raises:
	-------
	EnvironmentError
		If required FreeSurfer environment variables (FREESURFER_HOME or SUBJECTS_DIR) are not set.
	AssertionError
		If the subjects directory does not exist or if an invalid surface type is provided.

	Notes:
	------
	- The function ensures the FreeSurfer subjects directory is correctly set up.
	- Surface data is registered to fsaverage using 'mri_surf2surf' with spherical registration.
	- If the fsaverage subject is not found in the subjects directory, a symbolic link is created.
	- Parallel processing is used to accelerate registration across subjects.
	- Optionally applies smoothing using an approximate FWHM value.
	"""

	# Check environment variables
	fs_home = os.environ.get('FREESURFER_HOME')
	if not fs_home:
		raise EnvironmentError("FREESURFER_HOME environment variable not set")
	if subjects_dir is None:
		subjects_dir = os.environ.get('SUBJECTS_DIR')
		if not subjects_dir:
			raise EnvironmentError("SUBJECTS_DIR environment variable not set")
		print("Using subjects_dir : %s" % subjects_dir)
	assert os.path.exists(subjects_dir), "Error: subjects directory does not exist [%s]" % subjects_dir
	assert surface in ['area','thickness'], "Error: surface [%s] must be either {area, thickness}" % surface

	# set $SUBJECTS_DIR and create fsaverage symbolic link if necessary
	os.environ["SUBJECTS_DIR"] = subjects_dir
	if not os.path.exists(subjects_dir + "/fsaverage"):
		fsaverage_dir = fs_home + '/subjects/fsaverage'
		fsaverage_link = subjects_dir + "/fsaverage"
		os.symlink(fsaverage_dir, fsaverage_link)

	# Create temporary directory
	if tempdir is None:
		timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
		tempdir = f"temp_{surface}_{timestamp}"
	os.makedirs(tempdir, exist_ok=True)

	# Generate registration commands
	registration_commands = []
	for hemi in ['lh', 'rh']:
		for subject in subjects:
			src_surf = os.path.join(subjects_dir, subject, 'surf', f"{hemi}.{surface}")
			tval = os.path.join(tempdir, f"{hemi}.{subject}.{surface}.00.mgh")
			cmd = [
				os.path.join(fs_home, 'bin', 'mri_surf2surf'),
				'--srcsubject', subject,
				'--srchemi', hemi,
				'--srcsurfreg', 'sphere.reg',
				'--trgsubject', 'fsaverage',
				'--trghemi', hemi,
				'--trgsurfreg', 'sphere.reg',
				'--tval', tval,
				'--sval', src_surf,
				'--jac',
				'--sfmt', 'curv',
				'--noreshape',
				'--cortex'
			]
			registration_commands.append(cmd)

	def run_command(cmd):
		subprocess.run(cmd, check=True)
	_ = Parallel(n_jobs=num_cores)(delayed(run_command)(cmd) for cmd in registration_commands)

	if approximate_fwhm is not None:
		approximate_fwhm = int(approximate_fwhm)
		# 3mm approximate FWHM smoothing
		smoothing_commands = []
		for hemi in ['lh', 'rh']:
			pattern = os.path.join(tempdir, f"{hemi}.*.{surface}.00.mgh")
			for file_path in glob(pattern):
				output_path = file_path.replace('.00.mgh', '.0%dB.mgh' % approximate_fwhm)
				cmd = [
					os.path.join(fs_home, 'bin', 'mri_surf2surf'),
					'--hemi', hemi,
					'--s', 'fsaverage',
					'--sval', file_path,
					'--tval', output_path,
					'--fwhm-trg', '%d' % approximate_fwhm,
					'--noreshape',
					'--cortex'
				]
				smoothing_commands.append(cmd)
		_ = Parallel(n_jobs=num_cores)(delayed(run_command)(cmd) for cmd in smoothing_commands)
	return(tempdir)


def load_surface_geometry(path_to_surface):
	"""
	Load surface geometry (vertices and faces) from various neuroimaging file formats.
	
	Parameters
	----------
	path_to_surface : str
		Path to surface file. Supported formats:
		- FreeSurfer (.srf)
		- GIFTI (.surf.gii)
		- CIFTI (.d*.nii)
		- VTK (.vtk)
	
	Returns
	-------
	v : np.ndarray
		Vertex coordinates (N, 3)
	f : np.ndarray
		Face connectivity indices (M, 3)
	
	Raises
	------
	ValueError
		If file format is not supported or surface data cannot be extracted
	"""
	if not os.path.exists(path_to_surface):
		raise FileNotFoundError("Surface file not found: [%s]" % path_to_surface)
	ext = os.path.splitext(path_to_surface)[1].lower()

	# extra checks for ext
	if path_to_surface.endswith('.dtseries.nii') or path_to_surface.endswith('.dtseries.nii.gz'):
		ext = '.dtseries.nii'
	if path_to_surface.endswith('.dscalar.nii') or path_to_surface.endswith('.dscalar.nii.gz'):
		ext = '.dscalar.nii'
	if path_to_surface.endswith('.dlabel.nii') or path_to_surface.endswith('.dlabel.nii.gz'):
		ext = '.dlabel.nii'
	if ext == '.srf':
		v, f = nib.freesurfer.io.read_geometry(path_to_surface)
		return(v, f)
	elif ext == '.gii':
		gii = nib.load(path_to_surface)
		v, f = None, None
		for da in gii.darrays:
			if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
				v = da.data
			elif da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
				f = da.data
		if v is None or f is None:
			raise ValueError("GIFTI file missing vertex or face data")
		return(v, f)
	elif ext in ('.dtseries.nii', '.dscalar.nii', '.dlabel.nii'):
		if path_to_surface.endswith('.gz'):
			outfile = 'tempfile.nii'
			with gzip.open(path_to_surface, 'rb') as file_in:
				with open(outfile, 'wb') as file_out:
					shutil.copyfileobj(file_in, file_out)
			cifti = nib.load(outfile)
			os.remove(outfile)
		else:
			cifti = nib.load(path_to_surface)
		for brain_model in cifti.header.get_axis(1).iter_structures():
			print(brain_model)
			if brain_model[0] == 'CIFTI_MODEL_TYPE_SURFACE':
				v, f = brain_model[1].surface.surface_vertices, brain_model[1].surface.surface_faces
				return(v, f)
		raise ValueError("No surface data found in CIFTI file")
	elif ext == '.vtk':
		mesh = pv.read(path_to_surface)
		v = mesh.points
		f = mesh.faces.reshape(-1, 4)[:, 1:4]
		return(v, f)
	else:
		print("Warning: attempting to load [%s] as freesurfer surface mesh" % path_to_surface)
		try:
			v, f = nib.freesurfer.io.read_geometry(path_to_surface)
			return(v, f)
		except:
			raise ValueError("Unsupported surface file [%s] " % path_to_surface)


class CorticalSurfaceImage:
	"""
	Represents a cortical surface image,masking them by non-zero data, and calculatign their adjacency for statistical analyses from FreeSurfer files. 
	It is possible to save and load the data classes as *.pkl objects. The masking and conversion to np.float32 substantially reduces the
	file size and RAM requirements in large datasets.
	"""
	def __init__(self, *, surfaces_lh_path = None, surfaces_rh_path = None, adjacency_lh_path = None, adjacency_rh_path = None):
		"""
		Initializes the CorticalSurfaceImage instance by loading and processing cortical surface data.
		
		Parameters
		----------
		surfaces_lh_path : str, optional
			Path to the left hemisphere surface file.
		surfaces_rh_path : str, optional
			Path to the right hemisphere surface file.
		adjacency_lh_path : str, optional
			If None, then the precomputed left hemisphere adjacency matrix at 3mm geodesic distance.
		adjacency_rh_path : str, optional
			If None, then the precomputed right hemisphere adjacency matrix at 3mm geodesic distance.
		"""

		if surfaces_lh_path is not None:
			if isinstance(surfaces_lh_path, str):
				surface_lh = nib.freesurfer.mghformat.load(surfaces_lh_path)
				data_lh = surface_lh.get_fdata()[:,0,0,:]
				n_vertices_lh, n_subjects = data_lh.shape
				affine_lh = surface_lh.affine
				header_lh = surface_lh.header
				mask_index_lh = convert_fslabel('%s/lh.cortex.label' % static_directory)[0]
				
				bin_mask_lh = np.zeros_like(data_lh.mean(1))
				bin_mask_lh[mask_index_lh]=1
				bin_mask_lh = bin_mask_lh.astype(int)
				data_lh = data_lh[bin_mask_lh == 1].astype(np.float32, order = "C")
			if isinstance(surfaces_lh_path, list):
				data_lh, n_vertices_lh, affine_lh, header_lh, bin_mask_lh = self._process_freesurfer_surfaces_path_list(surfaces_path = surfaces_lh_path,
																																					hemisphere = 'lh')

		if surfaces_rh_path is not None:
			if isinstance(surfaces_rh_path, str):
				surface_rh = nib.freesurfer.mghformat.load(surfaces_rh_path)
				data_rh = surface_rh.get_fdata()[:,0,0,:]
				n_vertices_rh, _ = data_rh.shape
				affine_rh = surface_rh.affine
				header_rh = surface_rh.header
				mask_index_rh = convert_fslabel('%s/rh.cortex.label' % static_directory)[0]
				
				bin_mask_rh = np.zeros_like(data_rh.mean(1))
				bin_mask_rh[mask_index_rh]=1
				bin_mask_rh = bin_mask_rh.astype(int)
				data_rh = data_rh[bin_mask_rh == 1].astype(np.float32, order = "C")
			if isinstance(surfaces_rh_path, list):
				data_rh, n_vertices_rh, affine_rh, header_rh, bin_mask_rh = self._process_freesurfer_surfaces_path_list(surfaces_path = surfaces_rh_path,
																																					hemisphere = 'rh')

		if adjacency_lh_path is None and adjacency_rh_path is None:
			adjacency = get_precompiled_freesurfer_adjacency(geodesic_distance = 3)

		if surfaces_lh_path is None and surfaces_rh_path is None:
			print("Surface files (scalars) not provided. Use build_freesurfer_surfaces_parallel if the files do not exist")
		else:
			self.image_data_ = np.concatenate([data_lh, data_rh]).astype(np.float32, order = "C")
			self.affine_ = [affine_lh, affine_rh]
			self.header_ = [header_lh, header_rh]
			self.n_vertices_ = [n_vertices_lh, n_vertices_rh]
			self.mask_data_ = [bin_mask_lh, bin_mask_rh]
		
		self.adjacency_ = adjacency
		self.hemispheres_ = ['left-hemisphere', 'right-hemisphere']

	def _process_freesurfer_surfaces_path_list(self, surfaces_path, hemisphere):
		"""
		Loads and processes FreeSurfer surface data for a given hemisphere.

		Parameters:
		-----------
		surfaces_path : list of str
			A list of file paths to the FreeSurfer surface metric files (.mgh format).
		hemisphere : str
			The hemisphere being processed, either 'lh' (left hemisphere) or 'rh' (right hemisphere).

		Returns:
		--------
		data : numpy.ndarray
			A 2D array of shape (N, M) containing surface metric values, where N is the number of vertices
			in the cortical mask and M is the number of surfaces processed.
		n_vertices : int
			Total number of vertices in the original surface before masking.
		affine : numpy.ndarray
			The affine transformation matrix from the FreeSurfer MGH file.
		header : nibabel.freesurfer.mghformat.MGHHeader
			The header metadata from the MGH file.
		bin_mask : numpy.ndarray
			A 1D binary mask array indicating which vertices belong to the cortical mask.

		Notes:
		------
		- The function loads the first surface file to determine the number of vertices and retrieve the affine and header.
		- A binary mask is created based on the cortex label file for the given hemisphere.
		- The mask is applied to extract only valid cortical vertex data.
		- Data from all surface files is loaded and stored in a single array.
		"""
		assert hemisphere in ['lh', 'rh'], "Error: hemisphere must be either {lh, rh}"
		surface = nib.freesurfer.mghformat.load(surfaces_path[0])
		data = surface.get_fdata()[:,0,0]
		n_vertices = data.shape[0]
		affine = surface.affine
		header = surface.header
		mask_index = convert_fslabel('%s/%s.cortex.label' % (static_directory, hemisphere))[0]

		bin_mask = np.zeros_like(data)
		bin_mask[mask_index]=1
		bin_mask = bin_mask.astype(int)
		data = data[bin_mask == 1].astype(np.float32, order = "C")
		all_data = np.zeros((bin_mask.sum(),len(surfaces_path)))
		for s, surface_path in enumerate(surfaces_path):
			temp = nib.freesurfer.mghformat.load(surface_path).get_fdata()[:,0,0]
			all_data[:,s] = temp[bin_mask == 1].astype(np.float32, order = "C")
		data = all_data.astype(np.float32, order = "C")
		return(data, n_vertices, affine, header, bin_mask)

	def build_freesurfer_surfaces_parallel(self, subjects, surface, subjects_dir, num_cores=12):
		"""
		Processes and builds FreeSurfer surface data in parallel for both hemispheres.

		Parameters:
		-----------
		subjects : list of str
			A list of subject IDs corresponding to directories within the FreeSurfer subjects directory.
		surface : str
			The surface metric to process. Must be either 'area' or 'thickness'.
		subjects_dir : str
			Path to the FreeSurfer subjects directory.
		num_cores : int, optional
			Number of CPU cores to use for parallel processing. Default is 12.

		Returns:
		--------
		None

		Attributes Set:
		---------------
		image_data_ : numpy.ndarray
			Concatenated surface data for both hemispheres.
		affine_ : list of numpy.ndarray
			List containing affine transformation matrices for left and right hemispheres.
		header_ : list of nibabel.freesurfer.mghformat.MGHHeader
			List of headers for left and right hemispheres.
		n_vertices_ : list of int
			Number of vertices in the original surface before masking for each hemisphere.
		mask_data_ : list of numpy.ndarray
			Binary masks indicating valid cortical vertices for each hemisphere.

		Notes:
		------
		- Calls 'register_freesurfer_surfaces' to generate registered surface files.
		- Loads processed surface files for left and right hemispheres.
		- Uses '_process_freesurfer_surfaces_path_list' to extract relevant data.
		- Stores processed data in instance attributes.
		
		"""

		if subjects_dir is None:
			subjects_dir = os.environ.get('SUBJECTS_DIR')
			if not subjects_dir:
				raise EnvironmentError("SUBJECTS_DIR environment variable not set")
			print("Using subjects_dir : %s" % subjects_dir)
		assert os.path.exists(subjects_dir), "Error: subjects directory does not exist [%s]" % subjects_dir
		assert surface in ['area','thickness'], "Error: surface [%s] must be either {area, thickness}" % surface
		os.environ["SUBJECTS_DIR"] = subjects_dir
		
		output_directory = register_freesurfer_surfaces(subjects = subjects, surface = surface, subjects_dir = None, num_cores = num_cores)
		surfaces_lh_path = list(np.sort(glob(output_directory + "/lh*" + surface + ".03B.mgh")))
		data_lh, n_vertices_lh, affine_lh, header_lh, bin_mask_lh = self._process_freesurfer_surfaces_path_list(surfaces_path = surfaces_lh_path,
																																			hemisphere = 'lh')
		surfaces_rh_path = list(np.sort(glob(output_directory + "/rh*" + surface + ".03B.mgh")))
		data_rh, n_vertices_rh, affine_rh, header_rh, bin_mask_rh = self._process_freesurfer_surfaces_path_list(surfaces_path = surfaces_rh_path,
																																			hemisphere = 'rh')
		self.image_data_ = np.concatenate([data_lh, data_rh]).astype(np.float32, order = "C")
		self.affine_ = [affine_lh, affine_rh]
		self.header_ = [header_lh, header_rh]
		self.n_vertices_ = [n_vertices_lh, n_vertices_rh]
		self.mask_data_ = [bin_mask_lh, bin_mask_rh]

	def save(self, filename):
		"""
		Saves a VoxelImage instance as a pickle file.
		
		Parameters
		----------
			filename : str
				The name of the pickle file to be saved.
				
		Raises
		----------
			AssertionError: If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filename):
		"""
		Loads a VoxelImage instance from a pickle file.
		
		Parameters
		----------
			filename (str): The name of the pickle file to load.
		Returns
		-------
			VoxelImage : object
				The loaded instance.
		Raises
		-------
			AssertionError: If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'rb') as f:
			return pickle.load(f)


class MultiSurfaceImage:
	"""
	MultiSurfaceImage is a ImageObjectMRI in that stores the scalar values and adjacency. Multiple surfaces can be included as once. 
	It is possible to save and load the data classes as *.pkl objects. The masking and conversion to np.float32 substantially reduces the
	file size and RAM requirements in large datasets.
	"""

	def __init__(self, *, generic_affine=np.array([[-1, 0, 0, 90], [0, 0, 1, -126], [0, -1, 0, 72], [0, 0, 0, 1]])):
		"""
		Initializes the MultiSurfaceImage instance.

		Parameters
		----------
		generic_affine : np.ndarray, optional
			The affine transformation matrix roughly in FreeSurfer space when an affine isn't provided.
			Default is a standard FreeSurfer affine matrix.
		"""
		self.default_affine_ = generic_affine
		self.image_data_ = []
		self.affine_ = []
		self.header_ = []
		self.n_vertices_ = []
		self.mask_data_ = []
		self.adjacency_ = []
		self.surface_names_ = []
		self.n_subjects_ = None

	def _check_surface(self, n_subjects):
		"""
		Ensures that all scalar data have the same number of subjects. Maybe more checks later... 

		Parameters
		----------
		n_subjects : int
			The number of subjects in the current dataset.

		Raises
		------
		AssertionError
			If the number of subjects does not match the previously stored number of subjects.
		"""
		if self.n_subjects_ is not None:
			assert n_subjects == self.n_subjects_, "Error: all scalar data must have the same number of subjects"
		else:
			self.n_subjects_ = n_subjects

	def _check_adjacency(self, adjacency):
		"""
		Validates and loads the adjacency matrix.

		Parameters
		----------
		adjacency : str or np.ndarray
			Either a file path to a .npy file containing the adjacency matrix or a NumPy array.

		Returns
		-------
		np.ndarray
			The loaded or validated adjacency matrix.

		Raises
		------
		ValueError
			If the adjacency is neither a string ending with .npy nor a NumPy array.
		"""
		if isinstance(adjacency, str) and os.path.splitext(adjacency.lower())[1] == '.npy':
			return np.load(adjacency)
		elif isinstance(adjacency, np.ndarray):
			return adjacency
		else:
			raise ValueError("adjacency must be either a string ending with .npy or a NumPy array")

	def import_scalar_data_freesurfer(self, surface_name, mgh_path, adjacency=None, path_to_surface=None):
		"""
		Imports scalar data from a FreeSurfer MGH file.

		Parameters
		----------
		surface_name : str
			Name of the surface (e.g., 'lh', 'rh').
		mgh_path : str
			Path to the FreeSurfer MGH file.
		adjacency : str or np.ndarray, optional
			Either a file path to a .npy file containing the adjacency matrix or a NumPy array.
		path_to_surface : str, optional
			Path to the FreeSurfer surface file (.srf) to compute adjacency if 'adjacency' is not provided.

		Raises
		------
		AssertionError
			If neither 'adjacency' nor 'path_to_surface' is provided, or if both are provided.
		"""
		assert (adjacency is None) != (path_to_surface is None), "Error: exactly one of adjacency or srf must be None"
		scalar = nib.freesurfer.mghformat.load(mgh_path)
		data = scalar.get_fdata()[:, 0, 0, :]
		mask_data = np.zeros_like(data.mean(1))
		mask_data[data.mean(1) != 0] = 1
		n_vertices, n_subjects = data.shape
		self._check_surface(n_subjects)
		affine = scalar.affine
		header = scalar.header
		if adjacency is not None:
			adjacency = self._check_adjacency(adjacency)
		if path_to_surface is not None:
			v, f = load_surface_geometry(path_to_surface)
			adjacency = create_vertex_adjacency_neighbors(v, f)
		self.surface_names_.append(surface_name)
		self.image_data_.append(data[mask_data == 1].astype(np.float32, order = "C"))
		self.affine_.append(affine)
		self.header_.append(header)
		self.n_vertices_.append(n_vertices)
		self.mask_data_.append(mask_data)
		self.adjacency_.append(adjacency)

	def import_scalar_data_generic(self, surface_name, scalar_arr, adjacency=None, path_to_surface=None, affine=None, header=None):
		"""
		Imports scalar data from a generic NumPy array.

		Parameters
		----------
		surface_name : str
			Name of the surface (e.g., 'lh', 'rh').
		scalar_arr : np.ndarray
			Scalar data as a NumPy array.
		adjacency : str or np.ndarray, optional
			Either a file path to a .npy file containing the adjacency matrix or a NumPy array.
		path_to_surface : str, optional
			Path to the FreeSurfer surface file (.srf) to compute adjacency if 'adjacency' is not provided.
		affine : np.ndarray, optional
			Affine transformation matrix. If not provided, the default affine is used.
		header : str, optional
			Header information. If not provided, it is set to 'missing'.

		Raises
		------
		AssertionError
			If neither 'adjacency' nor 'path_to_surface' is provided, or if both are provided.
		"""
		assert (adjacency is None) != (path_to_surface is None), "Error: exactly one of adjacency or srf must be None"
		data = scalar_arr
		mask_data = np.zeros_like(data.mean(1))
		mask_data[data.mean(1) != 0] = 1
		n_vertices, n_subjects = data.shape
		self._check_surface(n_subjects)
		if adjacency is not None:
			adjacency = self._check_adjacency(adjacency)
		if path_to_surface is not None:
			v, f = load_surface_geometry(path_to_surface)
			adjacency = create_vertex_adjacency_neighbors(v, f)
		if affine is None:
			affine = self.default_affine_
		if header is None:
			header = 'missing'
		self.surface_names_.append(surface_name)
		self.image_data_.append(data[mask_data == 1].astype(np.float32, order = "C"))
		self.affine_.append(affine)
		self.header_.append(header)
		self.n_vertices_.append(n_vertices)
		self.mask_data_.append(mask_data)
		self.adjacency_.append(adjacency)

	def save(self, filename):
		"""
		Saves the MultiSurfaceImage instance as a pickle file.

		Parameters
		----------
		filename : str
			The name of the pickle file to be saved.

		Raises
		------
		AssertionError
			If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filename):
		"""
		Loads a MultiSurfaceImage instance from a pickle file.

		Parameters
		----------
		filename : str
			The name of the pickle file to load.

		Returns
		-------
		MultiSurfaceImage
			The loaded instance.

		Raises
		------
		AssertionError
			If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'rb') as f:
			return pickle.load(f)


class VoxelImage:
	"""
	VoxelImage is data class of efficiently loading voxel images, masking them by non-zero data, and calculatign their adjacency for statistical analyses. 
	It is possible to save and load the data classes as *.pkl objects. The masking and conversion to np.float32 substantially reduces the
	file size and RAM requirements in large datasets.
	e.g., TBSS_Skeleton.nii.gz with 1000 subjects ~ 800mb. TBSS_Skeleton.nii ~ 50gb. TBSS_Skeleton.pkl ~ 400mb.
		- This means that only 400mb is loaded into RAM instead of 50gb of RAM from loading the *.nii.gz directly.
	"""
	def __init__(self, *, binary_mask_path, image_path, gz_file_max = 400, tmp_dir = None):
		"""
		Initializes the VoxelImage class by loading the binary mask and voxel images.
		
		Parameters
		----------
		binary_mask_path : str
			Path to the binary mask file (must contain only 0s and 1s).
		image_path : str or list
			Path(s) to the image file(s).
		gz_file_max : float, default=400
			Sets an limit of *.gz file size in mb. A file above that limit will be 
			extracted to a temporary *.nii file prior to loading to save RAM. 
		tmp_dir : float, default=None
			Set the tempory directory. If None, the temporary *.nii will be created in 
			current directory.
		Raises
		----------
			AssertionError: If the mask is not binary.
			AssertionError: If image dimensions or affine transformations do not match the mask.
			TypeError: If 'image_path' is neither a string nor a list.
		"""
		# Load the mask
		mask = nib.load(binary_mask_path)
		mask_data = mask.get_fdata()
		assert np.all(np.unique(mask_data)==np.array([0,1])), "binary_mask_path must be a binary image containing only {1,0}."
		mask_data = self._check_mask(mask_data, connectivity = 3)
		self.mask_data_ = mask_data
		self.affine_ = mask.affine
		self.header_ = mask.header.copy()
		self.sform_ = mask.header.get_sform(coded=True)
		self.qform_ = mask.header.get_qform(coded=True)
		self.n_voxels_ = int(mask_data.sum())
		
		# Load the images
		self.image_path_ = image_path
		if isinstance(image_path, str):
			# be aware of zip size
			image_size_mb = np.divide(os.path.getsize(image_path), (1024*1024))
			if image_size_mb > gz_file_max:
				if image_path.endswith(".gz"):
					print("Extracting ['%s'] to save RAM" % image_path)
					if tmp_dir is None:
						outfile = '.tempfile.nii'
					else:
						outfile = os.path.join(tmp_dir, '.tempfile.nii')
					with gzip.open(image_path, 'rb') as file_in:
						with open(outfile, 'wb') as file_out:
							shutil.copyfileobj(file_in, file_out)
					img = nib.load(outfile)
					img_data = img.get_fdata()
					os.remove(outfile)
				else:
					warnings.warn("Cannot extract file [%s] with size %1.2fmb. Trying to load directly." % (image_path, image_size_mb), UserWarning)
					img = nib.load(image_path)
					img_data = img.get_fdata()
			else:
				img = nib.load(image_path)
				img_data = img.get_fdata()
			assert img_data.ndim == 4, "image must be ndim==4 if image_path is a str"
			assert np.all(self.affine_ == img.affine), "The affines of the mask and images must be equal"
			self.image_data_ = img_data[self.mask_data_==1].astype(np.float32, order = "C")
			self.n_images_ = self.image_data_.shape[1]
		elif isinstance(image_path, list):
			self.n_images_ = len(image_path)
			self.image_data_ = np.zeros((self.n_voxels_, self.n_images_)).astype(np.float32, order = "C")
			for s, path in enumerate(image_path):
				img_temp = nib.load(image_path)
				assert np.all(self.affine_ == img_temp.affine), "The affines of the mask and image [%s] must be equal" % os.path.basename(path)
				self.image_data_[:,s] = img_temp.get_fdata()[self.mask_data_==1]
		else:
			raise TypeError("image_path has to be a string or a list of strings")
		# Ensure that all data values are finite
		self.mask_data_ = self._finite_check(self.mask_data_)
		self.image_data_ = self._finite_check(self.image_data_)

	def _check_mask(self, mask_data, connectivity = 3):
		"""
		Ensures the binary mask is a single continuous region by keeping only the largest connected component.

		This method checks for disconnected regions in the binary mask and retains only the largest 
		connected component based on the specified connectivity. If multiple disconnected regions 
		are detected, the largest one is kept, and the mask is updated accordingly.

		Parameters
		----------
		mask_data : np.ndarray
			A 3D NumPy array representing the binary mask, containing only 0s and 1s.
		connectivity : int, default=3
			The connectivity used to define connected components in 3D space:
			- 1: 6-connectivity (faces only)
			- 2: 18-connectivity (faces and edges)
			- 3: 26-connectivity (faces, edges, and corners)

		Returns
		-------
		np.ndarray
			The processed binary mask where only the largest connected component is retained.

		Notes
		-----
		- If multiple disconnected components exist, the largest one is retained based on voxel count.
		- If only one continuous region exists, the mask remains unchanged.
		- The function assumes that the background (0s) is separate from the mask (1s).
		- If the mask is fully disconnected from the background, the largest region is selected 
		  based on the highest voxel count.
		"""
		labeled_array, num_labels = scipy_label(mask_data, structure=generate_binary_structure(3, connectivity))
		sizes = np.bincount(labeled_array.ravel())
		if num_labels > 2:
			if mask_data[labeled_array == 0].sum() == 0:
				sizes[0] = 0
				print("Non-continous mask detected with sizes: ", sizes[1:])
				print("Rebuilding mask detected largest continous label [%d]" % sizes[1])
				largest_label = sizes.argmax()
				mask_data == (labeled_array == largest_label)*1
			else:
				print("Non-continous mask detected with sizes: ", sizes)
				print("Rebuilding mask detected largest continous label [%d]" % sizes[0])
				largest_label = sizes.argmax()
				mask_data == (labeled_array == largest_label)*1
		return(mask_data)

	def _finite_check(self, arr):
		"""
		Check that are not any nan in a numpy array. If nan present, set the value to zero.

		Parameters
		----------
		arr : np.ndarray
			input array
		
		Returns
		-------
		finite_arr : np.ndarray
			output array with only finite values
		"""

		if np.sum(~np.isfinite(arr)) > 0:
			warnings.warn("nan values detected. They will be set to zero.", UserWarning)
			arr[~np.isfinite(arr)] = 0
		return(arr)

	def _create_adjacency_voxel(self, data_mask, connectivity_directions=26):
		"""
		Generates the adjacency set for the voxel image based on connectivity.
		
		Parameters
		----------
			data_mask : np.array
				A binary mask of non-zero data.
			connectivity_directions : int, default=26
				Number of connectivity directions {6 or 26}.
				Use 26 is all direction including diagonals (e.g., analysis of tbss skeleton data).
				Use 8 for only the immediatiately adjacent voxel (e.g., analysis of second level fMRI data).
		"""
		
		data_mask = data_mask.astype(np.float32)
		ind = np.where(data_mask == 1)
		dm = np.zeros_like(data_mask)
		x_dim, y_dim, z_dim = data_mask.shape
		adjacency = [set([]) for i in range(int(data_mask.sum()))]
		label = 0
		for x,y,z in zip(ind[0],ind[1],ind[2]):
			dm[x,y,z] = label
			label += 1
		for x,y,z in zip(ind[0],ind[1],ind[2]):
			xMin=max(x-1, 0)
			xMax=min(x+1, x_dim-1)
			yMin=max(y-1, 0)
			yMax=min(y+1, y_dim-1)
			zMin=max(z-1, 0)
			zMax=min(z+1, z_dim-1)
			local_area = dm[xMin:xMax+1,yMin:yMax+1,zMin:zMax+1]
			if int(connectivity_directions)==6:
				if local_area.shape!=(3,3,3): # check to prevent calculating adjacency at walls
					local_area = dm[x,y,z]
				else:
					local_area = local_area * np.array([0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,1,0,0,0,0]).reshape(3,3,3)
			cV = int(dm[x,y,z])
			for j in local_area[local_area>0]:
				adjacency[cV].add(int(j))
		adjacency = np.array([[x for x in sorted(i) if x != index] for index, i in enumerate(adjacency)], dtype=object)
		adjacency[0] = []
		return(adjacency)

	def generate_adjacency(self, connectivity_directions = 26):
		"""
		Helper function for creating the adjacency set based on connectivity.
		
		Parameters
		----------
			connectivity_directions : int, default=26
				Number of connectivity directions {6 or 26}.
				Use 26 is all direction including diagonals (e.g., analysis of tbss skeleton data).
				Use 8 for only the immediatiately adjacent voxel (e.g., analysis of second level fMRI data).
		Raises
		----------
			AssertionError: connectivity_directions is not 6 or 26.
		"""
		assert connectivity_directions==6 or connectivity_directions==26, "adjacency_directions must equal {6, 26}"
		if connectivity_directions == 6:
			new_mask = self._check_mask(self.mask_data_, connectivity = 1)
			if new_mask.sum() != self.mask_data_.sum():
				print("Resizing data to new mask")
				self.image_data_ = self.image_data_[new_mask[self.mask_data_==1] == 1]
				self.mask_data_ = new_mask
				self.n_voxels_ = int(self.mask_data.sum())
		self.adjacency_ = self._create_adjacency_voxel(self.mask_data_, connectivity_directions=connectivity_directions)

	def save(self, filename):
		"""
		Saves a VoxelImage instance as a pickle file.
		
		Parameters
		----------
			filename : str
				The name of the pickle file to be saved.
				
		Raises
		----------
			AssertionError: If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filename):
		"""
		Loads a VoxelImage instance from a pickle file.
		
		Parameters
		----------
			filename (str): The name of the pickle file to load.
		Returns
		-------
			VoxelImage : object
				The loaded instance.
		Raises
		-------
			AssertionError: If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'rb') as f:
			return pickle.load(f)

class LinearRegressionModelMRI:
	"""
	The LinearRegressionModelMRI class implements linear regression models specifically optimized for neuroimaging data analysis, with several technical advantages:
		Neuroimaging-Specific Tools: Supports both volume (NIfTI) and surface-based (FreeSurfer) data formats
		Threshold-Free Cluster Enhancement (TFCE) for both surface-based (FreeSurfer) and volumetric (NIfTI) data
		Advanced Vertex-wise/Voxel-wise Statistical Methods:
			Mediation analysis
			Nested model comparison for hypothesis testing
		Ease of Use: Patsy-like formula specification can be used with pandas dataframes
		Memory Efficiency: Much lower memory requirement that other neuroimaigng statistical software (e.g., ~36GB for ~1000 subjects with 300k verticies 
								with optional memory mapping for handling very large datasets.
		Computational Efficiency: For 1000 Subjects, voxel-based anaysis take around 25 minutes with 16 cores and 1.5h for vertex-wise analyses
		Visualization: Flexible exporting of results in standard neuroimaging formats and *.ply
		Reproducibility: Statistical models can be saved in data efficient manner
	"""
	def __init__(self, *, fit_intercept = True, n_jobs = 16, memory_mapping = False, use_tmp = True, fdr_correction = False, output_directory = None):
		"""
		Initialize the LinearRegressionModel instance.

		Parameters
		----------
		fit_intercept : bool, default=True
			Whether to calculate the intercept for this model. If set to False, no intercept will be used.
		n_jobs : int, default=None
			Currently, it does nothing. It is a placehold for further functions. The number of jobs to use for computation.
		memory_mapping : bool, default=False
			Write the dependent variable to a memory mapped file to save RAM.
		use_tmp : bool, default=True
			Write memory mapped files to temporary directory.
		tmp_directory : str, default = '/tmp'
			Path to temporary directory
		fdr_correction : bool, default=True
			Applies and BH_FDR correction to p-values
		output_directory : str, default = None
			The path/to results output directory. output_directory=None sets it to output_directory_{timestamp}
		"""
		self.fit_intercept_ = fit_intercept
		self.n_jobs_ = n_jobs
		self.memory_mapping_ = memory_mapping
		self.use_tmp_ = use_tmp
		self.tmp_directory_ = os.environ.get("TMPDIR", "/tmp/")
		self.fdr_correction_ = fdr_correction
		if output_directory is not None:
			self.output_directory_ = output_directory
		else:
			self.output_directory_ = "output_directory_%s" % datetime.datetime.now().strftime('%Y_%m_%d_%H%M')
		os.makedirs(self.output_directory_, exist_ok=True)

	def fit(self, X, y):
		"""
		Fit the linear regression model to the data. X can be a np.ndarray or a patsy-like formula (e.g., X='age + sex + site + diagnosis'). If using a
		a formula, make to to load a pandas dataframe first using load_pandas_dataframe or load_csv_dataframe.

		Parameters
		----------
		X : np.ndarray or str, shape(n_samples, n_features) or formula_like
			Exogneous variables
		
		y : np.ndarray or str, shape(n_samples, n_dependent_variables) or 'mapped'
			Endogenous variables

		Returns
		-------
		self : object
			Fitted model instance.
		"""
		if isinstance(X, str):
			assert hasattr(self, 'dataframe_'), "Pandas dataframe is missing (self.dataframe_) run load_pandas_dataframe or load_csv_dataframe first"
			X = self.dummy_code_from_formula(formula_like = X, save_columns_names = True)
			self.print_t_contrast_indices()
		y = self._handle_y_memmapping(y)
		X, y = self._check_inputs(X, y)
		if self.fit_intercept_:
			if np.mean(X[:,0]) != 1:
				X = self._stack_ones(X)
		n, k = X.shape # n_samples, rank
		df_between = k - 1
		df_within = n - k
		df_total = n - 1
		a = cy_lin_lstsqr_mat(X, y)
		leverage = (X * np.linalg.pinv(X).T).sum(1).reshape(-1,1)

		self.X_ = X 
		self.y_ = y
		self.n_ = n
		self.k_ = k
		self.df_between_ = df_between
		self.df_within_ = df_within
		self.df_total_ = df_total
		self.coef_ = a
		self.leverage_ = leverage
		return(self)

	def predict(self, X):
		"""
		Predict y values using the dot product of the coefficients.
		
		Parameters
		----------
		X : nd.array with shape (n_samples, n_features)
			Exogenous variables to predict y (yhat).
		Returns
		-------
		y_pred : ndarray of shape (n_samples,n_dependent_variables)
			The predicted endogenous variables.
		"""
		if self.fit_intercept_ and np.mean(X[:, 0]) != 1:
			X = self._stack_ones(X)
		return(np.dot(X, self.coef_))

	def _check_inputs(self, X, y):
		"""
		Check and validate inputs for the model.

		Parameters
		----------
		X : np.ndarray
			Exogeneous variables
			
		y : np.ndarray
			Endogenous variables
		
		Returns
		-------
		X : ndarray
			Reshaped if necessary.
			
		y : ndarray
		"""
		X = np.array(X)
		y = np.array(y)
		assert X.shape[0] == y.shape[0], "X and y have different lengths"
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if y.ndim == 1:
			y = y.reshape(-1,1)
		if np.sum(~np.isfinite(y)) > 0:
			warnings.warn("nan values detected in y. They will be set to zero.", UserWarning)
			y[~np.isfinite(y)] = 0
		if np.sum(~np.isfinite(X)) > 0:
			warnings.warn("nan values detected in y. They will be set to zero; however, this is a serious issue. Please check your exogenous variables.", UserWarning)
			X[~np.isfinite(X)] = 0
		return(X, y)
	
	def _stack_ones(self, arr):
		"""
		Add a column of ones to an array (used for intercept).
		
		Parameters
		----------
		arr : np.ndarray

		Returns
		---------
		arr : np.ndarray
			Array with a column of ones added as the first column.
		
		"""
		return(np.column_stack([np.ones(len(arr)),arr]))

	#Data Management

	def load_pandas_dataframe(self, df):
		"""
		Loads a pandas DataFrame into the object
		
		Parameters
		----------
		df : pd.DataFrame
			The pandas DataFrame to load
		
		Returns
		---------
		None
			This method does not return anything, it assigns the DataFrame to the object attribute.
		"""
		assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"
		self.dataframe_ = df

	def load_csv_dataframe(self, csv_file):
		"""
		Loads a pandas DataFrame from a CSV file into the object
		
		Parameters
		----------
		csv_file : str
			Path to the CSV file to load into the pandas DataFrame
		
		Returns
		---------
		None
			This method does not return anything, it assigns the DataFrame to the object attribute.
		"""
		assert csv_file.endswith(".csv"), "csv_file end with .csv"
		self.dataframe_ = pd.read_csv(csv_file)

	def dummy_code_from_formula(self, formula_like, save_columns_names = True, scale_dummy_arr = True, return_columns_names = False):
		"""
		Creates dummy-coded variables using patsy from the DataFrame and optionally scales them
		
		Parameters
		----------
		formula_like : str
			The formula that specifies which columns to include in the dummy coding (e.g., 'category1 + category2')
		save_columns_names : bool, optional, default=True
			If True, stores the names of the dummy-coded columns in the object attribute 't_contrast_names_'
		scale_dummy_arr : bool, optional, default=True
			If True, scales the resulting dummy variables (excluding the intercept column)
		return_columns_names : bool, default=False
			Special case in which the column names are also returned (dummy_arr, columns_names)
		
		Returns
		---------
		np.ndarray (if return_columns_names=True (np.ndarray, list))
			The scaled dummy-coded variables as a numpy array, with the intercept column excluded
		"""
		assert hasattr(self, 'dataframe_'), "Pandas dataframe is missing (self.dataframe_) run load_pandas_dataframe or load_csv_dataframe first"
		df_dummy = dmatrix(formula_like, data=self.dataframe_, NA_action="raise", return_type='dataframe')
		if save_columns_names:
			colnames =  df_dummy.columns.values
			self.t_contrast_names_ = np.array([sanitize_columns(col) for col in colnames])
			if not self.fit_intercept_:
				self.t_contrast_names_ = self.t_contrast_names_[1:]
			self.exogenous_variables_formula_ = formula_like
		dummy_arr = scale_arr(df_dummy.values[:,1:])
		if return_columns_names:
			colnames =  df_dummy.columns.values
			columns_names = np.array([sanitize_columns(col) for col in colnames])
			return(dummy_arr, columns_names)
		else:
			return(dummy_arr)

	def _handle_y_memmapping(self, y):
		"""Handles loading or creating memory-mapped y array."""
		if isinstance(y, str) and y == 'mapped':
			if not hasattr(self, 'memmap_y_name_') or self.memmap_y_name_ is None:
				raise FileNotFoundError("No memory mapped endogenous variables found. 'y' was 'mapped' but self.memmap_y_name_ is not set.")
			if not os.path.exists(self.memmap_y_name_):
				raise FileNotFoundError(f"Memory mapped file not found: {self.memmap_y_name_}")
			print(f"Loading memory mapped y from: {self.memmap_y_name_}")
			return(jload(self.memmap_y_name_, mmap_mode='r'))

		# If y is not 'mapped', check if we *should* map it
		y_checked = np.asarray(y) # Ensure it's an array first
		if self.memory_mapping_ and (not hasattr(self, 'memmap_y_name_') or self.memmap_y_name_ is None):
			timestamp = int(time.time() * 1000) # Use milliseconds for uniqueness
			data_filename_memmap = f"memmap_y_{timestamp}.mmap"

			if self.use_tmp_:
				filepath = os.path.join(self.tmp_directory_, data_filename_memmap)
			else:
				filepath = os.path.abspath(data_filename_memmap)

			# Ensure the directory exists
			os.makedirs(os.path.dirname(filepath), exist_ok=True)

			self.memmap_y_name_ = filepath
			print(f"Creating memory mapped y at: {self.memmap_y_name_}")
			dump(y_checked, self.memmap_y_name_)
			# Return the loaded memory-mapped array
			return(jload(self.memmap_y_name_, mmap_mode='r'))
		else:
			# If not mapping or already mapped, return the original (or checked) y
			return(y_checked)

	def _clean_memmmap(self):
		"""
		Delete memory mapped variables
		"""
		if hasattr(self, 'memmap_y_name_'):
			if os.path.isfile(self.memmap_y_name_):
				os.remove(self.memmap_y_name_)
			else:
				 print("Error: %s memory mapped file not found" % self.memmap_y_name_)
			self.memmap_y_name_ = None
		else:
			print("Error: No memory mapped files to remove.")

	# Statistical Analysis

	def calculate_tstatistics(self, calculate_probability = False):
		"""
		Calculate t-statistics for the model coefficients.

		Parameters
		----------
		calculate_probability : bool, default=False
			Whether to calculate the p-values for the t-statistics.

		Returns
		-------
		self : object
			Model instance with t-statistics and optionally p-values.
		"""
		assert hasattr(self, 'coef_'), "Run fit(X, y) first"
		invXX = np.linalg.inv(np.dot(self.X_.T, self.X_))
		sigma2 = np.divide(np.sum((self.y_ - np.dot(self.X_, self.coef_))**2, axis=0), (self.n_ - self.k_))
		se = fast_se_of_slope(invXX, sigma2)
		t = np.divide(self.coef_ , se)
		self.se_ = se
		self.t_ = t
		if np.sum(~np.isfinite(self.t_)) > 0:
			warnings.warn("nan values detected. They will be set to zero.", UserWarning)
			self.t_[~np.isfinite(self.t_)] = 0
		if calculate_probability:
			self.t_pvalues_ = tdist.sf(np.abs(self.t_), self.df_total_) * 2
			if self.fdr_correction_:
				self.t_qvalues_ = np.ones((self.t_.shape))
			for c in range(self.t_pvalues_.shape[0]):
				if self.fdr_correction_:
					self.t_qvalues_[c] = fdrcorrection(self.t_pvalues_[c])[1]
		return(self)

	def calculate_fstatistics(self, calculate_probability = True):
		"""
		Calculate F-statistics for the model.

		Parameters
		----------
		calculate_probability : bool, default=False
			Whether to calculate the p-values for the F-statistics.

		Returns
		-------
		self : object
			Model instance with F-statistics, R-squared, and optionally p-values
		"""
		assert hasattr(self, 'coef_'), "Run fit(X, y) first"
		if not hasattr(self, 'residuals_'):
			self.residuals_ = self.y_ - self.predict(self.X_)
			self.sse_ = np.sum(self.residuals_**2, axis=0)
			self.mse_ = np.divide(self.sse_, self.df_within_)
		self.sst_ = np.sum((self.y_ - np.mean(self.y_, 0))**2,0)
		self.ssb_ = self.sst_ - self.sse_
		self.f_ = np.divide(np.divide(self.ssb_, self.df_between_), self.mse_)
		self.Rsqr_ = 1 - (self.sse_/self.sst_)
		if calculate_probability:
			self.f_pvalues_ = fdist.sf(self.f_ , self.df_between_, self.df_within_)
			if self.fdr_correction_:
				self.f_qvalues_ = fdrcorrection(self.f_pvalues_)[1]
		return(self)

	def calculate_mediation_z_from_formula(self, ImageObjectMRI, X = None, M = None, y = None, covariates = None, calculate_probability = True):
		"""
		Perform Sobel mediation analysis using the provided formulas.

		Parameters
		----------
		ImageObjectMRI : object
			MRI image object containing:
			- image_data_: Image data  with shape(n_features, n_samples)
			- mask_data_: Binary mask of valid voxels/vertices
			- hemispheres_: (Optional) Present for surface-based data
		X : np.ndarray, shape (n_samples, n_features), optional
			Exogenous variable for the first stage in the mediation model.
		M : np.ndarray, shape (n_samples, n_features), optional
			Mediating variable in the model.
		y : np.ndarray, shape (n_samples, n_features), optional
			Endogenous variable (dependent variable) for the second stage.
		covariates : np.ndarray, shape (n_samples, n_covariates), optional
			Covariates to include in the model.
		calculate_probability : bool, default=True
			Whether to calculate the p-values for the z-statistics.
		
		Returns
		-------
		self : object
			Fitted model with mediation z-scores and p-values.
		"""
		assert hasattr(self, 'dataframe_'), "Pandas dataframe is missing (self.dataframe_) run load_pandas_dataframe or load_csv_dataframe first"
		not_none_count = sum(val is not None for val in (X, M, y))
		assert not_none_count == 2, "Two of X, M, and y must not be None"

		mri_data = ImageObjectMRI.image_data_.T
		if np.sum(~np.isfinite(mri_data)) > 0:
			warnings.warn("nan values detected in ImageObjectMRI. They will be set to zero.", UserWarning)
			mri_data[~np.isfinite(mri_data)] = 0

		if X is not None:
			varA = self.dummy_code_from_formula(X)
			assert varA.shape[1]==1, "Currently only 1d variables are supported"
			if covariates is not None:
				exogA = self._stack_ones(np.column_stack((self.dummy_code_from_formula(covariates), varA)))
			else:
				exogA = self._stack_ones(varA)
			if y is not None:
				exogB = np.column_stack((exogA, self.dummy_code_from_formula(y)))
				endogA = mri_data
				endogB = self.dummy_code_from_formula(y)
			if M is not None:
				exogB = np.column_stack((exogA, self.dummy_code_from_formula(M)))
				endogA = self.dummy_code_from_formula(M)
				endogB = mri_data
		else:
			varA = self.dummy_code_from_formula(M)
			if covariates is not None:
				exogA = self._stack_ones(np.column_stack((self.dummy_code_from_formula(covariates), varA)))
			else:
				exogA = self._stack_ones(varA)
			assert varA.shape[1]==1, "Currently only 1d variables are supported"
			exogB = np.column_stack((exogA, self.dummy_code_from_formula(y)))
			endogA = mri_data
			endogB = mri_data
		self.mediation_z_ = self._calculate_sobel(exogA, endogA, exogB, endogB)
		if calculate_probability:
			self.mediation_z_pvalues_ = 2 * norm.sf(abs(self.mediation_z_))
		self.mediation_exogA_ = exogA
		self.mediation_exogB_ = exogB
		self.mediation_endogA_ = endogA
		self.mediation_endogB_ = endogB
		return(self)

	def calculate_nested_model_from_formula(self, reduced_formula, calculate_effect_size = False, calculate_probability = True, calculate_log_ratio_test = False, calculate_aic = False, calculate_bic = False, estimate_z_statistic = True):
		"""Compare a full model to a nested reduced model using F-test and optionally LRT.
		
		This function evaluates the difference between a full model (based on 'self.X_') and a reduced model 
		specified by 'reduced_formula'. It computes the F-statistic for model comparison. Optionally, it calculates 
		effect sizes (partial eta-squared, partial omega-squared), p-values, z-statistics derived from the 
		F-statistic, and performs a Likelihood Ratio Test (LRT).

		Parameters
		----------
		reduced_formula : str
			Formula specifying the reduced model design matrix. Must result in
			fewer predictors than 'self.X_' and have the same number of
			observations. Predictors must be a subset of the full model.
		calculate_effect_size : bool, optional
			If True, computes partial eta squared and partial omega squared
			effect sizes. Default is False.
		calculate_probability : bool, optional
			If True, computes p-values for the F-statistic (and LRT if
			'calculate_log_ratio_test' is True). Applies FDR correction if
			'self.fdr_correction_' is enabled. Default is True.
		calculate_log_ratio_test : bool, optional
			If True, computes the Likelihood Ratio Test statistic (Chi-squared),
			associated p-value (if 'calculate_probability' is True), and the
			log-likelihood for both models. Required if 'calculate_aic' or 'calculate_bic'
			is True. Default is False.
		calculate_aic : bool, optional
			If True (and 'calculate_log_ratio_test' is True), calculates the
			difference in Akaike Information Criterion (AIC) between the full
			and reduced models. Default is False.
		calculate_bic : bool, optional
			If True (and 'calculate_log_ratio_test' is True), calculates the
			difference in Bayesian Information Criterion (BIC) between the full
			and reduced models. Default is False.
		estimate_z_statistic : bool, optional
			If True, estimates the z-statistic using an F-to-z conversion
			(Wilson-Hilferty approximation). Default is True.

		Attributes
		----------
		nested_model_f_ : numpy.ndarray
			The F-statistic comparing the reduced and full models for each
			dependent variable. Shape (n_targets,).
		nested_model_df_num_ : int
			Numerator degrees of freedom for the F-test (difference in number
			of predictors between full and reduced models).
		nested_model_df_den_ : int
			Denominator degrees of freedom for the F-test (observations -
			predictors in full model).
		nested_model_Xreduced_ : numpy.ndarray
			The design matrix for the reduced model. Shape (n_observations,
			n_reduced_predictors).
		nested_model_partial_eta_sq_ : numpy.ndarray, optional
			Partial eta squared, a measure of effect size.
			Shape (n_targets,). Only set if 'calculate_effect_size=True'.
		nested_model_partial_omega_sq_ : numpy.ndarray, optional
			Partial omega squared, an unbiased estimator of
			effect size. Shape (n_targets,). Only set if
			'calculate_effect_size=True'.
		nested_model_z_ : numpy.ndarray, optional
			The z-statistic derived from the F-statistic via Wilson-Hilferty
			approximation. Shape (n_targets,). Only set if
			'estimate_z_statistic=True'.
		nested_model_f_pvalues_ : numpy.ndarray, optional
			P-value associated with the F-statistic. Shape (n_targets,). Only
			set if 'calculate_probability=True'.
		nested_model_f_qvalues_ : numpy.ndarray, optional
			FDR-corrected p-value for the F-statistic. Shape (n_targets,).
			Only set if 'self.fdr_correction_' is enabled and
			'calculate_probability=True'.
		log_likelihood_ : numpy.ndarray, optional
			Log-likelihood of the full model. Shape (n_targets,). Only set if
			'calculate_log_ratio_test=True'.
		log_likelihood_reduced_ : numpy.ndarray, optional
			Log-likelihood of the reduced model. Shape (n_targets,). Only set
			if 'calculate_log_ratio_test=True'.
		nested_model_log_ratio_chi2_ : numpy.ndarray, optional
			Likelihood Ratio Test statistic,
			distributed as Chi-squared under the null hypothesis. Shape
			(n_targets,). Only set if 'calculate_log_ratio_test=True'.
		nested_model_log_ratio_test_chi2_pvalues_ : numpy.ndarray, optional
			P-value associated with the Likelihood Ratio Test statistic. Shape
			(n_targets,). Only set if 'calculate_log_ratio_test=True' and
			'calculate_probability=True'.
		nested_model_log_ratio_test_chi2_qvalues_ : numpy.ndarray, optional
			FDR-corrected p-value for the LRT statistic. Shape (n_targets,).
			Only set if 'calculate_log_ratio_test=True',
			'calculate_probability=True', and 'self.fdr_correction_' is True.
		nested_model_aic_difference_ : numpy.ndarray, optional
			Difference in AIC (AIC_full - AIC_reduced). Lower values favor the
			full model relative to the reduced one after penalizing for
			complexity. Shape (n_targets,). Only set if
			'calculate_log_ratio_test=True' and 'calculate_aic=True'.
		nested_model_bic_difference_ : numpy.ndarray, optional
			Difference in BIC (BIC_full - BIC_reduced). Lower values favor the
			full model relative to the reduced one after penalizing for
			complexity (more strongly than AIC). Shape (n_targets,). Only set
			if 'calculate_log_ratio_test=True' and 'use_bic=True'.

		Raises
		------
		AssertionError
			If 'self.dataframe_' or 'self.t_contrast_names_' are missing.
			If 'reduced_formula' results in a design matrix with more or equal
			predictors than 'self.X_', or a different number of observations.
			If predictors derived from 'reduced_formula' are not a subset of
			the full model predictors ('self.t_contrast_names_').
			If 'calculate_aic' or 'use_bic' is True but 'calculate_log_ratio_test'
			is False.

		Returns
		-------
		self
		"""
		assert hasattr(self, 'dataframe_'), "Pandas dataframe is missing (self.dataframe_) run load_pandas_dataframe or load_csv_dataframe first"
		assert hasattr(self, 't_contrast_names_'), "Contrast names are missing. Try first running: X = self.dummy_code_from_formula(formula_like = exogenous_formula, save_columns_names = True) or self.fit_from_formula(exogenous_formula, y)"
		Xreduced, columns_names = self.dummy_code_from_formula(reduced_formula, save_columns_names = False, scale_dummy_arr = True, return_columns_names = True)
		for col in columns_names:
			assert col in self.t_contrast_names_, "Reduced %s must be in full formula {%s}" % (col, self.exogenous_variables_formula_)
		if self.fit_intercept_:
			if np.mean(Xreduced[:,0]) != 1:
				Xreduced = self._stack_ones(Xreduced)
		assert self.X_.shape[1] > Xreduced.shape[1], "Xreduced must have a lower rank that the full model X"
		assert self.X_.shape[0] == Xreduced.shape[0], "Xreduced must have the same number of subjects as the full model X"
		coef_reduced = cy_lin_lstsqr_mat(Xreduced, self.y_)
		y_pred_full = self.predict(self.X_)
		y_pred_reduced = np.dot(Xreduced, coef_reduced)
		rss_full = np.sum((self.y_ - y_pred_full) ** 2, 0)
		rss_reduced = np.sum((self.y_ - y_pred_reduced) ** 2, 0)
		df_num = self.X_.shape[1] - Xreduced.shape[1] # df_effect
		df_den = self.X_.shape[0] - self.X_.shape[1] # df_error
		num_observations = self.X_.shape[0]
		self.nested_model_f_ = ((rss_reduced - rss_full) / df_num) / (rss_full / df_den)
		self.nested_model_f_ = np.maximum(0, self.nested_model_f_)
		if calculate_effect_size:
			self.nested_model_partial_eta_sq_ = (rss_reduced - rss_full) / rss_reduced
			omega_sq_p_num = df_num * (self.nested_model_f_ - 1)
			omega_sq_p_den = omega_sq_p_num + num_observations
			self.nested_model_partial_omega_sq_ = np.divide(omega_sq_p_num, omega_sq_p_den)
			self.nested_model_partial_omega_sq_ = np.clip(self.nested_model_partial_omega_sq_, 0, 1)
		if calculate_probability:
			self.nested_model_f_pvalues_ = fdist.sf(self.nested_model_f_, df_num, df_den)
			if self.fdr_correction_:
				self.nested_model_f_qvalues_ = fdrcorrection(self.nested_model_f_pvalues_)[1]
		if calculate_log_ratio_test:
			self.log_likelihood_ = self._calculate_log_likelihood(N = num_observations, rss = rss_full)
			self.log_likelihood_reduced_ = self._calculate_log_likelihood(N = num_observations, rss = rss_reduced)
			self.nested_model_log_ratio_chi2_ = -2*(self.log_likelihood_reduced_ - self.log_likelihood_)
			self.nested_model_log_ratio_chi2_ = np.maximum(0, self.nested_model_log_ratio_chi2_)
			if calculate_probability:
				self.nested_model_log_ratio_test_chi2_pvalues_ = chi2.sf(self.nested_model_log_ratio_chi2_, df_num)
				if self.fdr_correction_:
					self.nested_model_log_ratio_test_chi2_qvalues_ = fdrcorrection(self.nested_model_log_ratio_test_chi2_pvalues_)[1]
		if calculate_aic:
			assert calculate_log_ratio_test, ("'calculate_log_ratio_test' must be True for calculating AIC difference")
			self.nested_model_aic_difference_ = (self._calculate_aic(llf = self.log_likelihood_, k = int(self.X_.shape[1]+1)) - 
										self._calculate_aic(llf = self.log_likelihood_reduced_, k = int(Xreduced.shape[1]+1)))
		if calculate_bic:
			assert calculate_log_ratio_test, ("'calculate_log_ratio_test' must be True for calculating BIC difference")
			self.nested_model_bic_difference_ = (self._calculate_bic(llf = self.log_likelihood_, N = num_observations, k = int(self.X_.shape[1]+1)) -
										self._calculate_bic(llf = self.log_likelihood_reduced_, N = num_observations, k = int(Xreduced.shape[1]+1)))
		if estimate_z_statistic:
			self.nested_model_z_ = self.f_to_z_wilson_hilfert(self.nested_model_f_, df_num)
		self.nested_model_Xreduced_ = Xreduced
		self.nested_model_df_num_ = df_num
		self.nested_model_df_den_ = df_den
		return(self)

	def outlier_detection(self, f_quantile = 0.99, low_ram = True, outlier_tolerance_count = 2):
		"""
		Detect outliers using Cook's distance. Cook's distance is defined as the coefficient vector would move 
		if the sample were removed and the model refit.

		Parameters
		----------
		f_quantile : float
			The threshold for identifying outliers using the F-distribution.
		outlier_tolerance_count : int, default outlier_tolerance_count=2.
			The cutoff for the allowable number of outliers (i.e., outlier subjects). The percentage outlier will also be calculated.
		low_ram : bool, default = True
			Deletes self.residual_studentized_ and self.cooks_distance_ to save RAM
		
		Returns
		-------
		self : object
			Model instance with detected outliers based on Cook's distance.
		
		Reference
		-------
		Cook, R. D. (1977). Detection of Influential Observation in Linear Regression. Technometrics, 19(1), 15–18. doi:10.1080/00401706.1977.10489493 
		"""
		assert hasattr(self, 'coef_'), "Run fit(X, y) first"
		if not hasattr(self, 'residuals_'):
			self.residuals_ = self.y_ - np.dot(self.X_, self.coef_)
			self.sse_ = np.sum(self.residuals_**2, axis=0)
			self.mse_ = np.divide(self.sse_, self.df_within_)
		self.residuals_studentized_ = np.divide(np.divide(self.residuals_, np.sqrt(self.mse_)), np.sqrt(1 - self.leverage_))
		self.cooks_distance_ = np.divide(self.residuals_studentized_**2, self.k_) * np.divide(self.leverage_, (1 -  self.leverage_))
		self.cooks_distance_threshold_ = fdist.ppf(f_quantile, self.k_, (self.n_ - self.k_))
		self.n_outliers_ = (self.cooks_distance_ > self.cooks_distance_threshold_).sum(0)
		self.n_outliers_percentage_ = np.divide(self.n_outliers_ * 100, self.n_)
		self.outlier_ = np.zeros((len(self.n_outliers_)), int)
		self.outlier_[self.n_outliers_ > outlier_tolerance_count] = 1
		if low_ram:
			del self.residuals_studentized_ 
			del self.cooks_distance_
		return(self)

	# TFCE and Permutation Methods

	def calculate_statistics_tfce(self, ImageObjectMRI, mode, H = 2.0, E = 0.6667, contrast = None):
		"""
		Computes Threshold-Free Cluster Enhancement (TFCE) for specified statistics.

		This function applies the TFCE algorithm to enhance statistical maps 
		(t-statistics, mediation z-scores, or nested model z-scores)
		by accounting for spatial adjacency relationships, improving sensitivity 
		in neuroimaging analyses. Computes positive and potentially negative contrasts.

		Parameters
		----------
		ImageObjectMRI : object
			An instance containing neuroimaging data, including adjacency 
			information and mask data. Must have 'adjacency_' attribute.
			May have 'hemispheres_' attribute for surface-based analysis.
		mode : {'t', 'mediation', 'nested'}
			The type of statistic to apply TFCE to.
			- 't': Uses `self.t_`, computes positive and negative TFCE, respects 'contrast' parameter.
			- 'mediation': Uses 'self.mediation_z_', computes positive and negative TFCE.
			- 'nested': Uses 'self.nested_model_z_', computes only positive TFCE.
		H : float, optional
			The height exponent for TFCE computation (default is 2.0).
		E : float, optional
			The extent exponent for TFCE computation (default is 0.6667).
		contrast : int, None, optional
			For 't' mode only. Set which contrast (row index) to 
			calculate TFCE for. If None, calculates for all contrasts. 
			Other contrasts will be zero in the output. (Default is None).
		
		Raises
		------
		AssertionError
			If the required statistic (e.g., `self.t_`) has not been computed 
			before running TFCE, or if `ImageObjectMRI` lacks `adjacency_`.
		ValueError
			If `mode` is not one of the recognized values.
		
		Returns
		-------
		self : object
			The instance with updated attributes containing the computed 
			TFCE-enhanced statistics. Possible attributes set:
			- `self.t_tfce_positive_`, `self.t_tfce_negative_` (for 't')
			- `self.mediation_z_tfce_positive_`, `self.mediation_z_tfce_negative_` (for 'mediation_z')
			- `self.nested_model_z_tfce_` (for 'nested_model_z')
			Also sets `self.adjacency_set_`, `self.mask_data_`, `self.tfce_H_`, `self.tfce_E_`.
		
		Notes
		-----
		- If 'ImageObjectMRI' has a 'hemispheres_' attribute, TFCE is computed 
		  using a surface-based approach (`self._calculate_surface_tfce`).
		- Otherwise, a voxel-based TFCE computation is performed using 
		  `CreateAdjSet`.
		- For 't' statistics, skips TFCE if data is nearly constant or has 
		  extremely high values across most vertices/voxels.
		"""

		assert hasattr(ImageObjectMRI, 'adjacency_'), "ImageObjectMRI is missing adjacency_"
		if mode == 't':
			input_attr_name = 't_'
			output_attr_pos_name = 't_tfce_positive_'
			output_attr_neg_name = 't_tfce_negative_'
			assertion_message = "Run calculate_tstatistics() first"
			calculate_negative = True
		elif mode == 'mediation':
			input_attr_name = 'mediation_z_'
			output_attr_pos_name = 'mediation_z_tfce_positive_'
			output_attr_neg_name = 'mediation_z_tfce_negative_'
			assertion_message = "Run calculate_mediation_z_from_formula first"
			calculate_negative = True
		elif mode == 'nested':
			input_attr_name = 'nested_model_z_'
			output_attr_pos_name = 'nested_model_z_tfce_'
			assertion_message = "Run nested_model first with estimate_z_statistic = True"
			calculate_negative = False
		else:
			raise ValueError("Unknown mode: {%s}. Must be 't', 'mediation', or 'nested'." % mode)
		assert hasattr(self, input_attr_name), assertion_message
		input_stat_array = getattr(self, input_attr_name)

		output_stat_pos = np.zeros_like(input_stat_array).astype(np.float32, order="C")
		setattr(self, output_attr_pos_name, output_stat_pos)
		if calculate_negative:
			output_stat_neg = np.zeros_like(input_stat_array).astype(np.float32, order="C")
			setattr(self, output_attr_neg_name, output_stat_neg)

		iterator_ = None
		if mode == 't':
			iterator_ = np.arange(0, input_stat_array.shape[0])
			if contrast is not None:
				if isinstance(contrast, int):
					iterator_ = [iterator_[contrast]]
				else:
					iterator_ = [iterator_[i] for i in contrast]
		else:
			iterator_ = [0]
		is_surface = hasattr(ImageObjectMRI, 'hemispheres_')

		if is_surface:
			if mode == 't':
				for c in iterator_:
					current_stat = input_stat_array[c].astype(np.float32, order="C")
					if np.sum(current_stat > 0) < 100 or np.sum(current_stat < 0) < 100:
						print("The t-statistic is in the same direction for almost all vertices. Skipping TFCE calculation for Contrast-%d" % (c))
					elif np.sum(np.abs(current_stat) > 5) > int(current_stat.shape[0] * 0.90):
						print("abs(t-values)>5 detected for >90 percent of the vertices. Skipping TFCE calculation for Contrast-%d" % (c))
					else:
						tfce_values = self._calculate_surface_tfce(
							mask_data=ImageObjectMRI.mask_data_,
							statistic=current_stat,
							adjacency_set=ImageObjectMRI.adjacency_,
							H=H, E=E, return_max_tfce=False,
							only_positive_contrast=False)
						output_stat_pos[c] = tfce_values[0]
						output_stat_neg[c] = tfce_values[1]
			elif mode == 'mediation':
				current_stat = input_stat_array.astype(np.float32, order="C")
				tfce_values = self._calculate_surface_tfce(
					mask_data=ImageObjectMRI.mask_data_,
					statistic=current_stat,
					adjacency_set=ImageObjectMRI.adjacency_,
					H=H, E=E, return_max_tfce=False,
					only_positive_contrast=False)
				np.copyto(output_stat_pos, tfce_values[0])
				np.copyto(output_stat_neg, tfce_values[1])

			elif mode == 'nested':
				current_stat = input_stat_array.astype(np.float32, order="C")
				tfce_values = self._calculate_surface_tfce(
					mask_data=ImageObjectMRI.mask_data_,
					statistic=current_stat,
					adjacency_set=ImageObjectMRI.adjacency_,
					H=H, E=E, return_max_tfce=False,
					only_positive_contrast=True)
				np.copyto(output_stat_pos, tfce_values)
		else:
			# --- Voxel-based TFCE ---
			calcTFCE = CreateAdjSet(H, E, ImageObjectMRI.adjacency_)
			if mode == 't':
				for c in iterator_:
					current_stat = input_stat_array[c].astype(np.float32, order="C")
					if np.sum(current_stat > 0) < 100 or np.sum(current_stat < 0) < 100:
						print("The t-statistic is in the same direction for almost all vertices. Skipping TFCE calculation for Contrast-%d" % (c))
					elif np.sum(np.abs(current_stat) > 5) > int(current_stat.shape[0] * 0.90):
						print("abs(t-values)>5 detected for >90 percent of the vertices. Skipping TFCE calculation for Contrast-%d" % (c))
					else:
						current_stat = input_stat_array[c].astype(np.float32, order="C")
						stat_TFCE_pos = np.zeros_like(current_stat).astype(np.float32, order="C")
						calcTFCE.run(current_stat, stat_TFCE_pos)
						output_stat_pos[c] = stat_TFCE_pos
						stat_TFCE_neg = np.zeros_like(current_stat).astype(np.float32, order="C")
						calcTFCE.run(-current_stat, stat_TFCE_neg)
						output_stat_neg[c] = stat_TFCE_neg
			elif mode == 'mediation':
				current_stat = input_stat_array.astype(np.float32, order="C")
				stat_TFCE_pos = np.zeros_like(current_stat).astype(np.float32, order="C")
				calcTFCE.run(current_stat, stat_TFCE_pos)
				np.copyto(output_stat_pos, stat_TFCE_pos)
				stat_TFCE_neg = np.zeros_like(current_stat).astype(np.float32, order="C")
				calcTFCE.run(-current_stat, stat_TFCE_neg)
				np.copyto(output_stat_neg, stat_TFCE_neg)
			elif mode == 'nested':
				current_stat = input_stat_array.astype(np.float32, order="C")
				stat_TFCE_pos = np.zeros_like(current_stat).astype(np.float32, order="C")
				calcTFCE.run(current_stat, stat_TFCE_pos)
				np.copyto(output_stat_pos, stat_TFCE_pos)
		self.adjacency_set_ = ImageObjectMRI.adjacency_
		self.mask_data_ = ImageObjectMRI.mask_data_
		self.tfce_H_ = float(H)
		self.tfce_E_ = float(E)
		return(self)

	def permute_tfce(self, mode, n_permutations, contrast_index=None, whiten=True, use_chunks=True, 
					 chunk_size=768, stratification_blocks=None):
		"""
		Performs TFCE-based permutation testing for different statistical approaches.
		
		This function computes permutations and applies TFCE correction to obtain 
		the maximum TFCE values across permutations for t-statistics, mediation z-scores,
		or nested model comparisons.

		Parameters
		----------
		mode : str
			The type of permutation to run: {'t', 'mediation', 'nested'}
		n_permutations : int
			The number of permutations to perform.
		contrast_index : int, optional
			The index of the contrast for permutation testing (required for 't' mode).
		whiten : bool, optional
			Whether to whiten the residuals before permutation (default is True).
			Only applies to 't' and 'nested' modes.
		use_chunks : bool, default True
			Whether to use chunks for the permutation analysis. At the end of each chunk parallel 
			processing stops and restarts until the desired n_permutations is achieved. This is 
			helpful for any memory leaks. There should be anymore memory leaks now. The default 
			chunk_size is quite large at 768, so there's probably minimal impact on performance. 
			That is, it is safer to use chunking.
		chunk_size : int, default = 768
			The number of permutations per chunk. The default size is set as dividable by many 
			different number of cores such as 8, 6, 12, and 16.
			The number of permuations (total) will automatically adjust (increase in size) 
			so n_permutations % chunk_size = 0.
			For example, 2000 permutations ==> 2304 (3 chunks) or 10000 permutaions ==> 10752 (14 chunks).
		stratification_blocks : None or np.array (ndim =1), default None
			Shuffling within unique value of stratification block. 
			while still allowing for a valid assessment of the null hypothesis. This is particularly 
			useful when controlling for confounding variables or when dealing with clustered or 
			hierarchical data.
		"""
		assert hasattr(self, 'adjacency_set_'), "Run calculate_tfce first"
		
		# Mode-specific preparations
		if mode == 't':
			assert contrast_index is not None, "contrast_index is required for t mode"
			if self.memory_mapping_:
				assert hasattr(self, 'memmap_y_name_'), "No memory mapped endogenous variables found"
				y = jload(self.memmap_y_name_, mmap_mode='r')
			else:
				y = self.y_
			if whiten:
				y = y - self.predict(self.X_)
			X = self.X_
		elif mode == 'mediation':
			assert hasattr(self, 'mediation_z_tfce_positive_'), "Run calculate_mediation_z_tfce first"
		elif mode == 'nested':
			assert hasattr(self, 'adjacency_set_'), "Run calculate_nested_model_z_tfce first"
			if self.memory_mapping_:
				assert hasattr(self, 'memmap_y_name_'), "No memory mapped endogenous variables found"
				y = jload(self.memmap_y_name_, mmap_mode='r')
			else:
				y = self.y_
			if whiten:
				y = y - self.predict(self.X_)
			X = self.X_
			Xreduced = self.nested_model_Xreduced_
		else:
			raise ValueError(f"Unknown mode: {mode}. Use 't', 'mediation', or 'nested'")

		if stratification_blocks is not None:
			stratification_blocks = np.array(stratification_blocks)
			assert stratification_blocks.ndim == 1, "Error: stratification_blocks.ndim must equal 1"

		# Determine number of seeds based on mode
		seeds_divisor = 2 if mode in ['t', 'mediation'] else 1
		
		if use_chunks:
			tfce_maximum_values = []
			if n_permutations % chunk_size != 0:
				n_permutations += chunk_size - (n_permutations % chunk_size)
			print("Running %d permutations [p<0.0500 +/- %1.4f]" % (n_permutations,(2*np.sqrt(0.05*(1-0.05)/n_permutations))))
			n_chunks = int(n_permutations/chunk_size)
			for b in range(n_chunks):
				print("chunk[%d/%d]: %d Permutations" % (int(b+1), n_chunks, chunk_size))
				seeds = generate_seeds(n_seeds=int(chunk_size/seeds_divisor))
				
				if mode == 't':
					chunk_tfce_maximum_values = Parallel(n_jobs=self.n_jobs_, backend='multiprocessing')(
						delayed(self._run_tfce_t_permutation)(
							i=i, 
							X=X,
							y=y, 
							contrast_index=contrast_index,
							H=self.tfce_H_,
							E=self.tfce_E_,
							adjacency_set=self.adjacency_set_,
							mask_data=self.mask_data_,
							stratification_arr=stratification_blocks,
							seed=seeds[i]) for i in tqdm(range(int(chunk_size/seeds_divisor))))
				elif mode == 'mediation':
					chunk_tfce_maximum_values = Parallel(n_jobs=self.n_jobs_, backend='multiprocessing')(
						delayed(self._run_tfce_mediation_z_permutation)(
							i=i, 
							exogA=self.mediation_exogA_,
							endogA=self.mediation_endogA_,
							exogB=self.mediation_exogB_,
							endogB=self.mediation_endogB_,
							H=self.tfce_H_,
							E=self.tfce_E_,
							adjacency_set=self.adjacency_set_,
							mask_data=self.mask_data_,
							stratification_arr=stratification_blocks,
							seed=seeds[i]) for i in tqdm(range(int(chunk_size/seeds_divisor))))
				elif mode == 'nested':
					chunk_tfce_maximum_values = Parallel(n_jobs=self.n_jobs_, backend='multiprocessing')(
						delayed(self._run_nested_z_tfce_permutation)(
							i=i,
							X=X,
							Xreduced=Xreduced,
							y=y,
							H=self.tfce_H_,
							E=self.tfce_E_,
							adjacency_set=self.adjacency_set_,
							mask_data=self.mask_data_,
							stratification_arr=stratification_blocks,
							seed=seeds[i]) for i in tqdm(range(int(chunk_size/seeds_divisor))))
					
				tfce_maximum_values.append(chunk_tfce_maximum_values)
			tfce_maximum_values = np.array(tfce_maximum_values).ravel()
		else:
			seeds = generate_seeds(n_seeds=int(n_permutations/seeds_divisor))
			print("Running %d permutations [p<0.0500 +/- %1.4f]" % (n_permutations,(2*np.sqrt(0.05*(1-0.05)/n_permutations))))
			if mode == 't':
				tfce_maximum_values = Parallel(n_jobs=self.n_jobs_, backend='multiprocessing')(
					delayed(self._run_tfce_t_permutation)(
						i=i, 
						X=X,
						y=y, 
						contrast_index=contrast_index,
						H=self.tfce_H_,
						E=self.tfce_E_,
						adjacency_set=self.adjacency_set_,
						mask_data=self.mask_data_,
						stratification_arr=stratification_blocks,
						seed=seeds[i]) for i in tqdm(range(int(n_permutations/seeds_divisor))))
			elif mode == 'mediation':
				tfce_maximum_values = Parallel(n_jobs=self.n_jobs_, backend='multiprocessing')(
					delayed(self._run_tfce_mediation_z_permutation)(
						i=i, 
						exogA=self.mediation_exogA_,
						endogA=self.mediation_endogA_,
						exogB=self.mediation_exogB_,
						endogB=self.mediation_endogB_,
						H=self.tfce_H_,
						E=self.tfce_E_,
						adjacency_set=self.adjacency_set_,
						mask_data=self.mask_data_,
						stratification_arr=stratification_blocks,
						seed=seeds[i]) for i in tqdm(range(int(n_permutations/seeds_divisor))))
			elif mode == 'nested':
				tfce_maximum_values = Parallel(n_jobs=self.n_jobs_, backend='multiprocessing')(
					delayed(self._run_nested_z_tfce_permutation)(
						i=i,
						X=X,
						Xreduced=Xreduced,
						y=y,
						H=self.tfce_H_,
						E=self.tfce_E_,
						adjacency_set=self.adjacency_set_,
						mask_data=self.mask_data_,
						stratification_arr=stratification_blocks,
						seed=seeds[i]) for i in tqdm(range(int(n_permutations/seeds_divisor))))
			tfce_maximum_values = np.array(tfce_maximum_values).ravel()
		if mode == 't':
			self.t_tfce_max_permutations_ = np.array(tfce_maximum_values)
		elif mode == 'mediation':
			self.mediation_z_tfce_max_permutations_ = np.array(tfce_maximum_values)
		elif mode == 'nested':
			self.nested_model_z_tfce_max_permutations_ = np.array(tfce_maximum_values)

	def _calculate_surface_tfce(self, mask_data, statistic, adjacency_set, H = 2.0, E = 0.67, return_max_tfce = False, only_positive_contrast = False):
		"""
		Computes the TFCE (Threshold-Free Cluster Enhancement) statistic for surface-based data.
		
		This function calculates TFCE values separately for left and right hemispheres based on the provided adjacency set. 
		It supports returning either the full TFCE-enhanced statistic or only the maximum value.

		Parameters
		----------
		mask_data : list of numpy arrays
			Binary masks indicating valid data points for each hemisphere.
		statistic : numpy array
			The statistical values corresponding to the data points in the mask.
		adjacency_set : list
			A list containing adjacency information for left and right hemisphere.
		H : float, optional
			The height exponent for TFCE computation (default is 2.0).
		E : float, optional
			The extent exponent for TFCE computation (default is 0.67).
		return_max_tfce : bool, optional
			If True, returns only the maximum TFCE value; otherwise, returns full TFCE statistics (default is False).

		Returns
		-------
		tuple of numpy arrays
			TFCE-enhanced statistics for positive and negative contrasts.
		"""
		midpoint = mask_data[0].sum()
		
		vertStat_out_lh = np.zeros(mask_data[0].shape[0], dtype=np.float32, order="C")
		vertStat_out_rh = np.zeros(mask_data[1].shape[0], dtype=np.float32, order="C")
		vertStat_TFCE_lh = np.zeros_like(vertStat_out_lh).astype(np.float32, order = "C")
		vertStat_TFCE_rh = np.zeros_like(vertStat_out_rh).astype(np.float32, order = "C")

		vertStat_out_lh[mask_data[0] == 1] = statistic[:midpoint]
		vertStat_out_rh[mask_data[1] == 1] = statistic[midpoint:]

		calcTFCE_lh = CreateAdjSet(H, E, adjacency_set[0])
		calcTFCE_rh = CreateAdjSet(H, E, adjacency_set[1])
		calcTFCE_lh.run(vertStat_out_lh, vertStat_TFCE_lh)
		calcTFCE_rh.run(vertStat_out_rh, vertStat_TFCE_rh)

		if return_max_tfce:
			out_statistic_positive = np.max([vertStat_TFCE_lh, vertStat_TFCE_rh])
		else:
			out_statistic_positive = np.zeros_like(statistic).astype(np.float32, order = "C")
			out_statistic_positive[:midpoint] = vertStat_TFCE_lh[mask_data[0] == 1]
			out_statistic_positive[midpoint:] = vertStat_TFCE_rh[mask_data[1] == 1]

		if only_positive_contrast:
			adjacency_set = None
			vertStat_out_lh = None
			vertStat_out_rh = None
			vertStat_TFCE_lh = None
			vertStat_TFCE_rh = None
			calcTFCE_lh = None
			calcTFCE_rh = None
			del adjacency_set, vertStat_out_lh, vertStat_out_rh, vertStat_TFCE_lh, vertStat_TFCE_rh, calcTFCE_lh, calcTFCE_rh
			gc.collect()
			return(out_statistic_positive)
		else:
			vertStat_TFCE_lh.fill(0)
			vertStat_TFCE_rh.fill(0)

			calcTFCE_lh.run(-vertStat_out_lh, vertStat_TFCE_lh)
			calcTFCE_rh.run(-vertStat_out_rh, vertStat_TFCE_rh)

			if return_max_tfce:
				out_statistic_negative = np.max([vertStat_TFCE_lh, vertStat_TFCE_rh])
			else:
				out_statistic_negative = np.zeros_like(statistic).astype(np.float32, order = "C")
				out_statistic_negative[:midpoint] = vertStat_TFCE_lh[mask_data[0] == 1]
				out_statistic_negative[midpoint:] = vertStat_TFCE_rh[mask_data[1] == 1]
			adjacency_set = None
			vertStat_out_lh = None
			vertStat_out_rh = None
			vertStat_TFCE_lh = None
			vertStat_TFCE_rh = None
			calcTFCE_lh = None
			calcTFCE_rh = None
			del adjacency_set, vertStat_out_lh, vertStat_out_rh, vertStat_TFCE_lh, vertStat_TFCE_rh, calcTFCE_lh, calcTFCE_rh
			gc.collect()
			return(out_statistic_positive, out_statistic_negative)

	# inner permutation functions 
	def _run_tfce_t_permutation(self, i, X, y, contrast_index, H, E, adjacency_set, mask_data, stratification_arr, seed):
		"""
		Runs a single TFCE-based permutation test.
		
		This function shuffles the data, computes t-statistics, and applies the TFCE algorithm.
		
		Parameters
		----------
		i : int
			The permutation index (unused but required for parallel processing).
		X : numpy.ndarray
			The design matrix for the regression model.
		y : numpy.ndarray
			The response variable.
		contrast_index : int
			The contrast index being tested.
		H : float
			The height exponent for TFCE computation.
		E : float
			The extent exponent for TFCE computation.
		adjacency_set : list
			A set defining adjacency relationships between data points.
		seed : int or None
			The random seed for permutation.
		
		Returns
		-------
		tuple
			The maximum TFCE values for positive and negative contrasts.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		# Shuffle and compute regression
		n, k = X.shape
		if stratification_arr is not None:
			tmp_X = X[self._permute_stratified_blocks(stratification_arr, seed = seed)]
		else:
			tmp_X = np.random.permutation(X)
		a = cy_lin_lstsqr_mat(tmp_X, y)
		tmp_invXX = np.linalg.inv(np.dot(tmp_X.T, tmp_X))
		tmp_sigma2 = np.divide(np.sum((y - np.dot(tmp_X, a))**2, axis=0), (n - k))
		tmp_se = fast_se_of_slope(tmp_invXX, tmp_sigma2)
		tmp_t = np.divide(a , tmp_se)
		stat = tmp_t[contrast_index].astype(np.float32, order = "C")
		
		# Unlink variable from memory
		a = None
		tmp_X = None
		tmp_invXX = None
		tmp_sigma2 = None
		tmp_se = None
		tmp_t = None
		del a, tmp_X, tmp_invXX, tmp_sigma2, tmp_se, tmp_t # this is probably redundant, but won't hurt...
		
		if len(adjacency_set) == 2:
			tfce_values =  self._calculate_surface_tfce(mask_data = mask_data,
																		statistic = stat,
																		adjacency_set = adjacency_set,
																		H = H, E = E, return_max_tfce = True)
			max_pos, max_neg = tfce_values
		else:
			# Compute TFCE
			perm_calcTFCE = CreateAdjSet(H, E, adjacency_set)
			stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
			perm_calcTFCE.run(stat, stat_TFCE)
			max_pos = stat_TFCE.max()
			# Compute TFCE for negative statistics
			stat_TFCE.fill(0)
			perm_calcTFCE.run(-stat, stat_TFCE)
			max_neg = stat_TFCE.max()
			perm_calcTFCE = None
			stat_TFCE = None
		X = None
		y = None
		stat = None
		adjacency_set = None
		mask_data = None
		del adjacency_set, stat, mask_data, X, y
		gc.collect()
		return(max_pos, max_neg)

	def _run_tfce_mediation_z_permutation(self, i, exogA, endogA, exogB, endogB, H, E, adjacency_set, mask_data, stratification_arr, seed):
		"""
		Perform a single TFCE-based permutation test for mediation analysis.

		This method shuffles the data, calculates Sobel z-scores, and applies the TFCE algorithm 
		to assess the statistical significance of the mediation effect under permutation. 
		Returns the maximum positive and negative TFCE values from the permuted data.

		Parameters
		----------
		i : int
			The permutation index (required for parallel processing but unused).
		exogA : np.ndarray, shape (n_samples, n_features)
			Exogenous variables for the first stage in the mediation analysis.
		endogA : np.ndarray, shape (n_samples,)
			Endogenous variable for the first stage in the mediation analysis.
		exogB : np.ndarray, shape (n_samples, n_features)
			Exogenous variables for the second stage in the mediation analysis.
		endogB : np.ndarray, shape (n_samples,)
			Endogenous variable for the second stage in the mediation analysis.
		H : float
			Height exponent for TFCE computation (sensitivity to large values).
		E : float
			Extent exponent for TFCE computation (sensitivity to cluster size).
		adjacency_set : list
			Adjacency relationships between data points for TFCE neighborhood.
		mask_data : np.ndarray
			Binary mask indicating valid data points in the brain image.
		stratification_arr : np.ndarray or None
			Array defining blocks for constrained permutations. If None, performs 
			unconstrained permutations.
		seed : int or None
			Random seed for reproducibility. If None, uses random initialization.

		Returns
		-------
		tuple
			(max_positive_tfce, max_negative_tfce) values from the permutation.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)

		# Perform permutation
		if stratification_arr is not None:
			perm_idx = self._permute_stratified_blocks(stratification_arr, seed = seed)
		else:
			perm_idx = np.random.permutation(np.arange(exogA.shape[0]))
		tmp_z = self._calculate_sobel(exogA[perm_idx], endogA, exogB[perm_idx], endogB)

		# Compute TFCE
		if len(adjacency_set) == 2:
			tfce_values = self._calculate_surface_tfce(mask_data = mask_data,
																statistic = tmp_z.astype(np.float32, order = "C"),
																adjacency_set = adjacency_set,
																H = H,
																E = E,
																return_max_tfce = True,
																only_positive_contrast = False)
			max_pos, max_neg = tfce_values
		else:
			perm_calcTFCE = CreateAdjSet(H, E, adjacency_set)
			stat = tmp_z.astype(np.float32, order="C")
			stat_TFCE = np.zeros_like(stat).astype(np.float32, order="C")
			perm_calcTFCE.run(stat, stat_TFCE)
			max_pos = stat_TFCE.max()
			# Compute TFCE for negative statistics
			stat_TFCE.fill(0)
			perm_calcTFCE.run(-stat, stat_TFCE)
			max_neg = stat_TFCE.max()
			stat = None
			stat_TFCE = None
			perm_calcTFCE = None
			# Memory cleanup
			del stat, stat_TFCE, perm_calcTFCE
		mask_data = None
		tmp_z = None
		del tmp_z, mask_data
		gc.collect()
		return(max_pos, max_neg)

	def _run_nested_z_tfce_permutation(self, i, X, Xreduced, y, H, E, adjacency_set, mask_data, stratification_arr, seed):
		"""
		Runs a single TFCE-based permutation test on estimated z statistic from nested_model.
		
		This function shuffles the data, computes z-statistic, and applies the TFCE algorithm.
		
		Parameters
		----------
		i : int
			The permutation index.
		X : numpy.ndarray
			The design matrix for the regression model.
		Xreduced : numpy.ndarray
			The reduced design matrix for the regression model.
		y : numpy.ndarray
			The response variable.
		H : float
			The height exponent for TFCE computation.
		E : float
			The extent exponent for TFCE computation.
		adjacency_set : list
			A set defining adjacency relationships between data points.
		stratification_arr : list
			A list defining stratification blocks for permutation testing
		seed : int or None
			The random seed for permutation.
		
		Returns
		-------
		tuple
			The maximum TFCE values.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)

		assert X.shape[1] > Xreduced.shape[1], "Xreduced must have a lower rank that the full model X"
		assert X.shape[0] == Xreduced.shape[0], "Xreduced must have the same number of subjects as the full model X"

		if stratification_arr is not None:
			perm_idx = self._permute_stratified_blocks(stratification_arr, seed = seed)
		else:
			perm_idx = np.random.permutation(np.arange(X.shape[0]))
		tmp_X = X[perm_idx]
		tmp_Xreduced = Xreduced[perm_idx]

		tmp_coef = cy_lin_lstsqr_mat(tmp_X, y)
		tmp_coef_reduced = cy_lin_lstsqr_mat(tmp_Xreduced, y)
		tmp_y_pred_full = np.dot(tmp_X, tmp_coef)
		tmp_y_pred_reduced = np.dot(tmp_Xreduced, tmp_coef_reduced)
		tmp_rss_full = np.sum((y - tmp_y_pred_full) ** 2, 0)
		tmp_rss_reduced = np.sum((y - tmp_y_pred_reduced) ** 2, 0)
		tmp_df1 = tmp_X.shape[1] - tmp_Xreduced.shape[1]
		tmp_df2 = tmp_X.shape[0] - tmp_X.shape[1]
		temp_f = ((tmp_rss_reduced - tmp_rss_full) / tmp_df1) / (tmp_rss_full / tmp_df2)
		stat = self.f_to_z_wilson_hilfert(temp_f, tmp_df1).astype(np.float32, order = "C")

		# Unlink variable from memory
		a = None
		tmp_X = None
		tmp_Xreduced = None
		tmp_coef = None
		tmp_coef_reduced = None
		tmp_rss_full = None
		tmp_rss_reduced = None
		tmp_df1 = None
		tmp_df2 = None
		temp_f = None
		del a, tmp_X, tmp_Xreduced, tmp_coef, tmp_coef_reduced, tmp_rss_full, tmp_rss_reduced, tmp_df1, tmp_df2, temp_f # this is probably redundant, but won't hurt...

		if len(adjacency_set) == 2:
			max_pos = self._calculate_surface_tfce(mask_data = mask_data,
																		statistic = stat,
																		adjacency_set = adjacency_set,
																		H = H, E = E, return_max_tfce = True,
																		only_positive_contrast = True)
		else:
			# Compute TFCE
			perm_calcTFCE = CreateAdjSet(H, E, adjacency_set)
			stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
			perm_calcTFCE.run(stat, stat_TFCE)
			max_pos = stat_TFCE.max()
			perm_calcTFCE = None
			stat_TFCE = None
		X = None
		Xreduced = None
		y = None
		stat = None
		adjacency_set = None
		mask_data = None
		del adjacency_set, stat, mask_data, X, Xreduced, y
		gc.collect()
		return(max_pos)

	# Utility Functions

	def create_permutation_block_from_dataframe(self, stratification_variable):
		"""
		Creates a stratification array from the given variable in the DataFrame. 
		Ensures that no more than 25% of the sample has unique values to avoid over-stratification.

		Parameters
		----------
		stratification_variable : str
			The categorical variable in the DataFrame columns (self.dataframe_) used for stratification.

		Raises
		------
		AssertionError
			If self.dataframe_ is missing.
		AssertionError
			If more than 25% of the sample has unique values.

		Returns
		----
		stratification_arr : np.array
			np.array of stratification groups
		"""
		assert hasattr(self, 'dataframe_'), "Pandas dataframe is missing (self.dataframe_) run load_pandas_dataframe or load_csv_dataframe first"
		stratification_arr = np.array(self.dataframe_[stratification_variable].values)
		unique_variables = np.unique(stratification_arr)
		assert np.divide(len(unique_variables), len(stratification_arr)) < 0.25, "Error: More than 25% of the sample has unique variables"
		return(stratification_arr)

	def _permute_stratified_blocks(self, stratification_arr, seed = None):
		"""
		Perform stratified permutation of indices, maintaining group structure.

		Independently shuffles indices within each unique group defined by
		'stratification_arr', preserving the original group locations while
		randomizing order within groups.

		Parameters
		----------
		stratification_arr : np.ndarray, shape (n,)
			Categorical array defining group membership for each element.
			Elements with the same value are considered part of the same group.
		seed : int, optional
			Seed for reproducible random permutations. Uses a randomly generated
			seed from np.random.randint(0, 4294967295) when None (default).

		Returns
		-------
		np.ndarray, shape (n,)
			Array of indices permuted within stratification groups. Maintains:
			1. Original group locations (same values at same positions)
			2. All original indices appear exactly once
			3. Within-group order is randomized

		Notes
		-----
		- Sets numpy's global random seed during execution (via np.random.seed)
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		sorted_order = np.argsort(stratification_arr)
		inv_perm = np.argsort(sorted_order)
		_, counts = np.unique(stratification_arr[sorted_order], return_counts=True)
		split_indices = np.cumsum(counts)[:-1]
		chunks = np.split(sorted_order, split_indices)
		permuted_sorted = np.concatenate([np.random.permutation(chunk) for chunk in chunks])
		return(permuted_sorted[inv_perm])

	def f_to_z_wilson_hilfert(self, f_stats, df1):
		"""
		Wilson-Hilferty approximation of Z from F.
		
		Args:
			f_stats: NumPy array of F-statistic values (must be non-negative).
			df1: Numerator degrees of freedom (scalar).

		Returns:
			NumPy array of approximate Z-statistics.
		"""
		f_stats = np.asarray(f_stats)
		if not isinstance(df1, int) or df1 <= 0:
			raise ValueError("df1 must be a positive integer")

		# Calculate constants first
		nine_df1 = 9.0 * df1
		term2 = 1.0 - (2.0 / nine_df1)
		term3_sq_inv = nine_df1 / 2.0 # Calculate 1/term3^2
		term3_inv = np.sqrt(term3_sq_inv) # Calculate 1/term3

		with np.errstate(invalid='ignore'):
			term1 = np.power(f_stats, 1.0/3.0)
			z_approx = (term1 - term2) * term3_inv
			# Ensure negative F-stats result in Z-stat = 0
			z_approx[f_stats < 0] = 0
		return(z_approx)

	def _calculate_permuted_pvalue(self, permuted_distribution_arr, statistic_arr):
		"""
		Calculates p-values of an array from a permuted distribution
		
		Parameters
		----------
		permuted_distribution_arr : np.ndarray, shape(n_permutations)
			permuted statistic array
		
		statistic_arr : np.ndarray or str, shape(n_features)
			statistic array
		
		Returns
		---------
		arr : np.ndarray
			p-values of statistic array
		"""
		return(np.mean(permuted_distribution_arr[:, None] >= statistic_arr, axis=0, dtype=np.float32))

	def print_t_contrast_indices(self):
		"""
		Print the indices of t-contrasts.

		If the attribute 't_contrast_names_' exists, this function prints the index 
		and corresponding contrast name. Otherwise, it prints the numeric indices 
		for all available contrasts.

		Parameters:
		-----------
		self : object
			The instance containing 't_contrast_names_' and 't_' attributes.
		"""
		if hasattr(self, 't_contrast_names_'):
			for t in range(len(self.t_contrast_names_)):
				print("[index=%d] ==> %s" % (t, self.t_contrast_names_[t]))
		else:
			print(np.arange(self.t_.shape[0]))

	def _calculate_beta_se(self, exog, endog, index_var = -1):
		"""
		Calculate the standard error for the coefficients using linear regression.
		Parameters
		----------
		exog : np.ndarray, shape (n_samples, n_features)
			Exogenous variables (independent variables).
		endog : np.ndarray, shape (n_samples, n_dependent_variables)
			Endogenous variables (dependent variables).
		index_var : int, optional, default -1
			Index of the variable for which standard error is calculated.

		Returns
		-------
		tuple
			A tuple of (coefficient, standard error) for the specified variable.
		"""

		n = endog.shape[0]
		k = exog.shape[1]
		a = cy_lin_lstsqr_mat(exog, endog)
		sigma2 = np.sum((endog - np.dot(exog,a))**2,axis=0) / (n - k)
		invXX = np.linalg.inv(np.dot(exog.T, exog))
		se = fast_se_of_slope(invXX, sigma2)
		return(a[index_var], se[index_var])
	
	def _calculate_sobel(self, exogA, endogA, exogB, endogB):
		"""
		Calculate the Sobel z-score for mediation analysis.

		Parameters
		----------
		exogA : np.ndarray, shape (n_samples, n_features)
			Exogenous variables for the first stage.
		endogA : np.ndarray, shape (n_samples,)
			Endogenous variable for the first stage.
		exogB : np.ndarray, shape (n_samples, n_features)
			Exogenous variables for the second stage.
		endogB : np.ndarray, shape (n_samples,)
			Endogenous variable for the second stage.

		Returns
		-------
		float
			Sobel z-score for the mediation analysis.
		"""
		beta_a, se_a = self._calculate_beta_se(exogA, endogA, index_var = -1)
		beta_b, se_b = self._calculate_beta_se(exogB, endogB, index_var = -1)
		sobel_z = beta_a*beta_b / np.sqrt((beta_b**2 * se_a**2) + (beta_a**2 * se_b**2))
		return(sobel_z)

	def _calculate_aic(self, llf, k):
		"""Calculate Akaike Information Criterion (AIC) for a model.
		
		Parameters
		----------
		llf : float
			Log-likelihood of the model.
		k : int
			Number of parameters in the model (including intercept and variance).
			
		Returns
		-------
		float
			AIC value
		"""
		return(2*k - 2*llf)

	def _calculate_bic(self, llf, N, k):
		"""Calculate Bayesian Information Criterion (BIC) for a model.
		
		Parameters
		----------
		llf : float
			Log-likelihood of the model.
		N : int
			Number of observations
		k : int
			Number of parameters in the model (including intercept and variance).
			
		Returns
		-------
		float
			BIC value
		"""
		return(k*np.log(N) - 2*llf)

	def _calculate_log_likelihood(self, N, rss):
		"""Calculate the log-likelihood for a model.
		
		Parameters
		----------
		N : int
			Number of observations
		rss : float
			Residual sum of squares
			
		Returns
		-------
		float
			Log-likelihood value
		"""
		return(-N/2 * (np.log(2 * np.pi) + np.log(rss / N) + 1))

	# Results and visualization
	def write_tfce_results(self, ImageObjectMRI, mode='t', contrast_index=None, write_surface_ply=False, surface_ply_vmin=0.95, surface_ply_vmax=1.0, force_max_tfce_direction=False):
		"""
		Writes the Threshold-Free Cluster Enhancement (TFCE) results based on the specified mode.
		
		This function saves multiple NIfTI or mgh scalar images containing statistical values,
		TFCE values, and their respective corrected p-values.

		Parameters
		----------
		ImageObjectMRI : object
			An MRI image object containing mask data and affine transformation.
		mode : str, optional
			Processing mode: 't' for t-statistics, 'mediation' for mediation analysis,
			or 'nested' for nested model analysis (default: 't').
		contrast_index : int, optional
			The index of the contrast for which TFCE results will be written.
			Required for mode='t', ignored for other modes.
		write_surface_ply : bool, optional
			Whether to write surface PLY files for visualization (default: False).
		surface_ply_vmin : float, optional
			Minimum value threshold for surface visualization (default: 0.95).
		surface_ply_vmax : float, optional
			Maximum value threshold for surface visualization (default: 1.0).
		force_max_tfce_direction : bool, optional
			Advanced feature used only when there is bias in the permutated null 
			distribution due to stratification_blocks. Only applies to mode='t'.

		Raises
		------
		AssertionError
			If the required TFCE permutations have not been computed.
		ValueError
			If an invalid mode is specified or required parameters are missing.
		"""
		data_mask = ImageObjectMRI.mask_data_
		affine = ImageObjectMRI.affine_
		
		# Initialize based on mode
		if mode == 't':
			if contrast_index is None:
				raise ValueError("contrast_index is required for mode='t'")
			assert hasattr(self, 't_tfce_max_permutations_'), "Run permute_tfce with mode='t' first"
			# order is maintained for TFCE output
			if force_max_tfce_direction:
				max_tfce_arr_index = np.arange(len(self.t_tfce_max_permutations_))
				even_index = max_tfce_arr_index % 2 == 0
				t_tfce_max_permutations_pos = self.t_tfce_max_permutations_[even_index]
				t_tfce_max_permutations_neg = self.t_tfce_max_permutations_[~even_index]
			else:
				t_tfce_max_permutations_pos = self.t_tfce_max_permutations_
				t_tfce_max_permutations_neg = self.t_tfce_max_permutations_
			# FWER accuracy
			accuracy = {
				"n_permutations": len(t_tfce_max_permutations_pos),
				"p": 0.05,
				"confidence +/- ": "%1.6f" % (2*np.sqrt(0.05*(1-0.05)/len(t_tfce_max_permutations_pos))),
				"margin-of-error": "%1.3f" % np.divide((2*np.sqrt(0.05*(1-0.05)/len(t_tfce_max_permutations_pos))), 0.05)
			}
			self.t_tfce_permutation_accuracy_ = accuracy

			if hasattr(self, 't_contrast_names_') and self.t_.shape[0] == len(self.t_contrast_names_):
				contrast_name = "tvalue-%s" % self.t_contrast_names_[int(contrast_index)]
			else:
				contrast_name = "tvalue-con%d" % np.arange(0, len(self.t_),1)[int(contrast_index)]

			if not hasattr(self, 't_tfce_positive_oneminusp_'):
				self.t_tfce_positive_oneminusp_ = np.zeros_like(self.t_tfce_positive_)
				self.t_tfce_negative_oneminusp_ = np.zeros_like(self.t_tfce_negative_)
			
			values = self.t_[contrast_index]
			if len(data_mask) == 2:
				self.write_freesurfer_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + ".mgh")
				values = self.t_tfce_positive_[contrast_index]
				self.write_freesurfer_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive.mgh")
				oneminuspfwe_pos = 1 - self._calculate_permuted_pvalue(t_tfce_max_permutations_pos, values)
				self.write_freesurfer_image(values=oneminuspfwe_pos, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive-1minusp.mgh")
				
				values = self.t_tfce_negative_[contrast_index]
				self.write_freesurfer_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative.mgh")
				oneminuspfwe_neg = 1 - self._calculate_permuted_pvalue(t_tfce_max_permutations_neg, values)
				self.write_freesurfer_image(values=oneminuspfwe_neg, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative-1minusp.mgh")
				
				if write_surface_ply:
					write_cortical_surface_results_to_ply(
						positive_scalar_array=oneminuspfwe_pos,
						ImageObjectMRI=ImageObjectMRI,
						outname=os.path.join(self.output_directory_, contrast_name + "-tfce-1minusp.ply"),
						negative_scalar_array=oneminuspfwe_neg,
						vmin=surface_ply_vmin,
						vmax=surface_ply_vmax,
						lh_srf_path=os.path.join(static_directory, 'lh.midthickness.srf'),
						rh_srf_path=os.path.join(static_directory, 'rh.midthickness.srf'),
						perform_surface_smoothing=True, n_smoothing_iterations=50,
						positive_cmap='red-yellow',
						negative_cmap='blue-lightblue'
					)
			else:
				self.write_nibabel_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + ".nii.gz")
				values = self.t_tfce_positive_[contrast_index]
				self.write_nibabel_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive.nii.gz")
				oneminuspfwe_pos = 1 - self._calculate_permuted_pvalue(t_tfce_max_permutations_pos, values)
				self.write_nibabel_image(values=oneminuspfwe_pos, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive-1minusp.nii.gz")
				
				values = self.t_tfce_negative_[contrast_index]
				self.write_nibabel_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative.nii.gz")
				oneminuspfwe_neg = 1 - self._calculate_permuted_pvalue(t_tfce_max_permutations_neg, values)
				self.write_nibabel_image(values=oneminuspfwe_neg, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative-1minusp.nii.gz")
				
			self.t_tfce_positive_oneminusp_[contrast_index] = oneminuspfwe_pos
			self.t_tfce_negative_oneminusp_[contrast_index] = oneminuspfwe_neg
			
		elif mode == 'mediation':
			assert hasattr(self, 'mediation_z_tfce_positive_'), "Run calculate_mediation_z_tfce first"
			assert hasattr(self, 'mediation_z_tfce_max_permutations_'), "Run permute_tfce with mode='mediation' first"
			contrast_name = "mediation-zvalue"
			accuracy = {
				"n_permutations": len(self.mediation_z_tfce_max_permutations_),
				"p": 0.05,
				"confidence +/- ": "%1.6f" % (2*np.sqrt(0.05*(1-0.05)/len(self.mediation_z_tfce_max_permutations_))),
				"margin-of-error": "%1.3f" % np.divide((2*np.sqrt(0.05*(1-0.05)/len(self.mediation_z_tfce_max_permutations_))), 0.05)
			}
			self.mediation_z_tfce_permutation_accuracy_ = accuracy
			if not hasattr(self, 'mediation_z_tfce_positive_oneminusp_'):
				self.mediation_z_tfce_positive_oneminusp_ = np.zeros_like(self.mediation_z_tfce_positive_)
				self.mediation_z_tfce_negative_oneminusp_ = np.zeros_like(self.mediation_z_tfce_negative_)
			
			if len(data_mask) == 2:
				self.write_freesurfer_image(values=self.mediation_z_, data_mask=data_mask, affine=affine, outname=contrast_name + ".mgh")
				# Positive TFCE
				self.write_freesurfer_image(values=self.mediation_z_tfce_positive_, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive.mgh")
				oneminuspfwe_pos = 1 - self._calculate_permuted_pvalue(self.mediation_z_tfce_max_permutations_, self.mediation_z_tfce_positive_)
				self.write_freesurfer_image(values=oneminuspfwe_pos, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive-1minusp.mgh")
				# Negative TFCE
				self.write_freesurfer_image(values=self.mediation_z_tfce_negative_, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative.mgh")
				oneminuspfwe_neg = 1 - self._calculate_permuted_pvalue(self.mediation_z_tfce_max_permutations_, self.mediation_z_tfce_negative_)
				self.write_freesurfer_image(values=oneminuspfwe_neg, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative-1minusp.mgh")
				
				if write_surface_ply:
					write_cortical_surface_results_to_ply(
						positive_scalar_array=oneminuspfwe_pos,
						ImageObjectMRI=ImageObjectMRI,
						outname=os.path.join(self.output_directory_, contrast_name + "-tfce-1minusp.ply"),
						negative_scalar_array=oneminuspfwe_neg,
						vmin=surface_ply_vmin,
						vmax=surface_ply_vmax,
						lh_srf_path=os.path.join(static_directory, 'lh.midthickness.srf'),
						rh_srf_path=os.path.join(static_directory, 'rh.midthickness.srf'),
						perform_surface_smoothing=True, n_smoothing_iterations=50,
						positive_cmap='red-yellow',
						negative_cmap='blue-lightblue'
					)
			else:
				self.write_nibabel_image(values=self.mediation_z_, data_mask=data_mask, affine=affine, outname=contrast_name + ".nii.gz")
				# Positive TFCE
				self.write_nibabel_image(values=self.mediation_z_tfce_positive_, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive.nii.gz")
				oneminuspfwe_pos = 1 - self._calculate_permuted_pvalue(self.mediation_z_tfce_max_permutations_, self.mediation_z_tfce_positive_)
				self.write_nibabel_image(values=oneminuspfwe_pos, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_positive-1minusp.nii.gz")
				# Negative TFCE
				self.write_nibabel_image(values=self.mediation_z_tfce_negative_, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative.nii.gz")
				oneminuspfwe_neg = 1 - self._calculate_permuted_pvalue(self.mediation_z_tfce_max_permutations_, self.mediation_z_tfce_negative_)
				self.write_nibabel_image(values=oneminuspfwe_neg, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce_negative-1minusp.nii.gz")
				
			self.mediation_z_tfce_positive_oneminusp_ = oneminuspfwe_pos
			self.mediation_z_tfce_negative_oneminusp_ = oneminuspfwe_neg
		elif mode == 'nested':
			assert hasattr(self, 'nested_model_z_tfce_max_permutations_'), "Run permute_tfce with mode='nested'"
			nested_model_z_tfce_max_permutations = self.nested_model_z_tfce_max_permutations_
			accuracy = {
				"n_permutations": len(nested_model_z_tfce_max_permutations),
				"p": 0.05,
				"confidence +/- ": "%1.6f" % (2*np.sqrt(0.05*(1-0.05)/len(nested_model_z_tfce_max_permutations))),
				"margin-of-error": "%1.3f" % np.divide((2*np.sqrt(0.05*(1-0.05)/len(nested_model_z_tfce_max_permutations))), 0.05)
			}
			self.nested_model_z_tfce_permutation_accuracy_ = accuracy
			contrast_name = "nested-zvalue"
			
			self.nested_model_z_tfce_oneminusp_ = np.zeros_like(self.nested_model_z_)
			
			values = self.nested_model_z_
			if len(data_mask) == 2:
				self.write_freesurfer_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + ".mgh")
				values = self.nested_model_z_tfce_
				self.write_freesurfer_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce.mgh")
				oneminuspfwe = 1 - self._calculate_permuted_pvalue(nested_model_z_tfce_max_permutations, values)
				self.write_freesurfer_image(values=oneminuspfwe, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce-1minusp.mgh")
				
				if write_surface_ply:
					write_cortical_surface_results_to_ply(
						positive_scalar_array=oneminuspfwe,
						ImageObjectMRI=ImageObjectMRI,
						outname=os.path.join(self.output_directory_, contrast_name + "-tfce-1minusp.ply"),
						negative_scalar_array=None,
						vmin=surface_ply_vmin,
						vmax=surface_ply_vmax,
						lh_srf_path=os.path.join(static_directory, 'lh.midthickness.srf'),
						rh_srf_path=os.path.join(static_directory, 'rh.midthickness.srf'),
						perform_surface_smoothing=True, n_smoothing_iterations=50,
						positive_cmap='red-yellow',
						negative_cmap='blue-lightblue'
					)
			else:
				self.write_nibabel_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + ".nii.gz")
				values = self.nested_model_z_tfce_
				self.write_nibabel_image(values=values, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce.nii.gz")
				oneminuspfwe = 1 - self._calculate_permuted_pvalue(nested_model_z_tfce_max_permutations, values)
				self.write_nibabel_image(values=oneminuspfwe, data_mask=data_mask, affine=affine, outname=contrast_name + "-tfce-1minusp.nii.gz")
				
			self.nested_model_z_tfce_oneminusp_ = oneminuspfwe
		else:
			raise ValueError("Invalid mode. Choose from: 't', 'mediation', 'nested'")

	def write_freesurfer_image(self, values, data_mask, affine, outname):
		"""
		Saves an array of values as FreeSurfer MGH surface scalar images.

		This function applies a binary mask to the input values and saves them as separate 
		left-hemisphere (lh.mgh) and right-hemisphere (rh.mgh) FreeSurfer MGH files. The input 
		data is split based on the number of valid data points in the first hemisphere.

		Parameters
		----------
		values : numpy.ndarray
			A 1D array of scalar values to be mapped onto the FreeSurfer surface.
		data_mask : list of numpy.ndarray
			A list containing two binary masks (one for each hemisphere) indicating valid data points.
		affine : numpy.ndarray
			The affine transformation matrix associated with the image data.
		outname : str
			The base filename for the output MGH images. Must end with ".mgh".
		
		Notes
		-----
		- The function assumes that 'data_mask[0]' corresponds to the left hemisphere 
		  and 'data_mask[1]' corresponds to the right hemisphere.
		- The 'values' array should contain concatenated values for both hemispheres.
		- The output files will be saved as "<outname>.lh.mgh" and "<outname>.rh.mgh".
		"""
		if outname.endswith(".mgh"):
			midpoint = data_mask[0].sum()
			outdata = np.zeros((data_mask[0].shape[0]))
			outdata[data_mask[0]==1] = values[:midpoint]
			outname_lh = os.path.join(self.output_directory_, outname[:-4] + ".lh.mgh")
			nib.save(nib.freesurfer.mghformat.MGHImage(outdata[:,np.newaxis, np.newaxis].astype(np.float32), affine[0]), outname_lh)
			outdata.fill(0)
			outdata[data_mask[1]==1] = values[midpoint:]
			outname_rh = os.path.join(self.output_directory_, outname[:-4] + ".rh.mgh")
			nib.save(nib.freesurfer.mghformat.MGHImage(outdata[:,np.newaxis, np.newaxis].astype(np.float32), affine[1]), outname_rh)

	def write_nibabel_image(self, values, data_mask, affine, outname):
		"""
		Saves an array of values as a NIfTI image.
		
		This function applies a binary mask to the data and saves it as a NIfTI file using the given affine transformation.

		Parameters
		----------
		values : numpy.ndarray
			The data values to be saved in the NIfTI image.
		data_mask : numpy.ndarray
			A binary mask indicating valid data points.
		affine : numpy.ndarray
			The affine transformation matrix for the NIfTI image.
		outname : str
			The filename for the output NIfTI image. Must end with ".nii.gz".
		"""
		if outname.endswith(".nii.gz"):
			outdata = np.zeros((data_mask.shape))
			outdata[data_mask==1] = values
			nib.save(nib.Nifti1Image(outdata, affine), os.path.join(self.output_directory_, outname))

	# Serialization
	def save(self, filename):
		"""
		Saves a LinearRegressionModelMRI instance as a pickle file.
		
		Parameters
		----------
			filename : str
				The name of the pickle file to be saved.
				
		Raises
		----------
			AssertionError: If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filename):
		"""
		Loads a LinearRegressionModelMRI instance from a pickle file.
		
		Parameters
		----------
			filename (str): The name of the pickle file to load.
		Returns
		-------
			VoxelImage : object
				The loaded instance.
		Raises
		-------
			AssertionError: If the filename does not end with .pkl.
		"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'rb') as f:
			return pickle.load(f)


# additional functions
def save_ply(v, f, outname, color_array=None, output_binary=True):
	"""
	Save vertex and face data to a PLY file.

	Parameters:
	-----------
	v : numpy.ndarray
		A 2D array of shape (N, 3) containing the vertex coordinates (x, y, z).
	f : numpy.ndarray
		A 2D array of shape (M, 3) containing the face indices (vertex indices).
	outname : str
		The output filename. If the filename does not end with '.ply', the extension will be added.
	color_array : numpy.ndarray, optional
		A 2D array of shape (N, 3) containing the RGB color values for each vertex. 
		If provided, the colors will be included in the PLY file.
	output_binary : bool, optional
		If True, the PLY file will be saved in binary format. If False, it will be saved in ASCII format.
		Default is True.

	Returns:
	--------
	None
	"""
	
	# Check file extension
	if not outname.endswith('ply'):
		outname += '.ply'
	if not output_binary:
		outname = outname[:-4] + '.ascii.ply'
	if os.path.exists(outname):
		os.remove(outname)

	# Write header
	header = "ply\n"
	if output_binary:
		header += "format binary_{}_endian 1.0\n".format(sys.byteorder)
		if sys.byteorder == 'little':
			output_fmt = '<'
		else:
			output_fmt = '>'
	else:
		header += "format ascii 1.0\n"
	header += "comment made with TFCE_mediation\n"
	header += "element vertex {}\n".format(len(v))
	header += "property float x\n"
	header += "property float y\n"
	header += "property float z\n"
	if color_array is not None:
		header += "property uchar red\n"
		header += "property uchar green\n"
		header += "property uchar blue\n"
	header += "element face {}\n".format(len(f))
	header += "property list uchar int vertex_index\n"
	header += "end_header\n"

	# Write to file
	if output_binary:
		with open(outname, "a") as o:
			o.write(header)
		with open(outname, "ab") as o:
			for i in range(len(v)):
				if color_array is not None:
					o.write(struct.pack(output_fmt + 'fffBBB', v[i, 0], v[i, 1], v[i, 2], color_array[i, 0], color_array[i, 1], color_array[i, 2]))
				else:
					o.write(struct.pack(output_fmt + 'fff', v[i, 0], v[i, 1], v[i, 2]))
			for j in range(len(f)):
				o.write(struct.pack('<Biii', 3, f[j, 0], f[j, 1], f[j, 2]))
	else:
		with open(outname, "a") as o:
			o.write(header)
			for i in range(len(v)):
				if color_array is not None:
					o.write("%1.6f %1.6f %1.6f %d %d %d\n" % (v[i, 0], v[i, 1], v[i, 2], color_array[i, 0], color_array[i, 1], color_array[i, 2]))
				else:
					o.write("%1.6f %1.6f %1.6f\n" % (v[i, 0], v[i, 1], v[i, 2]))
			for j in range(len(f)):
				o.write("3 %d %d %d\n" % (f[j, 0], f[j, 1], f[j, 2]))


def linear_cm(c0, c1, c2=None):
	"""
	Creates a linear color map lookup table between two or three colors.

	Parameters:
	-----------
	c0 : tuple or list
		A tuple or list of length 3 representing the RGB values of the starting color.
	c1 : tuple or list
		A tuple or list of length 3 representing the RGB values of the middle color.
	c2 : tuple or list, optional
		A tuple or list of length 3 representing the RGB values of the ending color.
		If not provided, the color map transitions directly from 'c0' to 'c1'.

	Returns:
	--------
	c_map : numpy.ndarray
		A 2D array of shape (256, 3) representing the linear color map.
		Each row corresponds to an RGB color value.
	"""
	c_map = np.zeros((256, 3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128, i] = np.linspace(c0[i], c1[i], 128)
			c_map[127:256, i] = np.linspace(c1[i], c2[i], 129)
	else:
		for i in range(3):
			c_map[:, i] = np.linspace(c0[i], c1[i], 256)
	return c_map


def log_cm(c0, c1, c2=None):
	"""
	Creates a logarithmic color map lookup table between two or three colors.

	Parameters:
	-----------
	c0 : tuple or list
		A tuple or list of length 3 representing the RGB values of the starting color.
	c1 : tuple or list
		A tuple or list of length 3 representing the RGB values of the middle color.
	c2 : tuple or list, optional
		A tuple or list of length 3 representing the RGB values of the ending color.
		If not provided, the color map transitions directly from 'c0' to 'c1'.

	Returns:
	--------
	c_map : numpy.ndarray
		A 2D array of shape (256, 3) representing the logarithmic color map.
		Each row corresponds to an RGB color value.
	"""
	c_map = np.zeros((256, 3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128, i] = np.geomspace(c0[i] + 1, c1[i] + 1, 128) - 1
			c_map[127:256, i] = np.geomspace(c1[i] + 1, c2[i] + 1, 129) - 1
	else:
		for i in range(3):
			c_map[:, i] = np.geomspace(c0[i] + 1, c1[i] + 1, 256) - 1
	return c_map


def erf_cm(c0, c1, c2=None):
	"""
	Creates a color map lookup table using the error function (erf) between two or three colors.

	Parameters:
	-----------
	c0 : tuple or list
		A tuple or list of length 3 representing the RGB values of the starting color.
	c1 : tuple or list
		A tuple or list of length 3 representing the RGB values of the middle color.
	c2 : tuple or list, optional
		A tuple or list of length 3 representing the RGB values of the ending color.
		If not provided, the color map transitions directly from 'c0' to 'c1'.

	Returns:
	--------
	c_map : numpy.ndarray
		A 2D array of shape (256, 3) representing the error function-based color map.
		Each row corresponds to an RGB color value.
	"""
	c_map = np.zeros((256, 3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128, i] = erf(np.linspace(3 * (c0[i] / 255), 3 * (c1[i] / 255), 128)) * 255
			c_map[127:256, i] = erf(np.linspace(3 * (c1[i] / 255), 3 * (c2[i] / 255), 129)) * 255
	else:
		for i in range(3):
			c_map[:, i] = erf(np.linspace(3 * (c0[i] / 255), 3 * (c1[i] / 255), 256)) * 255
	return c_map


def create_rywlbb_gradient_cmap(linear_alpha=False, return_array=True):
	"""
	Creates a colormap for a gradient from red-yellow-white-light blue-blue (rywlbb).

	Parameters:
	-----------
	linear_alpha : bool, optional
		If True, applies a linear alpha gradient to the colormap. Default is False.
	return_array : bool, optional
		If True, returns the colormap as a NumPy array. If False, returns a matplotlib colormap object.
		Default is True.

	Returns:
	--------
	cmap_array : numpy.ndarray or matplotlib.colors.LinearSegmentedColormap
		If 'return_array' is True, returns a 2D array of shape (256, 4) representing the RGBA values of the colormap.
		If 'return_array' is False, returns a matplotlib colormap object.
	"""
	colors = ["#00008C", "#2234A8", "#4467C4", "#659BDF", "#87CEFB", "white", "#ffec19", "#ffc100", "#ff9800", "#ff5607", "#f6412d"]
	cmap = LinearSegmentedColormap.from_list("rywlbb-gradient", colors)
	cmap._init()  # Initialize the colormap
	if return_array:
		crange = np.linspace(0, 1, 256)
		cmap_array = cmap(crange)
		if linear_alpha:
			cmap_array[:, -1] = np.abs(np.linspace(-1, 1, 256))
		cmap_array *= 255
		cmap_array = cmap_array.astype(int)
		return cmap_array
	else:
		if linear_alpha:
			cmap._lut[:256, -1] = np.abs(np.linspace(-1, 1, 256))
		return cmap


def create_ryw_gradient_cmap(linear_alpha=False, return_array=True):
	"""
	Creates a colormap for a gradient from red-yellow-white (ryw).

	Parameters:
	-----------
	linear_alpha : bool, optional
		If True, applies a linear alpha gradient to the colormap. Default is False.
	return_array : bool, optional
		If True, returns the colormap as a NumPy array. If False, returns a matplotlib colormap object.
		Default is True.

	Returns:
	--------
	cmap_array : numpy.ndarray or matplotlib.colors.LinearSegmentedColormap
		If 'return_array' is True, returns a 2D array of shape (256, 4) representing the RGBA values of the colormap.
		If 'return_array' is False, returns a matplotlib colormap object.
	"""
	colors = ["white", "#ffec19", "#ffc100", "#ff9800", "#ff5607", "#f6412d"]
	cmap = LinearSegmentedColormap.from_list("ryw-gradient", colors)
	cmap._init()  # Initialize the colormap
	if return_array:
		crange = np.linspace(0, 1, 256)
		cmap_array = cmap(crange)
		if linear_alpha:
			cmap_array[:, -1] = np.linspace(0, 1, 256)
		cmap_array *= 255
		cmap_array = cmap_array.astype(int)
		return cmap_array
	else:
		if linear_alpha:
			cmap._lut[:256, -1] = np.linspace(0, 1, 256)
		return cmap


def create_lbb_gradient_cmap(linear_alpha=False, return_array=True):
	"""
	Creates a colormap for a gradient from light blue-blue (lbb).

	Parameters:
	-----------
	linear_alpha : bool, optional
		If True, applies a linear alpha gradient to the colormap. Default is False.
	return_array : bool, optional
		If True, returns the colormap as a NumPy array. If False, returns a matplotlib colormap object.
		Default is True.

	Returns:
	--------
	cmap_array : numpy.ndarray or matplotlib.colors.LinearSegmentedColormap
		If 'return_array' is True, returns a 2D array of shape (256, 4) representing the RGBA values of the colormap.
		If 'return_array' is False, returns a matplotlib colormap object.
	"""
	colors = ["white", "#87CEFB", "#659BDF", "#4467C4", "#2234A8", "#00008C"]
	cmap = LinearSegmentedColormap.from_list("lbb-gradient", colors)
	cmap._init()  # Initialize the colormap
	if return_array:
		crange = np.linspace(0, 1, 256)
		cmap_array = cmap(crange)
		if linear_alpha:
			cmap_array[:, -1] = np.linspace(0, 1, 256)
		cmap_array *= 255
		cmap_array = cmap_array.astype(int)
		return cmap_array
	else:
		if linear_alpha:
			cmap._lut[:256, -1] = np.linspace(0, 1, 256)
		return cmap


def vectorized_surface_smoothing(v, f, number_of_iter = 20, lambda_w = 0.5, use_taubin = False, weighted = True):
	"""
	Applies Laplacian (Gaussian) or Taubin (low-pass) smoothing with option to smooth single volume. Laplacian is the default.
	
	Citations
	----------
	
	Herrmann, Leonard R. (1976), "Laplacian-isoparametric grid generation scheme", Journal of the Engineering Mechanics Division, 102 (5): 749-756.
	Taubin, Gabriel. "A signal processing approach to fair surface design." Proceedings of the 22nd annual conference on Computer graphics and interactive techniques. ACM, 1995.
	
	Parameters
	----------
	v : array
		vertex array
	f : array
		face array
	adjacency : array
		adjacency array

	Flags
	----------
	number_of_iter : int
		number of smoothing iterations
	lambda_w : float
		lamda weighting of degree of movement for each iteration
		The weighting should never be above 1.0
	use_taubin : bool
		Use taubin (no shrinkage) instead of laplacian (which cause surface shrinkage) smoothing.
		
	Returns
	-------
	v_ : array
		smoothed vertices array
	f_ : array
		f = face array (unchanged)
	"""
	v_ = np.array(v).copy()
	f_ = np.array(f).copy()
	adjacency_ = create_vertex_adjacency_neighbors(v_, f_)

	k = 0.1
	mu_w = -np.divide(lambda_w, (1-k*lambda_w))

	lengths = np.array([len(a) for a in adjacency_])
	maxlen = max(lengths)
	padded = [list(a) + [-1] * (maxlen - len(a)) for a in adjacency_]
	adj = np.array(padded)
	adj_mask = adj != -1  # Mask for valid adjacency indices

	w = np.ones(adj.shape, dtype=float)
	w[adj<0] = 0.
	val = (adj>=0).sum(-1).reshape(-1, 1)
	val[val == 0] = 1
	w /= val
	w = w.reshape(adj.shape[0], adj.shape[1], 1)

	for iter_num in range(number_of_iter):
		if weighted:
			vadj = v_[adj]
			vadj = np.swapaxes(vadj, 1, 2)
			weights = np.zeros((v_.shape[0], maxlen))
			for col in range(maxlen):
				dist = np.linalg.norm(vadj[:, :, col] - v, axis=1)
				dist[dist == 0] = 1
				weights[:, col] = np.power(dist, -1)
			weights[~adj_mask] = 0
			vectors = np.einsum('abc,adc->acd', weights[:,None], vadj)
			with np.errstate(divide='ignore', invalid='ignore'):
				if iter_num % 2 == 0:
					v_ += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v_)
				elif use_taubin:
					v_ += mu_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v_)
				else:
					v_ += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v_)
			v_[np.isnan(v_)] = np.array(v)[np.isnan(v_)] # hacky vertex nan fix
		else:
			with np.errstate(divide='ignore', invalid='ignore'):
				if iter_num % 2 == 0:
					v_ += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v_[adj], 0, 1)-v_)).sum(0)
				elif use_taubin:
					v_ += np.array(mu_w*np.swapaxes(w,0,1)*(np.swapaxes(v_[adj], 0, 1)-v_)).sum(0)
				else:
					v_ += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v_[adj], 0, 1)-v_)).sum(0)
	else:
		return (v_, f_)

def display_luts():
	"""
	Displays a visual representation of colormaps (LUTs - Look-Up Tables) available in Matplotlib,
	including custom colormaps. This function is adapted from a Matplotlib example.

	The function creates a vertical stack of colormaps, including both built-in Matplotlib colormaps
	and custom colormaps. Each colormap is displayed as a horizontal gradient.

	Notes
	-----
	- The function uses 'matplotlib.pyplot' to render the colormaps.
	- Custom colormaps are defined using RGB values and are normalized to the range [0, 1].
	- The function includes both linear and logarithmic colormaps.

	References
	----------
	- Adapted from: https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
	- Original example from the SciPy Cookbook, author unknown.
	"""
	# Create a gradient array for visualization
	a = np.linspace(0, 1, 256).reshape(1, -1)
	a = np.vstack((a, a))

	maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
	custom_maps = [
		'red-yellow', 'blue-lightblue', 'green-lightgreen', 'tm-breeze', 'tm-sunset',
		'tm-broccoli', 'tm-octopus', 'tm-storm', 'tm-flow', 'tm-logBluGry',
		'tm-logRedYel', 'tm-erfRGB', 'rywlbb-gradient', 'ryw-gradient', 'lbb-gradient'
	]
	maps.extend(custom_maps)

	nmaps = len(maps) + 1
	fig = plt.figure(figsize=(8, 12))
	fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
	for i, m in enumerate(maps):
		ax = plt.subplot(nmaps, 1, i + 1)
		plt.axis("off")  # Hide axes
		if m == 'red-yellow':
			cmap_array = linear_cm([255, 0, 0], [255, 255, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'blue-lightblue':
			cmap_array = linear_cm([0, 0, 255], [0, 255, 255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'green-lightgreen':
			cmap_array = linear_cm([0, 128, 0], [0, 255, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-breeze':
			cmap_array = linear_cm([199, 233, 180], [65, 182, 196], [37, 52, 148]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-sunset':
			cmap_array = linear_cm([255, 255, 51], [255, 128, 0], [204, 0, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-storm':
			cmap_array = linear_cm([0, 153, 0], [255, 255, 0], [204, 0, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-flow':
			cmap_array = log_cm([51, 51, 255], [255, 0, 0], [255, 255, 255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-logBluGry':
			cmap_array = log_cm([0, 0, 51], [0, 0, 255], [255, 255, 255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-logRedYel':
			cmap_array = log_cm([102, 0, 0], [200, 0, 0], [255, 255, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-erfRGB':
			cmap_array = erf_cm([255, 0, 0], [0, 255, 0], [0, 0, 255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-broccoli':
			cmap_array = linear_cm([204, 255, 153], [76, 153, 0], [0, 102, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'tm-octopus':
			cmap_array = linear_cm([255, 204, 204], [255, 0, 255], [102, 0, 0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'rywlbb-gradient':
			cmap_array = create_rywlbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'ryw-gradient':
			cmap_array = create_ryw_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		elif m == 'lbb-gradient':
			cmap_array = create_lbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array, m), origin='lower')
		else:
			# Use built-in Matplotlib colormaps
			plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
		pos = list(ax.get_position().bounds)
		fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
	plt.show()

def get_cmap_array(lut, image_alpha=1.0, c_reverse=False):
	"""
	Generate an RGBA colormap array based on the specified lookup table (lut) and parameters.
	Use display_matplotlib_luts() to see the available luts.

	Parameters
	----------
	lut : str
		Lookup table name or abbreviation. Accepted values include:
		- 'r-y' or 'red-yellow'
		- 'b-lb' or 'blue-lightblue'
		- 'g-lg' or 'green-lightgreen'
		- 'tm-breeze', 'tm-sunset', 'tm-broccoli', 'tm-octopus', 'tm-storm', 'tm-flow', 'tm-logBluGry', 'tm-logRedYel', 'tm-erfRGB', 'tm-white'
		- 'rywlbb-gradient', 'ryw-gradient', 'lbb-gradient'
		- Any matplotlib colormap (e.g., 'viridis', 'plasma').
	image_alpha : float, optional
		Alpha value for the colormap colors. Default is 1.0.
	c_reverse : bool, optional
		Whether to reverse the colormap array. Default is False.

	Returns
	-------
	cmap_array : numpy.ndarray
		Custom RGBA colormap array of shape (256, 4) with values in the range of [0, 255].

	Raises
	------
	ValueError
		If the lookup table is not recognized.
	"""
	# Handle reversed colormaps
	if lut.endswith('_r'):
		c_reverse = True
		lut = lut[:-2]

	# Define custom colormap mappings
	custom_cmaps = {
		'r-y': linear_cm([255, 0, 0], [255, 255, 0]),
		'red-yellow': linear_cm([255, 0, 0], [255, 255, 0]),
		'b-lb': linear_cm([0, 0, 255], [0, 255, 255]),
		'blue-lightblue': linear_cm([0, 0, 255], [0, 255, 255]),
		'g-lg': linear_cm([0, 128, 0], [0, 255, 0]),
		'green-lightgreen': linear_cm([0, 128, 0], [0, 255, 0]),
		'tm-breeze': linear_cm([199, 233, 180], [65, 182, 196], [37, 52, 148]),
		'tm-sunset': linear_cm([255, 255, 51], [255, 128, 0], [204, 0, 0]),
		'tm-broccoli': linear_cm([204, 255, 153], [76, 153, 0], [0, 102, 0]),
		'tm-octopus': linear_cm([255, 204, 204], [255, 0, 255], [102, 0, 0]),
		'tm-storm': linear_cm([0, 153, 0], [255, 255, 0], [204, 0, 0]),
		'tm-flow': log_cm([51, 51, 255], [255, 0, 0], [255, 255, 255]),
		'tm-logBluGry': log_cm([0, 0, 51], [0, 0, 255], [255, 255, 255]),
		'tm-logRedYel': log_cm([102, 0, 0], [200, 0, 0], [255, 255, 0]),
		'tm-erfRGB': erf_cm([255, 0, 0], [0, 255, 0], [0, 0, 255]),
		'tm-white': linear_cm([255, 255, 255], [255, 255, 255]),
		'rywlbb-gradient': create_rywlbb_gradient_cmap(),
		'ryw-gradient': create_ryw_gradient_cmap(),
		'lbb-gradient': create_lbb_gradient_cmap(),
	}

	# Generate the colormap array
	if lut in custom_cmaps:
		cmap_array = custom_cmaps[lut]
		if isinstance(cmap_array, np.ndarray):
			cmap_array = np.column_stack((cmap_array, 255 * np.ones(256) * image_alpha))
	else:
		try:
			cmap_array = plt.cm.get_cmap(lut)(np.arange(256))
			cmap_array[:, 3] = image_alpha
			cmap_array *= 255
		except:
			raise ValueError(
				f"Lookup table '{lut}' is not recognized. "
				"Accepted values include custom colormaps (e.g., 'r-y', 'b-lb') "
				"or any matplotlib colormap (e.g., 'viridis', 'plasma')."
			)
	# Reverse the colormap if requested
	if c_reverse:
		cmap_array = cmap_array[::-1]
	return cmap_array.astype(int)


def _vertex_paint(positive_scalar, negative_scalar=None, vmin=0.95, vmax=1.0, background_color_rbga=[220, 210, 195, 255], positive_cmap='red-yellow', negative_cmap='blue-lightblue'):
	"""
	Applies color mapping to vertices based on positive and negative scalar values. 
	Positive values are mapped to a specified colormap (e.g., red-yellow), while negative values 
	are mapped to another colormap (e.g., blue-lightblue). Background color is applied to vertices 
	that do not meet the threshold.
	
	Important, thresholding on the 'negative_scalar' is done for same vmin/vmax positive values.
	So, if positive_scalar=tvalues, then negative_scalar=-tvalues.

	Parameters
	----------
	positive_scalar : array
		Array of positive scalar values for each vertex.
	negative_scalar : array, optional
		Array of negative scalar values for each vertex. If provided, it must have the same length as 'positive_scalar'.
	vmin : float, optional
		Minimum threshold value for applying colormap. Default is 0.95.
	vmax : float, optional
		Maximum threshold value for applying colormap. Default is 1.0.
	background_color_rbga : list, optional
		Background color in RGBA format for vertices that do not meet the threshold. Default is [220, 210, 195, 255].
	positive_cmap : str, optional
		Colormap name for positive scalar values. Default is 'red-yellow'.
	negative_cmap : str, optional
		Colormap name for negative scalar values. Default is 'blue-lightblue'.

	Returns
	-------
	out_color_arr : array
		Array of RGBA colors for each vertex, shaped as (n_vertices, 4).

	Notes
	-----
	- If 'negative_scalar' is provided, it must have the same length as 'positive_scalar'.
	- The function uses 'matplotlib.colors.Normalize' and 'ListedColormap' for color mapping.
	- Vertices with scalar values below 'vmin' are assigned the background color.
	"""
	if negative_scalar is not None:
		assert len(positive_scalar) == len(negative_scalar), "positive and negative scalar must have the same length"
	pos_cmap_arr = get_cmap_array(positive_cmap)
	neg_cmap_arr = get_cmap_array(negative_cmap)
	out_color_arr = np.ones((len(positive_scalar), 4), int) * 255
	out_color_arr[:] = background_color_rbga
	norm = Normalize(vmin, vmax)
	if np.sum(positive_scalar > vmin) != 0:
		cmap = ListedColormap(np.divide(pos_cmap_arr, 255))
		mask = positive_scalar > vmin
		vals = np.round(cmap(norm(positive_scalar[mask])) * 255).astype(int)
		out_color_arr[mask] = vals
	if np.sum(negative_scalar > vmin) != 0:
		cmap = ListedColormap(np.divide(neg_cmap_arr, 255))
		mask = negative_scalar > vmin
		vals = np.round(cmap(norm(negative_scalar[mask])) * 255).astype(int)
		out_color_arr[mask] = vals
	return(out_color_arr)


def _add_annotation_wireframe(v, f, freesurfer_annotation_path):
	labels, _, _ = nib.freesurfer.read_annot(freesurfer_annotation_path)
	a = np.array([len(set(labels[f[k]])) != 1 for k in range(len(f))])
	scalar_out = np.zeros_like(labels).astype(np.float32)
	scalar_out[np.unique(f[a])] = 1
	return(scalar_out)


def write_cortical_surface_results_to_ply(positive_scalar_array, ImageObjectMRI, outname, negative_scalar_array = None, vmin = 0.95, vmax = 1.0, lh_srf_path = os.path.join(static_directory, 'lh.midthickness.srf'), rh_srf_path = os.path.join(static_directory, 'rh.midthickness.srf'), perform_surface_smoothing = True, n_smoothing_iterations = 100, background_color_rbga=[220, 210, 195, 255], positive_cmap='red-yellow', negative_cmap='blue-lightblue'):
	"""
	Writes cortical surface results to PLY files by applying color mapping to vertices based on scalar values.
	The function processes both left and right hemispheres, optionally smooths the surface, and saves the 
	results as PLY files.

	Parameters
	----------
	positive_scalar_array : array
		Array of positive scalar values for each vertex.
	ImageObjectMRI : object
		MRI image object containing cortical surface mask data.
	outname : str
		Output filename prefix for the generated PLY files.
	negative_scalar_array : array, optional
		Array of negative scalar values for each vertex. If provided, it must have the same length as 'positive_scalar_array'.
	vmin : float, optional
		Minimum threshold value for applying colormap. Default is 0.95.
	vmax : float, optional
		Maximum threshold value for applying colormap. Default is 1.0.
	lh_srf_path : str, optional
		Path to the left hemisphere surface file. Default is 'lh.midthickness.srf' in 'static_directory'.
	rh_srf_path : str, optional
		Path to the right hemisphere surface file. Default is 'rh.midthickness.srf' in 'static_directory'.
	perform_surface_smoothing : bool, optional
		Whether to apply surface smoothing before saving the PLY file. Default is True.
	n_smoothing_iterations : int, optional
		Number of iterations for surface smoothing if enabled. Default is 100.
	background_color_rbga : list, optional
		Background color in RGBA format for vertices that do not meet the threshold. Default is [220, 210, 195, 255].
	positive_cmap : str, optional
		Colormap name for positive scalar values. Default is 'red-yellow'.
	negative_cmap : str, optional
		Colormap name for negative scalar values. Default is 'blue-lightblue'.

	Returns
	-------
	None
		The function saves the cortical surface results as PLY files for the left and right hemispheres.

	Notes
	-----
	- The function assumes ImageObjectMRI.mask_data_ contains two elements: one for each hemisphere.
	- Surface smoothing can be performed using a vectorized smoothing function.
	- Uses _vertex_paint to apply colormap-based coloring to the vertices.
	- Output files are saved as '{outname}.lh.ply' and '{outname}.rh.ply'.
	"""
	assert len(ImageObjectMRI.mask_data_)==2, "Error: ImageObjectMRI must be a surface (i.e., from CorticalSurfaceImage)"
	v_lh, f_lh = nib.freesurfer.read_geometry(lh_srf_path)
	if perform_surface_smoothing:
		v_lh, f_lh = vectorized_surface_smoothing(v_lh, f_lh, number_of_iter = n_smoothing_iterations, lambda_w = 0.5, use_taubin = False, weighted = True)
	v_rh, f_rh = nib.freesurfer.read_geometry(rh_srf_path)
	if perform_surface_smoothing:
		v_rh, f_rh = vectorized_surface_smoothing(v_rh, f_rh, number_of_iter = n_smoothing_iterations, lambda_w = 0.5, use_taubin = False, weighted = True)
	
	color_arr = _vertex_paint(positive_scalar = positive_scalar_array,
										negative_scalar = negative_scalar_array,
										vmin = vmin, vmax = vmax,
										background_color_rbga = background_color_rbga,
										positive_cmap = positive_cmap,
										negative_cmap = negative_cmap)
	# left hemisphere
	outdata = np.ones((len(ImageObjectMRI.mask_data_[0]),4), int) * 255 # write background 
	outdata[ImageObjectMRI.mask_data_[0] == 1] = color_arr[:np.sum(ImageObjectMRI.mask_data_[0] == 1)]
	save_ply(v_lh, f_lh, outname[:-4] + ".lh.ply", color_array=outdata, output_binary=True)
	# right hemisphere
	outdata = np.ones((len(ImageObjectMRI.mask_data_[1]),4), int) * 255
	outdata[ImageObjectMRI.mask_data_[1] == 1] = color_arr[np.sum(ImageObjectMRI.mask_data_[0] == 1):]
	save_ply(v_rh, f_rh, outname[:-4] + ".rh.ply", color_array=outdata, output_binary=True)

def mri_voxels_to_mesh(voxel_data, threshold=0.95, vmin=-1.0, vmax=1.0, clip = True, voxel_alpha=0.7, cmap_name='red-yellow', 
						surface_names=None, surface_alpha=0.2, mask_data = None, 
						mask_color='white', mask_alpha=0.2):
	"""
	Convert voxel-wise MRI statistics to a 3D mesh representation with brain surfaces and mask surface.
	
	Parameters:
	-----------
	voxel_data : numpy.ndarray
		3D array containing the voxel values
	threshold : float
		Minimum value for a voxel to be displayed (default: 0.95)
	voxel_alpha : float
		Transparency level for voxels between 0 and 1 (default: 0.7)
	cmap_name : str
		Name of the colormap to use (default: 'red-yellow')
	surface_names : list
		List of surface names to load and display
	surface_alpha : float
		Transparency level for brain surfaces between 0 and 1 (default: 0.2)
	mask_data : numpy.ndarray
		3D array containing the mask (non-zero values indicate the mask)
	mask_color : str or tuple
		Color for the mask surface (default: 'white')
	mask_alpha : float
		Transparency for the mask surface (default: 0.2)
	
	Returns:
	--------
	plotter : pyvista.Plotter
		PyVista plotter object with the mesh visualization
	"""

	cmap_positive = ListedColormap(get_cmap_array(cmap_name)/255)
	
	# Create a PyVista plotter
	plotter = pv.Plotter()
	
	x_indices, y_indices, z_indices = np.where(voxel_data > threshold)
	#clip the top and bottom values for normalization
	if clip:
		values = np.clip(voxel_data[x_indices, y_indices, z_indices], vmin, vmax)
	else:
		values = voxel_data[x_indices, y_indices, z_indices]
	# Normalize values for coloring
	if len(values) > 0:
		norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
		for i in range(len(x_indices)):
			x, y, z = x_indices[i], y_indices[i], z_indices[i]
			val = norm_values[i]
			cube = pv.Cube(center=(x, y, z), x_length=1, y_length=1, z_length=1)
			color = cmap_positive(val)
			plotter.add_mesh(cube, color=color[:3], opacity=voxel_alpha)
	
	# Add brain surfaces if provided
	if surface_names:
		for sn in surface_names:
			vertices, faces = load_surface_geometry(sn)
			faces_pv = np.column_stack((np.full(len(faces), 3), faces)).ravel()
			mesh = pv.PolyData(vertices, faces_pv)
			plotter.add_mesh(mesh, color='lightgray', opacity=surface_alpha, 
							smooth_shading=True)
	
	# Add mask as a surface if provided
	if mask_data is not None:
		binary_mask = (mask_data > 0).astype(np.int8)
		try:
			verts, faces, normals, values = marching_cubes(binary_mask, level=0.5)
			mask_mesh = pv.PolyData(verts, np.column_stack(
				(np.full(len(faces), 3), faces)).ravel())
			plotter.add_mesh(mask_mesh, color=mask_color, opacity=mask_alpha, 
							 smooth_shading=True)
			
		except Exception as e:
			print(f"Error creating mask surface: {e}")
			print("Falling back to simple mask outline...")
			kernel = np.ones((3, 3, 3), dtype=np.int8)
			kernel[1, 1, 1] = 0  
			neighbor_count = convolve(binary_mask, kernel, mode='constant', cval=0)
			boundary_mask = (binary_mask > 0) & (neighbor_count < 26)
			bx, by, bz = np.where(boundary_mask)
			points = np.column_stack((bx, by, bz))
			if len(points) > 0:
				boundary_point_cloud = pv.PolyData(points)
				plotter.add_mesh(boundary_point_cloud, color=mask_color, 
								point_size=5, render_points_as_spheres=True)
	return(plotter)

def interactive_surface_viewer(path_to_surface, positive_scalar_array, negative_scalar_array = None, scalar_mask = None, vmin = 0.95, vmax = 1.0, perform_surface_smoothing = True, n_smoothing_iterations = 100, background_color_rbga=[220, 210, 195, 255], positive_cmap='red-yellow', negative_cmap='blue-lightblue', plot_render = False):

	"""
	Visualize surface data with interactive 3D rendering using scalar arrays for color mapping.

	Loads a surface mesh, optionally smooths it, and visualizes positive/negative scalar arrays
	using specified colormaps. Supports masking of scalar application and background customization.

	Parameters
	----------
	path_to_surface : str
		Path to surface geometry file. The following are acceptable formats:
			- FreeSurfer (.srf)
			- GIFTI (.surf.gii)
			- CIFTI (.d*.nii)
			- VTK (.vtk)
	positive_scalar_array : array-like
		1D array of positive scalar values for contrast visualization.
	negative_scalar_array : array-like, optional
		1D array of negative scalar values for contrast visualization.
	scalar_mask : array-like, optional
		Boolean mask specifying which vertices to apply colors to (1=apply, 0=background).
	vmin : float, optional
		Minimum value for scalar normalization (default: 0.95).
	vmax : float, optional
		Maximum value for scalar normalization (default: 1.0).
	perform_surface_smoothing : bool, optional
		Enable surface smoothing using Laplacian smoothing (default: True).
	n_smoothing_iterations : int, optional
		Number of smoothing iterations (default: 100).
	background_color_rbga : list, optional
		Background color as [R, G, B, A] (0-255) (default: [220, 210, 195, 255]).
	positive_cmap : str, optional
		Colormap name for positive values (default: 'red-yellow').
	negative_cmap : str, optional
		Colormap name for negative values (default: 'blue-lightblue').
	plot_render : bool, optional
		Immediately display the plot (default: False).

	Returns
	-------
	None

	Notes
	-----
	- Surface smoothing uses lambda_w=0.5 with weighted Laplacian smoothing
	- Face arrays are converted to PyVista's triangular mesh format (n_faces, 3, vertex_indices)

	Example
	-------
	>>> interactive_surface_viewer(path_to_surface = os.path.join(static_directory, 'lh.midthickness.srf'),
								positive_scalar_array = 1minusP_scalar,
								scalar_mask = ImageObjectMRI.mask_data_[0],
								vmin=0.95, vmax=1.0,
								positive_cmap='red-yellow',
								plot_render=True)
	"""


	v, f = load_surface_geometry(path_to_surface)
	if perform_surface_smoothing:
		v, f = vectorized_surface_smoothing(v, f, number_of_iter = n_smoothing_iterations, lambda_w = 0.5, use_taubin = False, weighted = True)
	f_new = np.column_stack((np.ones((len(f)))*3, f)).astype(int)
	mesh = pv.PolyData(v, f_new)
	if scalar_mask is None:
		scalar_mask = np.ones((len(v)))
	outdata = np.ones((len(scalar_mask),4), int) * 255
	color_arr = _vertex_paint(positive_scalar = positive_scalar_array,
										negative_scalar = negative_scalar_array,
										vmin = vmin, vmax = vmax,
										background_color_rbga = background_color_rbga,
										positive_cmap = positive_cmap,
										negative_cmap = negative_cmap)
	outdata[scalar_mask == 1] = color_arr
	mesh.point_data["RGB"] = outdata
	plotter = pv.Plotter()
	plotter.add_mesh(mesh, scalars='RGB', rgb=True, show_scalar_bar=False)
	if plot_render:
		plotter.show()
		plotter.close()

def generate_orthographic_snapshot_ply(path_to_ply, output_base, output_filetype='.svg'):
	"""
	Generates and saves orthogonal view plots of a 3D mesh from a PLY file.

	Reads a PLY file containing a 3D mesh with RGB colors and saves six different
	orthographic projections (axial superior, axial inferior, coronal posterior,
	coronal anterior, sagittal right, sagittal left) as vector graphic files.

	Parameters:
		path_to_ply : str
			Path to the input PLY file. (vtk is probably fine too as long as you have RGB data)
		output_base : str
			Base path for output files. If it ends with the
			output_filetype extension, the extension is removed before appending
			view-specific suffixes and the output_filetype.
		output_filetype : str, optional
			File format extension for saved images.
			Must include the leading dot (e.g., '.svg', '.png'). Defaults to '.svg'.

	Raises:
		FileNotFoundError: If the input PLY file does not exist.
		ValueError: If the mesh cannot be read or lacks RGB data.

	Notes:
		The output images are saved with suffixes indicating the view direction and 
		anatomical orientation.
	"""
	if not output_filetype.startswith('.'):
		output_filetype = '.' + output_filetype
	if output_base.endswith(output_filetype):
		output_base = output_base.split(output_filetype)[0]

	mesh = pv.read(path_to_ply)

	plotter = pv.Plotter(off_screen=True, window_size=(2000, 2000))
	plotter.add_mesh(
		mesh,
		scalars='RGB',
		rgb=True,
		show_scalar_bar=False,
	)

	# Axial views (superior and inferior)
	plotter.view_xy(negative=False, render=False)
	plotter.reset_camera()
	plotter.save_graphic(output_base + "_axial_superior" + output_filetype)
	plotter.view_xy(negative=True, render=False)
	plotter.reset_camera()
	plotter.save_graphic(output_base + "_axial_inferior" + output_filetype)

	# Coronal views (posterior and anterior)
	plotter.view_xz(negative=False, render=False)
	plotter.reset_camera()
	plotter.save_graphic(output_base + "_coronal_posterior" + output_filetype)
	plotter.view_xz(negative=True, render=False)
	plotter.reset_camera()
	plotter.save_graphic(output_base + "_coronal_anterior" + output_filetype)

	# Sagittal views (right and left)
	plotter.view_yz(negative=False, render=False)
	plotter.reset_camera()
	plotter.save_graphic(output_base + "_sagittal_right" + output_filetype)
	plotter.view_yz(negative=True, render=False)
	plotter.reset_camera()
	plotter.save_graphic(output_base + "_sagittal_left" + output_filetype)
	plotter.close()

