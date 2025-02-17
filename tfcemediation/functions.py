#!/usr/bin/env python

import os
import sys
import struct
import warnings
import pickle
import gzip
import shutil
import gc
import time
import re

import nibabel as nib
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed, wrap_non_picklable_objects, dump
from joblib import load as jload
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import t as tdist, f as fdist
from scipy.stats import norm
from scipy.special import erf
from tfcemediation.tfce import CreateAdjSet
from tfcemediation.cynumstats import cy_lin_lstsqr_mat, fast_se_of_slope
from patsy import dmatrix
from scipy.ndimage import label as scipy_label
from scipy.ndimage import generate_binary_structure

import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colorbar import ColorbarBase

# get static resources
scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
static_directory = os.path.join(scriptwd, "tfcemediation", "static")
static_files = os.listdir(static_directory)

pack_directory = os.path.join(scriptwd, "tfcemediation", "static", "aseg-subcortical-Surf")
aseg_subcortical_files = np.sort(os.listdir(pack_directory))
pack_directory = os.path.join(scriptwd, "tfcemediation", "static", "JHU-ICBM-Surf")
jhu_white_matter_files = np.sort(os.listdir(pack_directory))

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


def get_precompiled_freesurfer_adjacency(spatial_smoothing = 3):
	"""
	Loads precomputed adjacency matrices from the midthickness FreeSurfer surface for left and right hemispheres.
	
	Parameters
	----------
	spatial_smoothing : int, optional
		The amount of spatial smoothing applied (default is 3).
	
	Returns
	-------
	tuple of np.ndarray
		adjacency_lh : numpy array
			Adjacency matrix for the left hemisphere.
		adjacency_rh : numpy array
			Adjacency matrix for the right hemisphere.
	"""
	adjacency_lh = np.load('%s/lh_adjacency_dist_%d.0_mm.npy' % (static_directory, spatial_smoothing), allow_pickle=True)
	adjacency_rh = np.load('%s/rh_adjacency_dist_%d.0_mm.npy' % (static_directory, spatial_smoothing), allow_pickle=True)
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
		For example, `adjacency[i]` contains the indices of vertices adjacent to vertex `i`.
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
		if adjacency_lh_path is None and adjacency_rh_path is None:
			adjacency = get_precompiled_freesurfer_adjacency(spatial_smoothing = 3)
		self.image_data_ = np.concatenate([data_lh, data_rh]).astype(np.float32, order = "C")
		self.affine_ = [affine_lh, affine_rh]
		self.header_ = [header_lh, header_rh]
		self.n_vertices_ = [n_vertices_lh, n_vertices_rh]
		self.mask_data_ = [bin_mask_lh, bin_mask_rh]
		self.adjacency_ = adjacency
		self.hemipheres_ = ['left-hemisphere', 'right-hemisphere']

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
			AssertionError: connectivity_directions is not 8 or 26.
		"""
		assert connectivity_directions==6 or connectivity_directions==26, "adjacency_directions must equal {8, 26}"
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
	Linear Regression Model
	"""
	def __init__(self, *, fit_intercept = True, n_jobs = 16, memory_mapping = False, use_tmp = True, fdr_correction = False):
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
		"""
		self.fit_intercept_ = fit_intercept
		self.n_jobs_ = n_jobs
		self.memory_mapping_ = memory_mapping
		self.use_tmp_ = use_tmp
		self.tmp_directory_ = os.environ.get("TMPDIR", "/tmp/")
		self.fdr_correction_ = fdr_correction

	def _check_inputs(self, X, y):
		"""
		Check and validate inputs for the model.

		Parameters
		----------
		X : np.ndarray
			Exogneous variables
			
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
		self.df_ = df

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
		self.df_ = pd.read_csv(csv_file)

	def dummy_code_from_formula(self, formula_like, save_columns_names = True, scale_dummy_arr = True):
		"""
		Creates dummy-coded variables using patsy from the DataFrame and optionally scales them
		
		Parameters
		----------
		formula_like : str
			The formula that specifies which columns to include in the dummy coding (e.g., 'category1 + category2')
		
		save_columns_names : bool, optional, default=True
			If True, stores the names of the dummy-coded columns in the object attribute `t_contrast_names_`
		
		scale_dummy_arr : bool, optional, default=True
			If True, scales the resulting dummy variables (excluding the intercept column)
		
		Returns
		---------
		np.ndarray
			The scaled dummy-coded variables as a numpy array, with the intercept column excluded
		"""
		assert hasattr(self, 'df_'), "Pandas dataframe is missing (self.df_) run load_pandas_dataframe or load_csv_dataframe first"
		df_dummy = dmatrix(formula_like, data=self.df_, NA_action="raise", return_type='dataframe')
		if save_columns_names:
			colnames =  df_dummy.columns.values
			self.t_contrast_names_ = np.array([sanitize_columns(col) for col in colnames])
		dummy_arr = scale_arr(df_dummy.values[:,1:])
		return(dummy_arr)

	def print_t_contrast_indices(self):
		"""
		Print the indices of t-contrasts.

		If the attribute `t_contrast_names_` exists, this function prints the index 
		and corresponding contrast name. Otherwise, it prints the numeric indices 
		for all available contrasts.

		Parameters:
		-----------
		self : object
			The instance containing `t_contrast_names_` and `t_` attributes.
		"""
		if hasattr(self, 't_contrast_names_'):
			for t in range(len(self.t_contrast_names_)):
				print("[index=%d] ==> %s" % (t, self.t_contrast_names_[t]))
		else:
			print(np.arange(self.t_.shape[0]))
	
	def fit(self, X, y):
		"""
		Fit the linear regression model to the data.

		Parameters
		----------
		X : np.ndarray, shape(n_samples, n_features)
			Exogneous variables
		
		y : np.ndarray or str, shape(n_samples, n_dependent_variables) or 'mapped'
			Endogenous variables

		Returns
		-------
		self : object
			Fitted model instance.
		"""
		if isinstance(y, str):
			if y == 'mapped':
				assert hasattr(self, 'memmap_y_name_'), "No memory mapped endogenous variables found"
				y = jload(self.memmap_y_name_, mmap_mode='r')
		X, y = self._check_inputs(X, y)
		if not hasattr(self, 'memmap_y_name_'): # confusing: checks for memory mapped y, 
			if self.memory_mapping_:
				data_filename_memmap = "memmap_y_%d" % int(time.time())
				if self.use_tmp_:
					data_filename_memmap = os.path.join(self.tmp_directory_, data_filename_memmap)
				else:
					data_filename_memmap = os.path.abspath(data_filename_memmap)
				self.memmap_y_name_ = data_filename_memmap
				dump(y, data_filename_memmap)
				y = jload(data_filename_memmap, mmap_mode='r')
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

	def _calculate_surface_tfce(self, mask_data, statistic, adjacency_set, H = 2.0, E = 0.67, return_max_tfce = False):
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

	def calculate_tstatistics_tfce(self, ImageObjectMRI, H = 2.0, E = 0.67, contrast = None):
		"""
		Computes Threshold-Free Cluster Enhancement (TFCE) enhanced t-statistics 
		for both positive and negative contrasts.

		This function applies the TFCE algorithm to enhance statistical maps 
		by accounting for spatial adjacency relationships, improving sensitivity 
		in neuroimaging analyses.

		Parameters
		----------
		ImageObjectMRI : object
			An instance containing neuroimaging data, including adjacency 
			information and mask data.
		H : float, optional
			The height exponent for TFCE computation (default is 2.0).
		E : float, optional
			The extent exponent for TFCE computation (default is 0.67).
		contrast : int, None
			Set which contrast to calculate TFCE. Other contrasts will be zero.
		Raises
		------
		AssertionError
			If the t-statistics have not been computed before running TFCE.
		
		Returns
		-------
		self : object
			The instance with updated attributes containing the computed 
			TFCE-enhanced t-statistics for both positive and negative contrasts.
		
		Notes
		-----
		- If 'ImageObjectMRI' has a 'hemispheres_' attribute, TFCE is computed 
		  using a surface-based approach.
		- Otherwise, a voxel-based TFCE computation is performed using adjacency sets.
		- The computed TFCE values are stored in 'self.t_tfce_positive_' and 
		  'self.t_tfce_negative_'.
		assert hasattr(self, 't_'), "Run calculate_tstatistics() first"
		adjacency_set = ImageObjectMRI.adjacency_
		"""
		assert hasattr(ImageObjectMRI, 'adjacency_'), "ImageObjectMRI is missing adjacency_"
		self.t_tfce_positive_ = np.zeros((self.t_.shape)).astype(np.float32, order = "C")
		self.t_tfce_negative_ = np.zeros((self.t_.shape)).astype(np.float32, order = "C")

		iterator_ = np.arange(0, self.t_.shape[0])
		if contrast is not None:
			iterator_ = [iterator_[contrast]]

		if hasattr(ImageObjectMRI, 'hemipheres_'):
			for c in iterator_:
				if np.sum(self.t_[c] > 0) < 100 or np.sum(self.t_[c] < 0) < 100:
					print("The t-statistic is in the same direction for almost all vertices. Skipping TFCE calculation for Contrast-%d" % (c))
				elif np.sum(np.abs(self.t_[c]) > 5) > int(self.t_[c].shape[0] * 0.90):
					print("abs(t-values)>5 detected for >90 percent of the vertices. Skipping TFCE calculation for Contrast-%d" % (c))
				else:
					tfce_values =  self._calculate_surface_tfce(mask_data = ImageObjectMRI.mask_data_,
																				statistic = self.t_[c].astype(np.float32, order = "C"),
																				adjacency_set = ImageObjectMRI.adjacency_,
																				H = H, E = E, return_max_tfce = False)
					self.t_tfce_positive_[c] = tfce_values[0]
					self.t_tfce_negative_[c] = tfce_values[1]
		else:
			calcTFCE = CreateAdjSet(H, E, ImageObjectMRI.adjacency_) # 18.7 ms; approximately 180s on 10k permutations => acceptable for voxel
			for c in iterator_:
				tval = self.t_[c]
				stat = tval.astype(np.float32, order = "C")
				stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
				calcTFCE.run(stat, stat_TFCE)
				self.t_tfce_positive_[c] = stat_TFCE
				stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
				calcTFCE.run(-stat, stat_TFCE)
				self.t_tfce_negative_[c] = stat_TFCE
		# for permutation testing
		self.adjacency_set_ = ImageObjectMRI.adjacency_
		self.mask_data_ = ImageObjectMRI.mask_data_
		self.tfce_H_ = float(H)
		self.tfce_E_ = float(E)
		return(self)

	def _run_tfce_t_permutation(self, i, X, y, contrast_index, H, E, adjacency_set, mask_data, seed):
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

	def permute_tstatistics_tfce(self, contrast_index, n_permutations, whiten = True, use_blocks = True, block_size = 384):
		"""
		Performs TFCE-based permutation testing for a given contrast index.
		
		This function computes t-statistic permutations and applies TFCE correction
		to obtain the maximum TFCE values across permutations.

		Parameters
		----------
		contrast_index : int
			The index of the contrast for permutation testing.
		n_permutations : int
			The number of permutations to perform.
		whiten : bool, optional
			Whether to whiten the residuals before permutation (default is True).
		"""
		assert hasattr(self, 'adjacency_set_'), "Run calculate_tstatistics_tfce first"
		if self.memory_mapping_:
			assert hasattr(self, 'memmap_y_name_'), "No memory mapped endogenous variables found"
			y = jload(self.memmap_y_name_, mmap_mode='r')
		else:
			y = self.y_
		if whiten:
			y = y - self.predict(self.X_)
		X = self.X_
		if use_blocks:
			tfce_maximum_values = []
			if not n_permutations % block_size == 0:
				res = n_permutations % block_size
				n_permutations += (block_size - res)
			print("Running %d permutations [p<0.0500 +/- %1.4f]" % (n_permutations,(2*np.sqrt(0.05*(1-0.05)/n_permutations))))
			n_blocks = int(n_permutations/block_size)
			for b in range(n_blocks):
				print("Block[%d/%d]: %d Permutations" % (int(b+1), n_blocks, block_size))
				seeds = generate_seeds(n_seeds = int(block_size/2))
				block_tfce_maximum_values = Parallel(n_jobs = self.n_jobs_, backend='multiprocessing')(
														delayed(self._run_tfce_t_permutation)(i = i, 
																						X = X,
																						y = y, 
																						contrast_index = contrast_index,
																						H = self.tfce_H_,
																						E = self.tfce_E_,
																						adjacency_set = self.adjacency_set_,
																						mask_data = self.mask_data_,
																						seed = seeds[i]) for i in tqdm(range(int(block_size/2))))
				tfce_maximum_values.append(block_tfce_maximum_values)
			tfce_maximum_values = np.array(tfce_maximum_values).ravel()
		else:
			seeds = generate_seeds(n_seeds = int(n_permutations/2))
			print("Running %d permutations [p<0.0500 +/- %1.4f]" % (n_permutations,(2*np.sqrt(0.05*(1-0.05)/n_permutations))))
			seeds = generate_seeds(n_seeds = int(n_permutations/2))
			tfce_maximum_values = Parallel(n_jobs = self.n_jobs_, backend='multiprocessing')(
													delayed(self._run_tfce_t_permutation)(i = i, 
																					X = X,
																					y = y, 
																					contrast_index = contrast_index,
																					H = self.tfce_H_,
																					E = self.tfce_E_,
																					adjacency_set = self.adjacency_set_,
																					mask_data = self.mask_data_,
																					seed = seeds[i]) for i in tqdm(range(int(n_permutations/2))))
			tfce_maximum_values = np.array(tfce_maximum_values).ravel()
		self.t_tfce_max_permutations_ = tfce_maximum_values

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

	def calculate_mediation_z_from_formula(self, mri_data, X = None, M = None, y = None, covariates = None, calculate_probability = True):
		"""
		Perform Sobel mediation analysis using the provided formulas.

		Parameters
		----------
		mri_data : np.ndarray, shape (n_samples,)
			MRI data.
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
		assert hasattr(self, 'df_'), "Pandas dataframe is missing (self.df_) run load_pandas_dataframe or load_csv_dataframe first"
		not_none_count = sum(val is not None for val in (X, M, y))
		assert not_none_count == 2, "Two of X, M, and y must not be None"
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

	def calculate_mediation_z_tfce(self, adjacency_set, H = 2., E = 0.67):
		"""
		Computes TFCE-enhanced t-statistics for both positive and negative contrasts.
		
		This function applies the TFCE algorithm to enhance statistical maps using adjacency sets.
		
		Parameters
		----------
		adjacency_set : list
			A set defining the adjacency relationships between data points.
		H : float, optional
			The height exponent for TFCE computation (default is 2.0).
		E : float, optional
			The extent exponent for TFCE computation (default is 0.67).
		
		Raises
		------
		AssertionError
			If the t-statistics have not been computed before running TFCE.
		"""
		assert hasattr(self, 'mediation_z_'), "Run calculate_tstatistics() first"
		calcTFCE = CreateAdjSet(H, E, adjacency_set) # 18.7 ms; approximately 180s on 10k permutations => acceptable for voxel
		zval = self.mediation_z_.astype(np.float32, order = "C")
		stat = zval.astype(np.float32, order = "C")
		stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
		calcTFCE.run(stat, stat_TFCE)
		self.mediation_z_tfce_ = stat_TFCE
		self.adjacency_set_ = adjacency_set
		self.tfce_H_ = float(H)
		self.tfce_E_ = float(E)
		return(self)

	def _run_tfce_mediation_z_permutation(self, i, exogA, endogA, exogB, endogB, H, E, adjacency_set, seed):
		"""
		Perform a single TFCE-based permutation test for mediation analysis.

		This method shuffles the data, calculates Sobel z-scores, and applies the TFCE (Threshold-Free Cluster Enhancement) algorithm to 
		assess the statistical significance of the mediation effect under the permutation scheme. The function computes the maximum TFCE 
		value for the permuted z-statistic, providing insight into the robustness of the mediation effect.

		Parameters
		----------
		i : int
			The permutation index. This parameter is required for parallel processing but is not used directly within the function.
		exogA : np.ndarray, shape (n_samples, n_features)
			Exogenous variables for the first stage in the mediation analysis.
		endogA : np.ndarray, shape (n_samples,)
			Endogenous variable for the first stage in the mediation analysis.
		exogB : np.ndarray, shape (n_samples, n_features)
			Exogenous variables for the second stage in the mediation analysis.
		endogB : np.ndarray, shape (n_samples,)
			Endogenous variable for the second stage in the mediation analysis.
		H : float
			The height exponent used in the TFCE computation. Controls the sensitivity to large values in the statistic.
		E : float
			The extent exponent used in the TFCE computation. Controls the sensitivity to the spatial extent of clusters.
		adjacency_set : list
			A list defining adjacency relationships between data points, typically used for establishing neighborhood connections in the TFCE algorithm.
		seed : int or None
			The random seed for permutation. If `None`, a random seed will be selected.

		Returns
		-------
		float
			The maximum TFCE value calculated for the permuted z-statistic (mediation effect) during the permutation test.
		"""
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		perm_index = np.random.permutation(range(len(exogA)))
		tmp_z = self._calculate_sobel(exogA[perm_index], endogA, exogB[perm_index], endogB)

		# Compute TFCE
		perm_calcTFCE = CreateAdjSet(H, E, adjacency_set)
		stat = tmp_z.astype(np.float32, order = "C")
		stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
		perm_calcTFCE.run(stat, stat_TFCE)
		max_tfce = stat_TFCE.max()

		# Garbage collections
		del stat_TFCE, stat, perm_calcTFCE, tmp_z
		gc.collect()
		return(max_tfce)

	def permute_mediation_z_tfce(self, n_permutations):
		"""
		Perform TFCE-based permutation testing for a given contrast index.

		This method runs a series of permutations, computes Sobel z-scores, and applies the TFCE 
		(Threshold-Free Cluster Enhancement) correction to obtain the maximum TFCE values across permutations.

		Parameters
		----------
		n_permutations : int
			The number of permutations to perform in the permutation testing process.

		Returns
		-------
		None
			The function updates the `mediation_z_tfce_max_permutations_` attribute with the computed
			maximum TFCE values across all permutations.
		"""
		assert hasattr(self, 'adjacency_set_'), "Run calculate_tstatistics_tfce first"
		print("Running %d permutations [p<0.0500 +/- %1.4f]" % (n_permutations,(2*np.sqrt(0.05*(1-0.05)/n_permutations))))
		seeds = generate_seeds(n_seeds = n_permutations)
		tfce_maximum_values = Parallel(n_jobs = self.n_jobs_, backend='multiprocessing')(
												delayed(self._run_tfce_mediation_z_permutation)(i = i, 
																				exogA = self.mediation_exogA_,
																				endogA = self.mediation_endogA_,
																				exogB = self.mediation_exogB_,
																				endogB = self.mediation_endogB_,
																				H = self.tfce_H_,
																				E = self.tfce_E_,
																				adjacency_set = self.adjacency_set_,
																				seed = seeds[i]) for i in tqdm(range(n_permutations)))
		self.mediation_z_tfce_max_permutations_ = np.array(tfce_maximum_values)

	def outlier_detection(self, f_quantile = 0.99, cooks_distance_threshold = None, low_ram = True):
		"""
		Detect outliers using Cook's distance. Cook's distance is defined as the coefficient vector would move 
		if the sample were removed and the model refit.

		Parameters
		----------
		f_quantile : float
			The threshold for identifying outliers using the F-distribution.
		cooks_distance_threshold : float
			Manually, set the threshold for outliers.
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
		if cooks_distance_threshold is None:
			self.cooks_distance_threshold_ = fdist.ppf(f_quantile, self.df_between_, self.df_within_, loc=self.cooks_distance_.mean(0), scale=self.cooks_distance_.std(0))
		else:
			self.cooks_distance_threshold_ = cooks_distance_threshold
		self.n_outliers_ = (self.cooks_distance_ > self.cooks_distance_threshold_).sum(0)
		self.n_outliers_percentage_ = np.divide(self.n_outliers_ * 100, self.n_)
		if low_ram:
			del self.residuals_studentized_ 
			del self.cooks_distance_
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

	def write_t_tfce_results(self, ImageObjectMRI, contrast_index):
		"""
		Writes the Threshold-Free Cluster Enhancement (TFCE) results for a given contrast index.
		
		This function saves multiple NIfTI or mgh scalar images containing the t-values, positive and negative
		TFCE values, and their respective corrected p-values.

		Parameters
		----------
		contrast_index : int
			The index of the contrast for which TFCE results will be written.
		data_mask : numpy.ndarray
			A binary mask indicating valid data points in the brain image.
		affine : numpy.ndarray
			The affine transformation matrix for the NIfTI images.

		Raises
		------
		AssertionError
			If the required TFCE permutations have not been computed.
		"""
		assert hasattr(self, 't_tfce_max_permutations_'), "Run permute_tstatistics_tfce first"
		if hasattr(self, 't_contrast_names_') and self.t_.shape[0] == len(self.t_contrast_names_):
			contrast_name = "tvalue-%s" % self.t_contrast_names_[int(contrast_index)]
		else:
			contrast_name = "tvalue-con%d" % np.arange(0, len(self.t_),1)[int(contrast_index)]
		data_mask = ImageObjectMRI.mask_data_
		affine = ImageObjectMRI.affine_
		values = self.t_[contrast_index]

		if len(data_mask) == 2:
			self.write_freesurfer_image(values = values, data_mask = data_mask, affine = affine, outname = contrast_name + ".mgh")
			values = self.t_tfce_positive_[contrast_index]
			self.write_freesurfer_image(values = values, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_positive.mgh")
			oneminuspfwe = 1 - self._calculate_permuted_pvalue(self.t_tfce_max_permutations_,values)
			self.write_freesurfer_image(values = oneminuspfwe, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_positive-1minusp.mgh")

			values = self.t_tfce_negative_[contrast_index]
			self.write_freesurfer_image(values = values, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_negative.mgh")
			oneminuspfwe = 1 - self._calculate_permuted_pvalue(self.t_tfce_max_permutations_, values)
			self.write_freesurfer_image(values = oneminuspfwe, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_negative-1minusp.mgh")
		else:
			self.write_nibabel_image(values = values, data_mask = data_mask, affine = affine, outname = contrast_name + ".nii.gz")
			values = self.t_tfce_positive_[contrast_index]
			self.write_nibabel_image(values = values, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_positive.nii.gz")
			oneminuspfwe = 1 - self._calculate_permuted_pvalue(self.t_tfce_max_permutations_,values)
			self.write_nibabel_image(values = oneminuspfwe, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_positive-1minusp.nii.gz")

			values = self.t_tfce_negative_[contrast_index]
			self.write_nibabel_image(values = values, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_negative.nii.gz")
			oneminuspfwe = 1 - self._calculate_permuted_pvalue(self.t_tfce_max_permutations_, values)
			self.write_nibabel_image(values = oneminuspfwe, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce_negative-1minusp.nii.gz")

	def write_mediation_z_tfce_results(self, data_mask, affine):
		"""
		Write the Threshold-Free Cluster Enhancement (TFCE) results for the Sobel Z-score.

		This function saves multiple NIfTI images containing the Z-values, TFCE values, and 
		their respective FWER corrected p-values. The results are saved as NIfTI images with file names
		indicating the type of analysis (Z-values, TFCE, and TFCE p-values).

		Parameters
		----------
		data_mask : numpy.ndarray
			A binary mask indicating valid data points in the brain image. The mask is used to 
			ensure that only the valid regions of the image are considered for saving the results.
		
		affine : numpy.ndarray
			The affine transformation matrix for the NIfTI images, which is necessary for the 
			proper spatial alignment of the data when writing to NIfTI format.

		Raises
		------
		AssertionError
			If the required TFCE permutations have not been computed before calling this method.
		
		Notes
		-----
		This function assumes that the permutation-based TFCE analysis has been run, and the
		resulting data is available in `mediation_z_tfce_max_permutations_` and `mediation_z_tfce_`.
		The function also calls `write_nibabel_image` to save the generated images.
		"""
		assert hasattr(self, 'mediation_z_tfce_max_permutations_'), "Run permute_mediation_z_tfce first"
		contrast_name = "mediation_z"
		self.write_nibabel_image(values = self.mediation_z_, data_mask = data_mask, affine = affine, outname = contrast_name + ".nii.gz")
		self.write_nibabel_image(values = self.mediation_z_tfce_, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce.nii.gz")
		oneminuspfwe = 1 - self._calculate_permuted_pvalue(self.mediation_z_tfce_max_permutations_,self.mediation_z_tfce_)
		self.write_nibabel_image(values = oneminuspfwe, data_mask = data_mask, affine = affine, outname = contrast_name + "-tfce-1minusp.nii.gz")

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
		- The function assumes that `data_mask[0]` corresponds to the left hemisphere 
		  and `data_mask[1]` corresponds to the right hemisphere.
		- The `values` array should contain concatenated values for both hemispheres.
		- The output files will be saved as "<outname>.lh.mgh" and "<outname>.rh.mgh".
		"""
		if outname.endswith(".mgh"):
			midpoint = data_mask[0].sum()
			outdata = np.zeros((data_mask[0].shape[0]))
			outdata[data_mask[0]==1] = values[:midpoint]
			nib.save(nib.freesurfer.mghformat.MGHImage(outdata[:,np.newaxis, np.newaxis].astype(np.float32), affine[0]), outname[:-4] + ".lh.mgh")
			outdata.fill(0)
			outdata[data_mask[1]==1] = values[midpoint:]
			nib.save(nib.freesurfer.mghformat.MGHImage(outdata[:,np.newaxis, np.newaxis].astype(np.float32), affine[1]), outname[:-4] + ".rh.mgh")


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
			nib.save(nib.Nifti1Image(outdata, affine), outname)

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
		If not provided, the color map transitions directly from `c0` to `c1`.

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
		If not provided, the color map transitions directly from `c0` to `c1`.

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
		If not provided, the color map transitions directly from `c0` to `c1`.

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
		If `return_array` is True, returns a 2D array of shape (256, 4) representing the RGBA values of the colormap.
		If `return_array` is False, returns a matplotlib colormap object.
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
		If `return_array` is True, returns a 2D array of shape (256, 4) representing the RGBA values of the colormap.
		If `return_array` is False, returns a matplotlib colormap object.
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
		If `return_array` is True, returns a 2D array of shape (256, 4) representing the RGBA values of the colormap.
		If `return_array` is False, returns a matplotlib colormap object.
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



def vectorized_surface_smoothing(v, f, number_of_iter = 20, scalar = None, lambda_w = 0.5, use_taubin = False, weighted = True):
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


#def _vertex_paint(positive_scalar, negative_scalar = None, vmin = 0.95, vmax = 1.0, background_color_rbga = [220, 210, 195, 255], positive_cmap = 'red-yellow', negative_cmap = 'blue-lightblue'):


#	positive_scalar = 1 - model._calculate_permuted_pvalue(model.t_tfce_max_permutations_, model.t_tfce_positive_[-3])
#	negative_scalar = 1 - model._calculate_permuted_pvalue(model.t_tfce_max_permutations_, model.t_tfce_negative_[-3])
#	if negative_scalar is not None:
#		assert len(positive_scalar) == len(negative_scalar), "positive and negative scalar must have the same length"
#	pos_cmap_arr = get_cmap_array(positive_cmap)
#	neg_cmap_arr = get_cmap_array(negative_cmap)
#	
#	out_color_arr = np.ones((len(positive_scalar),4), int) * 255
#	out_color_arr[:] = background_color_rbga
#	if np.sum(positive_scalar > vmin) != 0:
#		cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#		cmap = ListedColormap(np.divide(pos_cmap_arr,255))
#		mask = positive_scalar > vmin
#		vals = np.round(cmap(positive_scalar[mask])*255).astype(int)
#		out_color_arr[mask] = vals
#	if np.sum(negative_scalar > vmin) != 0:
#		cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#		cmap = ListedColormap(np.divide(neg_cmap_arr,255))
#		mask = negative_scalar > vmin
#		vals = np.round(cmap(negative_scalar[mask])*255).astype(int)
#		out_color_arr[mask] = vals
#	outdata = np.ones((len(corticalthickness_fu3.mask_data_[0]),4), int) * 255
#	outdata[corticalthickness_fu3.mask_data_[0] == 1] = out_color_arr[:np.sum(corticalthickness_fu3.mask_data_[0] == 1)]
#	save_ply(vs, fs, "test_lh_stat_neg3con.ply", color_array=outdata, output_binary=True)

#	outdata = np.ones((len(corticalthickness_fu3.mask_data_[1]),4), int) * 255
#	outdata[corticalthickness_fu3.mask_data_[1] == 1] = out_color_arr[np.sum(corticalthickness_fu3.mask_data_[0] == 1):]
#	save_ply(vsr, fsr, "test_rh_stat_neg3con.ply", color_array=outdata, output_binary=True)




