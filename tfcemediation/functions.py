#!/usr/bin/env python

import os
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
from joblib import Parallel, delayed, wrap_non_picklable_objects, dump, load
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import t as tdist, f as fdist
from scipy.stats import norm
from tfcemediation.tfce import CreateAdjSet
from tfcemediation.cynumstats import cy_lin_lstsqr_mat, fast_se_of_slope
from patsy import dmatrix
from scipy.ndimage import label as scipy_label
from scipy.ndimage import generate_binary_structure

def generate_seeds(n_seeds, maxint = int(2**32 - 1)):
	"""
	Generates a list of random integer seeds.
	
	This function creates a list of `n_seeds` random integers within the range [0, maxint],
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
		A list of `n_seeds` randomly generated integers.
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

	def _create_adjac_voxel(self, data_mask, connectivity_directions=26):
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
		self.adjacency_ = self._create_adjac_voxel(self.mask_data_, connectivity_directions=connectivity_directions)

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
				y = load(self.memmap_y_name_, mmap_mode='r')
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
				y = load(data_filename_memmap, mmap_mode='r')
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

	def calculate_tstatistics_tfce(self, adjacency_set, H = 2., E = 0.67):
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
		assert hasattr(self, 't_'), "Run calculate_tstatistics() first"
		calcTFCE = CreateAdjSet(H, E, adjacency_set) # 18.7 ms; approximately 180s on 10k permutations => acceptable for voxel
		self.t_tfce_positive_ = np.zeros((self.t_.shape)).astype(np.float32, order = "C")
		self.t_tfce_negative_ = np.zeros((self.t_.shape)).astype(np.float32, order = "C")
		for c in range(self.t_.shape[0]):
			tval = self.t_[c]
			stat = tval.astype(np.float32, order = "C")
			stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
			calcTFCE.run(stat, stat_TFCE)
			self.t_tfce_positive_[c] = stat_TFCE
			stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
			calcTFCE.run(-stat, stat_TFCE)
			self.t_tfce_negative_[c] = stat_TFCE
		self.adjacency_set_ = adjacency_set
		self.tfce_H_ = float(H)
		self.tfce_E_ = float(E)
		return(self)

	def _run_tfce_t_permutation(self, i, X, y, contrast_index, H, E, adjacency_set, seed):
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
		
		# Compute TFCE
		perm_calcTFCE = CreateAdjSet(H, E, adjacency_set)
		tval = tmp_t[contrast_index]
		stat = tval.astype(np.float32, order = "C")
		
		stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
		perm_calcTFCE.run(stat, stat_TFCE)
		max_pos = stat_TFCE.max()
		
		# Garbage collections
		del stat_TFCE, tmp_invXX, tmp_sigma2, tmp_se, tmp_t
		gc.collect()
		
		stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
		perm_calcTFCE.run(-stat, stat_TFCE)
		max_neg = stat_TFCE.max()
		
		# Garbage collections 2 electric bungalow
		del stat_TFCE, perm_calcTFCE
		gc.collect()
		
		return(max_pos, max_neg)

	def permute_tstatistics_tfce(self, contrast_index, n_permutations, whiten = True):
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
		if whiten:
			y = self.y_ - self.predict(self.X_)
		else:
			y = self.y_.copy()
		X = self.X_.copy()
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
																				seed = seeds[i]) for i in tqdm(range(int(n_permutations/2))))
		tfce_maximum_values = np.concatenate(tfce_maximum_values)
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

	def write_t_tfce_results(self, contrast_index, data_mask, affine):
		"""
		Writes the Threshold-Free Cluster Enhancement (TFCE) results for a given contrast index.
		
		This function saves multiple NIfTI images containing the t-values, positive and negative
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
		values = self.t_[contrast_index]

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
