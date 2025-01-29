#!/usr/bin/env python

import os
import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm
from tfcemediation.tfce import CreateAdjSet
from tfcemediation.cynumstats import cy_lin_lstsqr_mat, tval_fast, fast_se_of_slope, cy_lin_lstsqr_mat_residual
from joblib import Parallel, delayed, wrap_non_picklable_objects, dump, load


def create_adjac_voxel(data_mask, dirtype=26): # default is 26 directions
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
		if int(dirtype)==6:
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

def voxel_adjacency(binary_mask, connectivity_directions = 26):
	assert connectivity_directions==8 or connectivity_directions==26, "adjacency_directions must equal {8, 26}"
	assert binary_mask.ndim==3, "binary_mask must have ndim==3"
	assert binary_mask.max()==1, "binary_mask max value must be 1"
	return(create_adjac_voxel(binary_mask, dirtype=connectivity_directions))

def calculate_tfce(statistic_flat, adjacency_set, H = 2., E = 0.67):
	calcTFCE = CreateAdjSet(H, E, adjacency_set) # 18.7 ms; approximately 180s on 10k permutations => acceptable for voxel
	stat = statistic_flat.astype(np.float32, order = "C")
	stat_TFCE = np.zeros_like(stat).astype(np.float32, order = "C")
	calcTFCE.run(stat, stat_TFCE)
	return(stat_TFCE)

def paralle_tfce(i, statistic_flat, adjacency_set, H = 2., E = 1., seed = None):
	if seed is None:
		np.random.seed(np.random.randint(4294967295))
	else:
		np.random.seed(seed)
	statistic_img_TFCE = calculate_tfce(statistic_flat, adjacency_set, H = H, E = E)
	return(np.max(statistic_img_TFCE))

def generate_seeds(n_seeds, maxint = int(2**32 - 1)):
	return([np.random.randint(0, maxint) for i in range(n_seeds)])


seeds = generate_seeds(1000)



seed = None

tfce_values = np.array(Parallel(n_jobs = 16, backend='multiprocessing')(
										delayed(paralle_tfce)(i, statistic_flat = np.random.uniform(0.1, 1, 94223), # tfce needs to be included with the permutation
												adjacency_set = adjacency_set,
												H = 2., E = 0.67,
												seed = seeds[i]) for i in tqdm(range(200))))

class VoxelImage:
	"""
	Input of voxel images, their mask, and calculation of adjacency
		"""
		def __init__(self, *, binary_mask_path, images_path):
		"""
		Initializes the VoxelImage class by loading the binary mask and voxel images.
		
		Args:
			binary_mask_path (str): Path to the binary mask file (must contain only 0s and 1s).
			images_path (str or list): Path(s) to the image file(s).
		
		Raises:
			AssertionError: If the mask is not binary.
			AssertionError: If image dimensions or affine transformations do not match the mask.
			TypeError: If `images_path` is neither a string nor a list.
		"""
		# Load the mask
		mask = nib.load(binary_mask_path)
		mask_data = mask.get_fdata()
		assert np.all(np.unique(mask_data)==np.array([0,1])), "binary_mask_path must be a binary image containing only {1,0}."
		self.affine_ = mask.affine
		self.mask_data_ = mask_data
		self.n_voxels_ = int(mask_data.sum())
		
		# Load the images
		self.images_path_ = images_path
		if isinstance(images_path, str):
			img = nib.load(images_path)
			img_data = img.get_fdata()
			assert img_data.ndim == 4, "image must be ndim==4 if images_path is a str"
			assert np.all(self.affine_ == img_data.affine), "The affines of the mask and images must be equal"
			self.image_data_ = img_data[self.mask_data_==1].astype(np.float32, order = "C")
			self.n_images_ = self.image_data_.shape[1]
		elif isinstance(images_path, list):
			self.n_images_ = len(images_path)
			self.image_data_ = np.zeros((self.n_voxels_, self.n_images_), ).astype(np.float32, order = "C")
			for s, path in enumerate(images_path):
				img_temp = nib.load(images_path)
				assert np.all(self.affine_ == img_temp.affine), "The affines of the mask and image [%s] must be equal" % os.path.basename(path)
				self.image_data_[:,s] = img_temp.get_fdata()[self.mask_data_==1]
		else:
			raise TypeError("images_path has to be a string or list of strings")
	def generate_adjacency(self, connectivity_directions = 26):
		"""
		Generates the adjacency set for the voxel image based on connectivity.
		
		Args:
			connectivity_directions (int, optional): Number of connectivity directions (default is 26).
		"""
		assert connectivity_directions==8 or connectivity_directions==26, "adjacency_directions must equal {8, 26}"
		self.adjacency_ = create_adjac_voxel(binary_mask, dirtype=connectivity_directions)

	def save(self, filename):
	"""
	Saves a VoxelImage instance as a pickle file.
	
	Args:
		filename (str): The name of the pickle file to be saved.
	Raises:
		AssertionError: If the filename does not end with .pkl.
	"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
	@classmethod
	def load(cls, filename):
	"""
	Loads a VoxelImage instance from a pickle file.
	
	Args:
		filename (str): The name of the pickle file to load.
	Returns:
		VoxelImage: The loaded instance.
	Raises:
		AssertionError: If the filename does not end with .pkl.
	"""
		assert filename.endswith(".pkl"), "filename must end with extension *.pkl"
		with open(filename, 'rb') as f:
			return pickle.load(f)


class LinearRegressionModelMRI:
	"""
	Linear Regression Model
	"""
	def __init__(self, *, fit_intercept = True, n_jobs = None, memory_mapping = False, use_tmp = True, fdr_correction = False, adjacency_set = None, image_mask = None):
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
		self.fdr_correction = fdr_correction
		self.adjacency_set = adjacency_set
		self.image_mask = image_mask

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
			Reshaped if necessary.
		"""
		X = np.array(X)
		y = np.array(y)
		assert len(X) == len(y), "X and y have different lengths"
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
		
	def fit(self, X, y):
		"""
		Fit the linear regression model to the data.

		Parameters
		----------
		X : np.ndarray, shape(n_samples, n_features)
			Exogneous varialbes
		
		y : np.ndarray or str, shape(n_samples, n_dependent_variables) or 'mapped'
			Endogenous variables

		Returns
		-------
		self : object
			Fitted model instance.
		"""
		
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
	
	def calculate_tstatistics(self, calculate_probability = True):
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
		if calculate_probability:
			self.t_pvalues_ = tdist.sf(np.abs(self.t_), self.df_total_) * 2
			self.t_qvalues_ = np.ones((self.t_.shape))
			for c in range(self.t_pvalues_.shape[0]):
				if self.fdr_correction:
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
			f.sf(sim_Fmodel, DF_Between, DF_Within)
			self.f_pvalues_ = fdist.sf(self.f_ , self.df_between_, self.df_within_)
			if self.fdr_correction:
				self.f_qvalues_ = fdrcorrection(self.f_pvalues_)[1]
		return(self)

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




mask_img = nib.load('/mnt/raid1/projects/tris/CRHR1_PROJECT/ENVIRONMENTAL_12SEP2024/MRI_ANALYSIS/mean_FA_skeleton_mask.nii.gz')
binary_mask = mask_img.get_fdata()

adjacency = voxel_adjacency(binary_mask)



statistic_image

calcTFCE.run()


		voxelStat_out = voxelStat.astype(np.float32, order = "C")
		voxelStat_TFCE = np.zeros_like(voxelStat_out).astype(np.float32, order = "C")
		TFCEfunc.run(voxelStat_out, voxelStat_TFCE)
		
		
