#!/usr/bin/env python

import os
import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm
from tfcemediation.tfce import CreateAdjSet
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

def voxel_adjacency(binary_mask, adjacency_directions = 26):
	assert adjacency_directions==8 or adjacency_directions==26, "adjacency_directions must equal {8, 26}"
	assert binary_mask.ndim==3, "binary_mask must have ndim==3"
	assert binary_mask.max()==1, "binary_mask max value must be 1"
	return(create_adjac_voxel(binary_mask, dirtype=adjacency_directions))

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




mask_img = nib.load('/mnt/raid1/projects/tris/CRHR1_PROJECT/ENVIRONMENTAL_12SEP2024/MRI_ANALYSIS/mean_FA_skeleton_mask.nii.gz')
binary_mask = mask_img.get_fdata()

adjacency = voxel_adjacency(binary_mask)



statistic_image

calcTFCE.run()


		voxelStat_out = voxelStat.astype(np.float32, order = "C")
		voxelStat_TFCE = np.zeros_like(voxelStat_out).astype(np.float32, order = "C")
		TFCEfunc.run(voxelStat_out, voxelStat_TFCE)
		
		
