import glob
import googlemaps
import json
import math as m
import numpy as np
import os
import pandas as pd
import pprint
import sys
import tsp

from matplotlib import pyplot as plt
from scipy import linalg as la
from utils import *

def main():

    hq_file = 'rothko_hq.csv'
    coords_file = 'random_coordinates.csv'

    num_gen_coords = 30
    num_sample_coords = 20

    regen_coords = False
    recompute_dist_mat = False

    if coords_file in [f for f in glob.glob("*.csv")]:
        os.remove(coords_file)

    df = coords_generation(num_gen_coords, hq_file, 
                      coords_file, regen_coords, 
                      recompute_dist_mat)

    df = df.drop(columns='Unnamed: 0')
    drive_time_mat = np.array(np.memmap('drive_time_mat.dat', 
                                         dtype='float64', mode='r', 
                                         shape=(num_gen_coords, num_gen_coords)))

    # sample a random submatrix of drive_time_mat
    # of size (num_coords x num_coords)

    # num_samples = 1

    # for i in range(num_samples):

    r_gen_coords = range(num_gen_coords)
    sample_idxs = list(np.random.choice(r_gen_coords, 
                                   num_sample_coords, 
                                   replace=False))

    sample_mat = get_submat_from_idx_list(drive_time_mat, sample_idxs)
    drop_idxs = [elem for elem in r_gen_coords if not elem in sample_idxs]
    df_sample = df.drop(drop_idxs, axis=0)
    df_sample = df_sample.reset_index()

    n_clstrs = 2
    K = 3
    cluster_method = 'spectral'

    clstr_idxs = coords_cluster(df_sample, sample_mat, 
                        cluster_method, n_clstrs, K)

    tsp = tsp_on_clusters(sample_mat, clstr_idxs)
        # tsp_solns.append(tsp)
        # max_clstr_time = np.max(tsp[1])
        # max_clstr_times.append(max_clstr_time)
        # hyperopt_min.append([K, n_clstrs, clstr_idxs])       

        # # loop over Kth nearest neighbor distances 
        # # for local statistics params; for hyperparameter 
        # # optimization over cluster number and K neighbors
        # # we want to minimize the required time of the longest cluster

        # # HYPERPARAMETER OPTIMIZATION LOOP(S):
        # # 1. nearest-neighbor scaling and 2. # of clusters


if __name__=="__main__": 

    main() 
