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

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.cluster import SpectralClustering, KMeans

gmaps_key = os.environ.get('GMAPS_KEY')
gmaps = googlemaps.Client(key=gmaps_key)

def get_submat_from_idx_list(arr, idx_list):
    return np.array([np.take(arr[idx], idx_list) 
                       for idx in idx_list])

def drive_time(origin, destination):
    dist_mat_info = gmaps.distance_matrix(origin, destination)
    return dist_mat_info['rows'][0]['elements'][0]['duration']['value']

def drive_dist(origin, destination):
    drive = gmaps.distance_matrix(origin,destination)['rows'][0]['elements'][0]
    drive_dist = drive['distance']['value']
    return drive_dist

def compute_and_store_dist_mat(df, cost_func):

    num_addr = len(df)

    lats = df.Latitude.tolist()
    lngs = df.Longitude.tolist()
    dist_mat = list()

    for idx_1, row_1 in df.iterrows():

        temp = list()
        origin = (df.iloc[idx_1].Latitude, df.iloc[idx_1].Longitude)

        for idx_2, row_2 in df.iterrows():

            destination = (df.iloc[idx_2].Latitude, df.iloc[idx_2].Longitude)
            gmaps_resp = gmaps.distance_matrix(origin, destination)['rows'][0]['elements'][0]

            if cost_func == drive_time:
                # convert seconds to minutes
                cost = gmaps_resp['duration']['value']/60.0

            if cost_func == drive_dist:
                # convert meters to miles: miles per meter = 0.000621371
                cost = 0.000621371*gmaps_resp['distance']['value']

            temp.append(cost)

        dist_mat.append(temp)


    cost_func_name = str(cost_func).split(' ')[1]
    dist_mat_file_name = f'{cost_func_name}_mat.dat'
    dist_mat_cache = np.memmap(dist_mat_file_name, dtype='float64', mode='w+', shape=(num_addr, num_addr))
    dist_mat_cache[:] = np.array(dist_mat)[:]

    return dist_mat

def generate_random_coords(num_addrs, hq_file, out_file):

    # sample random addresses in Houston
    # around Rothko chapel headquarters

    # Rothko chapel coords and address
    # rothko_hq = '3900 Yupon St, Houston, TX 77006, USA'

    rothko_lat = 29.7376
    rothko_lng = -95.3962

    df_rothko = pd.DataFrame(columns=['Latitude', 'Longitude'])
    df_rothko['Latitude'] = [rothko_lat]
    df_rothko['Longitude'] = [rothko_lng]

    if hq_file not in [file for file in glob.glob("*.csv")]:
        df_rothko.to_csv(hq_file, index=False)

    # min and max lat and lng for Houston addresses
    min_lat = 29.70238 # DeBakey VA Medical Center
    max_lat = 29.813222 # Heights @ 6-10

    min_lng = -95.623658 # Eldridge @ Westheimer
    max_lng = -95.32577 # Eastwood Park

    lat_rng = max_lat-min_lat
    lng_rng = max_lng-min_lng

    # sample random points in the rectangle 
    # above according to normal dist in lat
    # and exp dist in lng

    lat_scale = 0.15
    lng_scale = 0.1

    lats = list()
    lngs = list()
    coord_pairs = list()

    r_addrs = range(num_addrs)

    for i in r_addrs:

        lat = np.random.normal(rothko_lat, scale=lat_scale*(lat_rng))
        lng = rothko_lng-np.random.exponential(scale=lng_scale*lng_rng)
        lats.append(lat)
        lngs.append(lng)
        coord_pairs.append([lat, lng])

    df_map = pd.DataFrame(columns=['Latitude', 'Longitude'])
    df_map['Latitude'] = lats
    df_map['Longitude'] = lngs
    df_map.iloc[0] = df_rothko.iloc[0]
    df_map.to_csv(out_file, index=True)

    return df_map

def coords_generation(n_addrs, file_coords, file_hq, regen_coords, recompute_dist_mat):

    if file_coords not in [f for f in glob.glob("*.csv")] or regen_coords:
        generate_random_coords(n_addrs, file_hq, file_coords) # does not use API calls

    df_coords = pd.read_csv(file_coords, dtype={ 'High': np.float64, 
                                                  'Low': np.float64 })
    cost_funcs = [drive_time]

    if recompute_dist_mat:
        for cost_func in cost_funcs:
            # (!) uses API calls
            compute_and_store_dist_mat(df_coords, cost_func)

    return df_coords

def coords_cluster(coords_df, dist_mat, cluster_method, n_clusters, kth_neighbor):

    # implement local statistics a la Zelnik-Manor and Perona:
    # "Self-tuning spectral clustering." NIPS 17 (2004)

    K = kth_neighbor
    n_coords = len(coords_df)

    Kth_neighbors = np.array( 
                   [ coords_df.iloc[np.where(dist_mat[i]==np.sort(dist_mat[i])[K])[0][0]].values
                     for i in range(n_coords) ]
                    )
    # index of Kth_neighbor of location i
    Kth_neighbor_idxs = [np.where(dist_mat[i]==np.sort(dist_mat[i])[K])[0][0] 
                        for i in range(n_coords)]

    sigma = np.array([dist_mat[i][Kth_neighbor_idxs[i]] 
                        for i in range(n_coords)])

    local_sim_mat = np.array([[np.exp(-dist_mat[i][j]*dist_mat[j][i]/(2*sigma[i]*sigma[j])) 
                               for i in range(n_coords)] 
                               for j in range(n_coords)])

    for file in glob.glob("spectral_cluster_*.csv"):
        os.remove(file)

    r_clstrs = range(n_clusters)
    cluster_files = [ f'spectral_cluster_{i}.csv' for i in r_clstrs]

     # SPECTRAL CLUSTERING
    if cluster_method=='spectral':

        spectral = SpectralClustering(n_clusters = n_clusters, 
                                        affinity='precomputed',
                                        assign_labels='discretize', 
                                        random_state=0).fit(local_sim_mat)

        labels = spectral.labels_

    # k-MEANS CLUSTERING
    if cluster_method=='kmeans':

        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=0).fit(local_sim_mat)

        labels = kmeans.labels_

    coords_df['Label'] = labels

    # list of dataframes
    list_of_dfs = [coords_df.loc[coords_df.Label==i] for i in r_clstrs]

    for i in range(n_clusters):

        list_of_dfs[i] = list_of_dfs[i].rename(columns={'Unnamed: 0': 'Index'})
        list_of_dfs[i].to_csv(cluster_files[i], index=False)

    clstr_idxs = {i:list_of_dfs[i].index.tolist() if 0 in list_of_dfs[i].index.tolist() else [0] + list_of_dfs[i].index.tolist() for i in r_clstrs}

    return clstr_idxs

def tsp_on_clusters(dist_mat, cluster_idxs):

    # symm_time_mat = 0.5*(dist_mat+np.matrix.transpose(dist_mat))
    n_clstrs = len(cluster_idxs)
    r_clstrs = range(n_clstrs)

    print(cluster_idxs)

    # get submatrices of drive_time_mat 
    # corresponding to cluster indices

    clstr_mats = [[i, get_submat_from_idx_list(dist_mat, cluster_idxs[i])] 
                    for i in r_clstrs]

    clstr_tsp_seqs = list()
    clstr_tsp_times = list()

    tsp_solns = list()

    for clstr_mat in clstr_mats:

        clstr_label = clstr_mat[0]

        r = range(len(clstr_mat[1]))
        # dist = {(i,j): clstr_mat[1][i][j] for i in r for j in r}

        # TSP solution pair: (total cost, sequence traversed)
        tsp_solns.append(google_OR_tools_TSP_soln(clstr_mat[1]))
        # clstr_tsp_seqs.append([clstr_label, t[1]])

        # if t[0]==None:
        #     clstr_tsp_times.append(0.0)
        # else:
        #     clstr_tsp_times.append(t[0])

    return tsp_solns
    # return sum(clstr_tsp_times), clstr_tsp_times, [ np.take(cluster_idxs[clstr_tsp_seq[0]], [ clstr_tsp_seq[1]]).tolist()[0] for clstr_tsp_seq in clstr_tsp_seqs]

def hyperparam_opt(df, dist_mat, cluster_method, max_clstrs, max_K):

    # (*) can optimize over cluster method as well

    hyperopt_min = list()
    max_clstr_times = list()
    tsp_solns = list()

    for K in range(1,max_K):
        for n_clstrs in range(1, max_clstrs):

            clstr_idxs = coords_cluster(df, dist_mat, 
                                        cluster_method, n_clstrs, K)
            tsp = tsp_on_clusters(dist_mat, clstr_idxs)
            tsp_solns.append(tsp)
            max_clstr_time = np.max(tsp[1])
            max_clstr_times.append(max_clstr_time)
            hyperopt_min.append([K, n_clstrs, clstr_idxs])
        
    opt_idx = max_clstr_times.index(min(max_clstr_times))
    clstr_times = list(tsp_solns[opt_idx])[1]
    return len(clstr_times), m.ceil(max(clstr_times))

def google_OR_tools_mTSP_soln(dist_mat, n_clusters):

    """Solve the CVRP problem."""
    # Instantiate the data problem.
    # `data` is a dictionary with the following keys:
    # ('dist_mat', 'num_clusters', 'depot_index')
    data = dict()
    data['distance_matrix'] = dist_mat
    data['num_vehicles'] = n_clusters
    data['depot'] = 0

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], 
                                               data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return print_mTSP_solution(data, manager, routing, solution)

def google_OR_tools_TSP_soln(dist_mat):

    data = dict()
    data['distance_matrix'] = dist_mat
    data['num_vehicles'] = 1
    data['depot'] = 0

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    output = list()
    if solution:

        output.append(solution.ObjectiveValue())

        index = routing.Start(0)
        plan_output = list()
        route_distance = 0

        while not routing.IsEnd(index):

            plan_output.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

        plan_output.append(manager.IndexToNode(index))
        output.append(plan_output)

        return output

def print_mTSP_solution(data, manager, routing, solution):

    """Prints solution on console."""
    max_route_distance = 0
    plan_output = list()
    route_distances = list()

    for vehicle_id in range(data['num_vehicles']):

        vehicle_plan = list()
        index = routing.Start(vehicle_id)
        route_distance = 0

        while not routing.IsEnd(index):

            vehicle_plan.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        plan_output.append([route_distance, vehicle_plan])
        route_distances.append(route_distance)
        max_route_distance = max(route_distance, max_route_distance)

    print()
    print(plan_output)
    print(f'Max of route times: {max_route_distance}')
    print(f'Sum of route times: {sum(route_distances)}')

