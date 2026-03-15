import numpy as np
import librosa
import trimesh
from scipy.spatial import cKDTree
import os
from scipy.io import wavfile
import h5py


def load_mesh(mesh_folder):
    """
    Load the mesh from the specified folder.

    Parameters:
    mesh_folder (str): Path to the folder containing the mesh files.

    Returns:
    mesh (trimesh.Trimesh): Loaded mesh object.
    """
    # Path to the necessary file
    # mesh_path = os.path.join(mesh_folder, 'habitat/mesh_semantic.ply')  # Load the semantic mesh
    mesh_path = os.path.join(mesh_folder, 'house.obj')  # Load the semantic mesh
    # mesh_path = os.path.join(mesh_folder, '20_possion_10.ply')  # Load the semantic mesh


    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    return mesh

def get_rays_Fibonacci(N):
    """
    Generate N uniformly distributed rays using the Fibonacci lattice method.

    Parameters:
    N (int): The number of rays to generate.

    Returns:
    directions (np.ndarray): An N x 3 array where each row represents a direction vector of a ray.
    """
    phi = (1 + np.sqrt(5)) / 2
    indices = np.arange(0, N) + 0.5
    z = 1 - 2 * indices / N
    radius = np.sqrt(1 - z * z)
    theta = 2 * np.pi * indices / phi
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    directions = np.stack((x, y, z), axis=-1)
    return directions

def extract_features(mesh, origin, occlusion_lst, direction_method="Fibonacci", N1=1024, N2=None):
    """
    Extract features by shooting rays and analyzing intersections with the mesh.

    Parameters:
    mesh (trimesh.Trimesh): The mesh object.
    origin (np.ndarray): The origin point from which rays are shot.
    occlusion_lst (list): List of distances for occlusion counts.
    direction_method (str): Method for generating ray directions ("Fibonacci").
    N1 (int): Number of rays to generate with Fibonacci method.
    N2 (int, optional): Number of rays to generate with Uniform method (not used in this function).

    Returns:
    np.ndarray: Feature vector extracted from the mesh.
    """
    if direction_method == "Fibonacci":
        directions = get_rays_Fibonacci(N1)
    else:
        raise TypeError("Wrong arguments in direction method.")

    # Perform ray-mesh intersection for all directions at once
    ray_origins = np.tile(origin, (N1, 1))
    ray_directions = directions

    # print(ray_origins.shape, ray_directions.shape)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions
    )

    # print(locations.shape)
    # print(len(np.unique(index_ray)))
    # print(len(np.unique(ray_origins, axis=1)))

    # # Identify the rays that are missing intersections
    # all_rays = np.arange(N1)
    # rays_with_intersections = np.unique(index_ray)
    # missing_rays = np.setdiff1d(all_rays, rays_with_intersections)

    # # Print or inspect missing rays
    # print("Rays without intersections:", missing_rays)

    # Calculate the distance between each ray origin and its corresponding intersection points
    raw_distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    # Create an array to store the first intersection point for each ray
    first_hit_location = np.zeros((N1, 3))
    first_hit_index_ray = np.zeros(N1, dtype=int)
    first_hit_index_tri = np.zeros(N1, dtype=int)

    distances = np.zeros(N1)
    normals = np.zeros((N1, 3))
    # Iterate through all rays
    for ray_idx in range(N1):
        # Get all intersections for the current ray
        hits_for_ray = np.where(index_ray == ray_idx)[0]
        
        # If there are multiple intersections, find the closest one (smallest distance)
        if len(hits_for_ray) > 0:
            i = np.argmin(raw_distances[hits_for_ray])
            closest_hit_idx = hits_for_ray[i]

            distances[ray_idx] = raw_distances[closest_hit_idx]
            normals[ray_idx] = mesh.face_normals[index_tri[closest_hit_idx]]
        else:
            distances[ray_idx] = -0.1
            normals[ray_idx] = np.array([0,0,0])
        

    # # Process the intersection results
    # for loc, ray_idx, tri_idx in zip(locations, index_ray, index_tri):
    #     distance = np.linalg.norm(np.array(loc) - np.array(origin))
    #     # print(loc, origin, distance)
    #     distances[ray_idx] = distance
    #     # print(distances[ray_idx])
    #     mesh.face_normals[tri_idx]

    # Calculate distance statistics
    tree = cKDTree(directions)
    mean_distances = []
    var_distances = []
    for i, distance in enumerate(distances):
        _, idx = tree.query(directions[i], k=1+8)
        neighbor_distances = distances[idx]
        valid_distances = neighbor_distances[neighbor_distances != -0.1]
        if len(valid_distances) > 0:
            mean_distance = np.mean(valid_distances)
            var_distance = np.var(valid_distances)
        else:
            mean_distance = -0.1
            var_distance = -0.1
        mean_distances.append(mean_distance)
        var_distances.append(var_distance)

    mean_distances = np.array(mean_distances)
    var_distances = np.array(var_distances)
    
    # Calculate occlusion counts
    occlusion_count = np.array([np.sum(distances < threshold_distance) for threshold_distance in occlusion_lst])

    # Concatenate all features into a single feature vector
    feature_vector = np.concatenate([
        distances,               # N
        mean_distances,          # N
        var_distances,           # N
        normals.flatten(),       # 3*N
        occlusion_count          # 8
    ])
    
    return feature_vector

def probe_environment(points, mesh_folder, occlusion_lst, direction_method="Fibonacci", N1=1024, N2=None):
    """
    Probe the environment by extracting features for each point in the given array of points.

    Parameters:
    points (np.ndarray): Array of points (N, 3) to probe.
    mesh_folder (str): Path to the folder containing the mesh and related files.
    occlusion_lst (list): List of distances for occlusion counts.
    direction_method (str): Method for generating ray directions ("Fibonacci").
    N1 (int): Number of rays to generate with Fibonacci method.
    N2 (int, optional): Number of rays to generate with Uniform method (not used in this function).

    Returns:
    np.ndarray: Array of feature vectors for each point.
    """
    from tqdm import tqdm
    # Load the mesh
    mesh = load_mesh(mesh_folder)
    
    # Initialize a list to hold feature vectors for all points
    features_list = []
    
    # Iterate over each point to extract features
    for point in tqdm(points):
        features = extract_features(mesh, point, occlusion_lst, direction_method, N1, N2)
        # print(features.shape)
        features_list.append(features)
    
    # Convert the list of feature vectors to a NumPy array
    features_array = np.array(features_list)
    
    # Normalize each feature vector
    avg_features = np.mean(features_array, axis=0)
    std_features = np.std(features_array, axis=0)
    
    normalized_features = (features_array - avg_features) / std_features
    
    return normalized_features


def probe_environment_dict(named_points, mesh_path, occlusion_lst, direction_method="Fibonacci", N1=1024, N2=None):
    """
    Probe the environment by extracting features for each named point.

    Parameters:
    named_points (dict): Dictionary of point_name → (x, y, z).
    occlusion_lst (list): List of distances for occlusion counts.
    direction_method (str): Method for generating ray directions ("Fibonacci").
    N1 (int): Number of rays to generate with Fibonacci method.
    N2 (int, optional): Number of rays to generate with Uniform method (not used in this function).

    Returns:
    dict: Mapping from point name to normalized feature vector (np.ndarray).
    """
    from tqdm import tqdm
    import numpy as np

    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh')

    # Extract raw (unnormalized) features for all points
    names = list(named_points.keys())
    raw_features = []

    for name in tqdm(names, desc="Probing points"):
        point = named_points[name]
        feat = extract_features(mesh, point, occlusion_lst, direction_method, N1, N2)
        raw_features.append(feat)

    raw_features = np.array(raw_features)

    # Normalize features across all points
    avg = np.mean(raw_features, axis=0)
    std = np.std(raw_features, axis=0)
    normed = (raw_features - avg) / (std+1e-6)

    # Return as name → feature vector dict
    feature_dict = {
        name: normed[i]
        for i, name in enumerate(names)
    }

    return feature_dict