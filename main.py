import time
import json
from collections import defaultdict

import numpy as np
import pickle
from numpy.linalg import norm

from utils import evaluate_clustering_result


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def load_normalized_features(features_file):
    try:
        with open(features_file, 'rb') as f:
            data = pickle.load(f)

        image_names = list(data.keys())
        feature_vectors = np.array([
            vec / norm(vec) for vec in data.values()
        ])

        return image_names, feature_vectors

    except FileNotFoundError:
        print(f"Error: File '{features_file}' not found.")
    except pickle.UnpicklingError:
        print(f"Error: File '{features_file}' is not a valid pickle file.")
    except Exception as e:
        print(f"Unexpected error loading or processing '{features_file}': {e}")

    return None, None


def assign_to_clusters(image_names, feature_vectors, centroids, min_similarity):
    clusters = defaultdict(list)

    for img_name, vec in zip(image_names, feature_vectors):
        best_sim = -1
        best_cluster = None

        for cid, centroid in enumerate(centroids):
            sim = cosine(vec, centroid)
            if sim > min_similarity and sim > best_sim:
                best_sim = sim
                best_cluster = cid

        if best_cluster is not None:
            clusters[best_cluster].append(img_name)
        else:
            new_cid = len(centroids)
            centroids.append(vec)
            clusters[new_cid].append(img_name)

    return clusters, centroids


def recalculate_centroids(clusters, feature_vectors, image_names):
    new_centroids = []
    for cid in range(len(clusters)):
        members = clusters[cid]
        if not members:
            continue
        vectors = np.array([feature_vectors[image_names.index(name)] for name in members])
        centroid = np.mean(vectors, axis=0)
        new_centroids.append(centroid / norm(centroid))
    return new_centroids


def filter_small_clusters(clusters, min_cluster_size):
    return {
        cid: members for cid, members in clusters.items()
        if len(members) >= min_cluster_size
    }


def cluster_data(features_file, min_cluster_size, iterations=10):
    print(f'starting clustering images in file {features_file}')

    image_names, feature_vectors = load_normalized_features(features_file)
    if image_names is None:
        print("Failed to load features. Exiting.")
        exit(1)
    min_similarity = 0.60
    centroids = []
    clusters = {}

    for iteration in range(iterations):
        clusters, centroids = assign_to_clusters(
            image_names, feature_vectors, centroids, min_similarity
        )

        new_centroids = recalculate_centroids(clusters, feature_vectors, image_names)

        if len(new_centroids) == len(centroids) and all(
            np.allclose(a, b, atol=1e-4) for a, b in zip(centroids, new_centroids)
        ):
            break

        centroids = new_centroids

    cluster2filenames = filter_small_clusters(clusters, min_cluster_size)
    return cluster2filenames


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    result = cluster_data(config['features_file'],
                          config['min_cluster_size'],
                          config['max_iterations'])

    evaluation_scores = evaluate_clustering_result(config['labels_file'], result)  # implemented
    
    print(f'total time: {round(time.time()-start, 0)} sec')
