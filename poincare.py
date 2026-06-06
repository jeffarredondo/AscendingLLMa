import numpy as np
from sklearn.cluster import KMeans
import ollama

# --- Poincaré Ball: 3D Hyperbolic Space ---

def get_embedding(text):
    """Get embedding from ollama nomic-embed-text"""
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def fit_domains(concepts_data, n_clusters=4):
    """
    Derive domain clusters automatically from concept embeddings.
    Returns cluster centroids and labels.
    """
    texts = [f"{c['name']}: {c.get('description', '')}" for c in concepts_data]
    embeddings = np.array([get_embedding(t) for t in texts])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans.cluster_centers_

def cluster_to_angles(cluster_id, n_clusters):
    """
    Map cluster id to spherical angles (theta, phi) on the ball.
    Distribute clusters evenly across the sphere surface.
    """
    # Evenly space clusters using spherical coordinates
    theta = np.pi * (cluster_id + 1) / (n_clusters + 1)  # polar angle
    phi = 2 * np.pi * cluster_id / n_clusters              # azimuthal angle
    return theta, phi

def to_cartesian_3d(r, theta, phi):
    """Convert spherical to cartesian coordinates on the ball"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def poincare_distance(u, v):
    """
    Compute hyperbolic distance between two points on the Poincaré ball.
    Generalizes naturally from disk to N dimensions.
    """
    norm_u = np.dot(u, u)
    norm_v = np.dot(v, v)
    diff = u - v
    norm_diff = np.dot(diff, diff)
    arg = 1 + 2 * norm_diff / ((1 - norm_u) * (1 - norm_v))
    # Clamp for numerical stability
    arg = max(arg, 1.0 + 1e-8)
    return np.arccosh(arg)

def place_concept(concept_name, association_strength, cluster_id, 
                  n_clusters, offset_theta=0.0, offset_phi=0.0,
                  description="", properties=None):
    """
    Derive position on ball from concept properties.
    association_strength: 0.0 to 1.0
    cluster_id: derived from fit_domains
    """
    r = 1 - (association_strength * 0.9)
    r = min(r, 0.95)
    
    theta, phi = cluster_to_angles(cluster_id, n_clusters)
    theta += offset_theta
    phi += offset_phi
    
    coords = to_cartesian_3d(r, theta, phi)
    
    return {
        'name': concept_name,
        'r': r,
        'theta': theta,
        'phi': phi,
        'coords': coords,
        'description': description,
        'properties': properties or [],
        'cluster_id': cluster_id
    }

def find_nearest(query_concept, all_concepts, top_k=3):
    """
    Given a concept, find nearest neighbors on the ball.
    """
    distances = []
    for concept in all_concepts:
        d = poincare_distance(query_concept['coords'], concept['coords'])
        distances.append((concept['name'], d))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]