import numpy as np
from poincare import place_concept, poincare_distance, find_nearest, fit_domains
from manifold_utils import classify_geometry
import ollama

COMPONENTS = ['hyperbolic', 'spherical', 'flat']

def get_embedding(text):
    """Get embedding from ollama nomic-embed-text"""
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def embed_all(concepts_data):
    """Embed all concepts. Returns list of embeddings in same order."""
    embeddings = []
    for c in concepts_data:
        text = f"{c['name']}: {c.get('description', '')}"
        emb = get_embedding(text)
        embeddings.append(emb)
    return embeddings

def cosine_similarity(a, b):
    """Cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# --- S¹ placement ---
def place_spherical(concept_name, cluster_id, n_clusters,
                    offset=0.0, description="", properties=None, embedding=None):
    theta = (2 * np.pi * cluster_id / n_clusters) + offset
    coords = np.array([np.cos(theta), np.sin(theta)])
    return {
        'name': concept_name,
        'geometry': 'spherical',
        'theta': theta,
        'coords_s1': coords,
        'description': description,
        'properties': properties or [],
        'cluster_id': cluster_id,
        'embedding': embedding
    }

def spherical_distance(u, v):
    """Great circle distance on S¹"""
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arccos(dot)

# --- ℝ placement ---
def place_flat(concept_name, position, description="", properties=None, embedding=None):
    return {
        'name': concept_name,
        'geometry': 'flat',
        'position': position,
        'coords_r': np.array([position]),
        'description': description,
        'properties': properties or [],
        'cluster_id': -1,
        'embedding': embedding
    }

def flat_distance(u, v):
    """Euclidean distance on ℝ"""
    return abs(u - v)

# --- Product manifold distance ---
def product_distance(c1, c2):
    """
    Distance across H² × S¹ × ℝ.
    Same geometry: use geometric distance.
    Cross geometry: use embedding cosine similarity as bridge.
    High semantic similarity = small distance even across geometry components.
    """
    g1 = c1.get('geometry', 'hyperbolic')
    g2 = c2.get('geometry', 'hyperbolic')

    if g1 == g2:
        if g1 == 'hyperbolic':
            return poincare_distance(c1['coords'], c2['coords'])
        elif g1 == 'spherical':
            return spherical_distance(c1['coords_s1'], c2['coords_s1'])
        else:
            return flat_distance(c1['position'], c2['position'])
    else:
        # Cross-geometry bridge via embedding similarity
        e1 = c1.get('embedding')
        e2 = c2.get('embedding')
        if e1 is not None and e2 is not None:
            sim = cosine_similarity(e1, e2)
            # similarity 1.0 = distance 0.0, similarity 0.0 = distance 1.0
            return 1.0 - sim
        return 10.0

# --- Find nearest on product manifold ---
def find_nearest_product(query_concept, all_concepts, top_k=3):
    """Find nearest neighbors across the product manifold."""
    distances = []
    for concept in all_concepts:
        d = product_distance(query_concept, concept)
        distances.append((concept['name'], d))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]

# --- Main placement function ---
def place_all(concepts_data, n_clusters=4, k_neighbors=5):
    """
    Full product manifold placement pipeline.
    1. Embed all concepts
    2. Classify geometry per concept
    3. Fit domain clusters
    4. Place each concept in its geometry component
    5. Store embedding on each concept for cross-geometry bridging
    """
    print("Embedding all concepts...")
    embeddings = embed_all(concepts_data)

    print("Classifying geometry per concept...")
    geometry_assignments = {}
    for c in concepts_data:
        geo = classify_geometry(c, concepts_data, embeddings, k=k_neighbors)
        geometry_assignments[c['name']] = geo

    from collections import Counter
    counts = Counter(geometry_assignments.values())
    print(f"  Geometry assignments: {dict(counts)}")

    print("Fitting domain clusters...")
    labels, centroids = fit_domains(concepts_data, n_clusters=n_clusters)
    label_map = {c['name']: int(labels[i]) for i, c in enumerate(concepts_data)}

    print("Placing concepts on product manifold...")
    placed = []
    for i, c in enumerate(concepts_data):
        geo = geometry_assignments[c['name']]
        cluster_id = label_map[c['name']]
        strength = c['association_strength']
        embedding = embeddings[i]

        if geo == 'hyperbolic':
            concept = place_concept(
                c['name'],
                strength,
                cluster_id=cluster_id,
                n_clusters=n_clusters,
                description=c.get('description', ''),
                properties=c.get('properties', [])
            )
            concept['geometry'] = 'hyperbolic'
            concept['embedding'] = embedding

        elif geo == 'spherical':
            concept = place_spherical(
                c['name'],
                cluster_id=cluster_id,
                n_clusters=n_clusters,
                offset=c.get('offset', 0.0),
                description=c.get('description', ''),
                properties=c.get('properties', []),
                embedding=embedding
            )

        else:  # flat
            position = 1.0 - strength
            concept = place_flat(
                c['name'],
                position=position,
                description=c.get('description', ''),
                properties=c.get('properties', []),
                embedding=embedding
            )

        placed.append(concept)
        print(f"  {c['name']:20} → {geo} (cluster={cluster_id})")

    return placed