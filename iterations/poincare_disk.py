import numpy as np

# Domain clusters — hardcoded for Les Mis experiment
DOMAINS = {
    "justice":    0,
    "redemption": np.pi / 2,
    "revolution": np.pi,
    "poverty":    3 * np.pi / 2
}

def to_cartesian(r, theta):
    """Convert polar to cartesian coordinates on the disk"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

def poincare_distance(u, v):
    """
    Compute hyperbolic distance between two points on the Poincaré disk
    u, v: cartesian coordinate arrays
    """
    norm_u = np.dot(u, u)
    norm_v = np.dot(v, v)
    diff = u - v
    norm_diff = np.dot(diff, diff)
    
    arg = 1 + 2 * norm_diff / ((1 - norm_u) * (1 - norm_v))
    return np.arccosh(arg)

def place_concept(concept_name, association_strength, domain, offset=0.0, description="", properties=None):
    """
    Derive position on disk from concept properties
    association_strength: 0.0 to 1.0 (Resonance style)
    domain: one of DOMAINS keys
    description: what this concept actually is
    properties: list of tags/attributes
    """
    r = 1 - (association_strength * 0.9)
    r = min(r, 0.95)
    theta = DOMAINS.get(domain, 0) + offset
    coords = to_cartesian(r, theta)
    
    return {
        "name": concept_name,
        "r": r,
        "theta": theta,
        "coords": coords,
        "description": description,
        "properties": properties or []
    }

def find_nearest(query_concept, all_concepts, top_k=3):
    """
    Given a concept, find nearest neighbors on the disk
    """
    distances = []
    for concept in all_concepts:
        d = poincare_distance(query_concept["coords"], concept["coords"])
        distances.append((concept["name"], d))
    
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
