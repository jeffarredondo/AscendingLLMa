import numpy as np

def compute_distance_matrix(embeddings):
    """
    Compute pairwise Euclidean distance matrix from embeddings.
    embeddings: list of numpy arrays
    Returns: NxN distance matrix
    """
    n = len(embeddings)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = embeddings[i] - embeddings[j]
            d = np.sqrt(np.dot(diff, diff))
            D[i, j] = d
            D[j, i] = d
    return D

def gromov_delta(D):
    """
    Approximate Gromov delta hyperbolicity from a distance matrix.
    
    For all quadruples (i,j,k,l), compute the three sums:
        S1 = d(i,j) + d(k,l)
        S2 = d(i,k) + d(j,l)
        S3 = d(i,l) + d(j,k)
    
    Sort descending. Delta = (S1 - S2) / 2 for the worst case quadruple.
    
    Low delta (~0) = tree-like = hyperbolic
    High delta = not tree-like = spherical or flat
    
    O(n^4) but fine for small concept sets.
    """
    n = D.shape[0]
    max_delta = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    s = sorted([
                        D[i,j] + D[k,l],
                        D[i,k] + D[j,l],
                        D[i,l] + D[j,k]
                    ], reverse=True)
                    delta = (s[0] - s[1]) / 2.0
                    if delta > max_delta:
                        max_delta = delta

    return max_delta

def eigenvalue_signature(D):
    """
    Analyze eigenvalue structure of distance matrix to classify geometry.
    
    Hyperbolic (tree-like):
        - One large positive eigenvalue
        - Rest near zero or negative
        
    Spherical (ring-like):
        - Two dominant eigenvalues of similar magnitude
        - Oscillating sign pattern
        
    Flat (linear):
        - One dominant eigenvalue
        - Second eigenvalue also significant
        - Monotone decrease
        
    Returns: 'hyperbolic', 'spherical', or 'flat'
    """
    # Double center the distance matrix (standard MDS preprocessing)
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    
    # Normalize
    total = np.sum(np.abs(eigenvalues))
    if total < 1e-10:
        return 'flat'
    
    normed = eigenvalues / total
    
    # How dominant is the first eigenvalue?
    first_dominance = normed[0]
    
    # How similar are the first two eigenvalues?
    if len(normed) > 1:
        ratio = normed[1] / (normed[0] + 1e-10)
    else:
        ratio = 0.0

    # How many eigenvalues are significantly negative?
    neg_count = np.sum(eigenvalues < -0.05 * np.abs(eigenvalues[0]))

    if ratio > 0.6:
        # Two dominant eigenvalues of similar size = ring structure
        return 'spherical'
    elif neg_count > n // 3:
        # Many negative eigenvalues = hyperbolic structure
        return 'hyperbolic'
    elif first_dominance > 0.7:
        # One very dominant eigenvalue = linear structure
        return 'flat'
    else:
        # Default to hyperbolic — hierarchy is everywhere in language
        return 'hyperbolic'

def classify_geometry(concept, all_concepts, all_embeddings, k=5):
    """
    Classify a concept's local neighborhood geometry.
    
    1. Find k nearest neighbors in embedding space
    2. Compute distance matrix of neighborhood
    3. Compute Gromov delta
    4. Analyze eigenvalue signature
    5. Return geometry assignment
    
    Returns: 'hyperbolic', 'spherical', or 'flat'
    """
    concept_idx = next(
        i for i, c in enumerate(all_concepts) if c['name'] == concept['name']
    )
    concept_embedding = all_embeddings[concept_idx]

    # Find k nearest neighbors by cosine similarity
    similarities = []
    for i, emb in enumerate(all_embeddings):
        if i == concept_idx:
            continue
        sim = np.dot(concept_embedding, emb) / (
            np.linalg.norm(concept_embedding) * np.linalg.norm(emb) + 1e-10
        )
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbor_indices = [concept_idx] + [i for i, _ in similarities[:k]]

    # Build neighborhood embeddings
    neighborhood = [all_embeddings[i] for i in neighbor_indices]

    # Need at least 4 points for Gromov delta
    if len(neighborhood) < 4:
        return 'hyperbolic'

    D = compute_distance_matrix(neighborhood)
    delta = gromov_delta(D)
    sig = eigenvalue_signature(D)

    print(f"    {concept['name']:20} delta={delta:.3f} signature={sig}")

    # Gromov delta as tiebreaker
    # Low delta strongly suggests hyperbolic regardless of eigenvalue signature
    if delta < 1.0:
        return 'hyperbolic'

    return sig