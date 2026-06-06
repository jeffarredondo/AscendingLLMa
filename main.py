import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from product_manifold import place_all, find_nearest_product, product_distance
from concepts import get_context
from middleware import build_prompt, ask_llama

MANIFOLD_CACHE = "manifold.pkl"

def load_concepts(path="concepts.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return place_all(data["concepts"], n_clusters=4, k_neighbors=5)

def load_or_build(path="concepts.json"):
    rebuild = "--rebuild" in sys.argv

    if os.path.exists(MANIFOLD_CACHE) and not rebuild:
        print("Loading saved manifold from cache...")
        with open(MANIFOLD_CACHE, "rb") as f:
            return pickle.load(f)
    else:
        print("Building manifold...")
        concepts = load_concepts(path)
        print("Saving manifold to cache...")
        with open(MANIFOLD_CACHE, "wb") as f:
            pickle.dump(concepts, f)
        print(f"Saved to {MANIFOLD_CACHE}")
        return concepts

def visualize_manifold(concepts):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.2)

    colors = {
        'hyperbolic': 'royalblue',
        'spherical':  'crimson',
        'flat':       'forestgreen'
    }

    for c in concepts:
        geo = c.get('geometry', 'hyperbolic')
        color = colors[geo]

        if geo == 'hyperbolic':
            x, y, z = c['coords']
            ax.scatter(x, y, z, s=100, color=color, zorder=5)
            ax.text(x, y, z, f"  {c['name']}", fontsize=8)
        elif geo == 'spherical':
            cx, cy = c['coords_s1']
            ax.scatter(cx, cy, 1.05, s=100, color=color, zorder=5)
            ax.text(cx, cy, 1.05, f"  {c['name']}", fontsize=8)
        else:
            pos = c['position']
            ax.scatter(pos, 0, -1.05, s=100, color=color, zorder=5)
            ax.text(pos, 0, -1.05, f"  {c['name']}", fontsize=8)

    ax.set_title(
        "AscendingLLMa — Product Manifold H² × S¹ × ℝ\n"
        "Blue=Hyperbolic  Red=Spherical  Green=Flat",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("manifold.png", dpi=150)
    plt.show()
    plt.close()
    print("Saved to manifold.png")

def visualize_projections(concepts):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("AscendingLLMa — Product Manifold Projections\nBlue=Hyperbolic  Red=Spherical  Green=Flat", fontsize=12)

    colors = {
        'hyperbolic': 'royalblue',
        'spherical':  'crimson',
        'flat':       'forestgreen'
    }

    def get_xyz(c):
        geo = c.get('geometry', 'hyperbolic')
        if geo == 'hyperbolic':
            x, y, z = c['coords']
        elif geo == 'spherical':
            cx, cy = c['coords_s1']
            x, y, z = cx, cy, 1.05
        else:
            x, y, z = c['position'], 0, -1.05
        return x, y, z

    projection_pairs = [
        (0, 1, "Top View (X-Y)"),
        (0, 2, "Front View (X-Z)"),
        (1, 2, "Side View (Y-Z)"),
    ]

    coord_labels = ["X", "Y", "Z"]

    for ax, (i, j, title) in zip(axes, projection_pairs):
        for c in concepts:
            geo = c.get('geometry', 'hyperbolic')
            color = colors[geo]
            coords = get_xyz(c)
            ax.scatter(coords[i], coords[j], s=100, color=color, zorder=5)
            ax.annotate(c['name'], (coords[i], coords[j]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=7)
        
        ax.set_xlabel(coord_labels[i])
        ax.set_ylabel(coord_labels[j])
        ax.set_title(title)
        ax.axhline(0, color='lightgray', linewidth=0.5)
        ax.axvline(0, color='lightgray', linewidth=0.5)
        ax.set_aspect('equal')
        ax.grid(False)

    plt.tight_layout()
    plt.savefig("manifold_projections.png", dpi=150)
    plt.show()
    plt.close()
    print("Saved to manifold_projections.png")

# --- Load or build ---
concepts = load_or_build()

print("\n=== Geometry Summary ===")
from collections import Counter
geo_counts = Counter(c.get('geometry', 'hyperbolic') for c in concepts)
for geo, count in geo_counts.items():
    print(f"  {geo}: {count} concepts")

print("\n=== Manifold Size ===")
import pickle as pk
if os.path.exists(MANIFOLD_CACHE):
    size_bytes = os.path.getsize(MANIFOLD_CACHE)
    print(f"  manifold.pkl: {size_bytes / 1024:.1f} KB")

print("\n=== Query Tests ===")
queries = [
    "What was SpaceX's revenue and loss in Q1 2026?",
    "What is SpaceX's valuation for the IPO?",
    "What happened with xAI and SpaceX?",
    "Is SpaceX a good investment given its losses?",
    "Has market mania affected SpaceX's valuation?",
]

for query in queries:
    print(f"\nQuery: {query}")
    context = get_context(query, concepts)
    prompt = build_prompt(query, context, concepts)
    print(f"Prompt:\n{prompt}\n")
    response = ask_llama(prompt)
    print(f"Response:\n{response}")
    print("-" * 60)

visualize_manifold(concepts)
visualize_projections(concepts)