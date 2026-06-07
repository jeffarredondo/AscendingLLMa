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
import imageio
import matplotlib

matplotlib.use('Agg')
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

def visualize_manifold_gif(concepts, output_path="manifold.gif", fps=20, n_frames=72):
    """
    Rotating 3D manifold saved as a gif.
    Infrared hotspot coloring by association strength.
    """
    import imageio
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    cmap = matplotlib.colormaps['hot']

    fig = plt.figure(figsize=(12, 10), facecolor='#779eb2')
    ax = fig.add_subplot(111, projection='3d', facecolor='#779eb2')

    def get_xyz(c):
        geo = c.get('geometry', 'hyperbolic')
        if geo == 'hyperbolic':
            return np.array(c['coords'])
        elif geo == 'spherical':
            cx, cy = c['coords_s1']
            return np.array([cx, cy, 1.05])
        else:
            return np.array([c['position'], 0, -1.05])

    # Pre-compute normalization ONCE
    strengths = [c.get('association_strength', 0.5) for c in concepts]
    min_s = min(strengths)
    max_s = max(strengths)

    def normalize(s):
        """Normalize to 0.3-1.0 so nothing hits pure black"""
        n = (s - min_s) / (max_s - min_s + 1e-8)
        return 0.3 + (n * 0.7)

    def draw_frame(angle):
        ax.cla()
        ax.set_facecolor('#779eb2')
        fig.patch.set_facecolor('#779eb2')

        # --- Hyperboloid surface ---
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(-1.0, 1.0, 30)
        x_hyp = np.outer(np.cosh(v), np.cos(u))
        y_hyp = np.outer(np.cosh(v), np.sin(u))
        z_hyp = np.outer(np.sinh(v), np.ones_like(u)) * 1.2
        ax.plot_surface(x_hyp, y_hyp, z_hyp,
                       alpha=0.18, color='#9B6EEE',
                       linewidth=0, antialiased=True)

        # --- Flat plane ---
        has_flat = any(c.get('geometry') == 'flat' for c in concepts)
        if has_flat:
            px = np.linspace(-1.2, 1.2, 10)
            py = np.linspace(-1.2, 1.2, 10)
            px, py = np.meshgrid(px, py)
            pz = np.full_like(px, -1.05)
            ax.plot_surface(px, py, pz,
                           alpha=0.06, color='#50C878',
                           linewidth=0, antialiased=True)

        # --- Poincare ball wireframe ---
        u2 = np.linspace(0, 2 * np.pi, 30)
        v2 = np.linspace(0, np.pi, 30)
        xs = np.outer(np.cos(u2), np.sin(v2))
        ys = np.outer(np.sin(u2), np.sin(v2))
        zs = np.outer(np.ones(np.size(u2)), np.cos(v2))
        ax.plot_wireframe(xs, ys, zs,
                         color='#4a6a8a', alpha=0.4, linewidth=0.4)

        # --- Neighbor lines ---
        for c in concepts:
            if 'nearest' not in c:
                continue
            x1, y1, z1 = get_xyz(c)
            for neighbor_name, dist in c.get('nearest', [])[:2]:
                neighbor = next((n for n in concepts if n['name'] == neighbor_name), None)
                if neighbor and neighbor['name'] != c['name']:
                    x2, y2, z2 = get_xyz(neighbor)
                    ax.plot([x1, x2], [y1, y2], [z1, z2],
                           color='#333333', alpha=0.15, linewidth=0.5)

        # --- Infrared hotspots ---
        for c in concepts:
            strength = c.get('association_strength', 0.5)
            n = normalize(strength)
            color = cmap(n)
            x, y, z = get_xyz(c)

            # Outer glow
            ax.scatter(x, y, z, s=200 + (n * 800),
                      color=color, alpha=0.15, zorder=4)
            # Mid glow
            ax.scatter(x, y, z, s=100 + (n * 400),
                      color=color, alpha=0.35, zorder=5)
            # Core
            ax.scatter(x, y, z, s=40 + (n * 150),
                      color=color, alpha=0.95, zorder=6,
                      edgecolors='white', linewidths=0.2)

        # --- Density labels ---
        clusters = {}
        for c in concepts:
            cid = c.get('cluster_id', 0)
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(c)

        for cid, members in clusters.items():
            points = [get_xyz(c) for c in members]
            centroid = np.mean(points, axis=0)
            density = len(members)
            avg_strength = np.mean([c.get('association_strength', 0.5)
                                   for c in members])
            n_avg = normalize(avg_strength)
            label_color = cmap(n_avg)
            ax.text(centroid[0], centroid[1], centroid[2] + 0.15,
                   f"{density} concepts",
                   fontsize=7, color='#253f4b',
                   alpha=0.8, ha='center')

        ax.view_init(elev=20, azim=angle)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_axis_off()

        ax.set_title(
            "Knowledge Manifold  H² × S¹ × ℝ\n"
            "Data stored as Hierarchical · Cyclical · Flat ℝ",
            fontsize=24, color='#253f4b', pad=15
        )

    # Generate frames
    print(f"Rendering {n_frames} frames...")
    frames = []
    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        draw_frame(angle)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:]
        frames.append(buf)
        if i % 10 == 0:
            print(f"  Frame {i}/{n_frames}")

    plt.close()

    print("Saving gif...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved to {output_path}")

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

#visualize_manifold(concepts)
#visualize_projections(concepts)
#visualize_manifold_gif(concepts, output_path="manifold.gif", fps = 30, n_frames= 100)