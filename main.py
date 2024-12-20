import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import label

def read_dataset(file_path):
    points = np.loadtxt(file_path, dtype=int)
    return points

def find_connected_regions(points, canvas_size):
    canvas = np.zeros(canvas_size, dtype=int)
    for x, y in points:
        canvas[y, x] = 1
    labeled_array, num_features = label(canvas)
    return labeled_array, num_features

def calculate_centroids(labeled_array, num_features):
    centroids = []
    for label_id in range(1, num_features + 1):
        indices = np.argwhere(labeled_array == label_id)
        centroid = indices.mean(axis=0)
        centroids.append((centroid[1], centroid[0]))
    return np.array(centroids)

def plot_results(points, centroids, canvas_size, output_file):
    plt.figure(figsize=(canvas_size[0] / 100, canvas_size[1] / 100))
    plt.gca().set_aspect('equal', adjustable='box')

    vor = Voronoi(centroids)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='blue', line_width=1.5, point_size=0, ax=plt.gca())

    plt.scatter(points[:, 0], points[:, 1], c='black', alpha=0.01, s=10, label='Original Points')

    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=25, label='Centroids')

    plt.xlim(0, canvas_size[0])
    plt.ylim(0, canvas_size[1])
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_file, dpi=100)
    plt.close()

def main():
    input_file = 'DS6.txt'
    output_file = 'result.png'
    canvas_size = (960, 960) # Встановити розмір 960х540 неможливо через побудову діаграми

    points = read_dataset(input_file)

    labeled_array, num_features = find_connected_regions(points, canvas_size)

    centroids = calculate_centroids(labeled_array, num_features)

    plot_results(points, centroids, canvas_size, output_file)

if __name__ == "__main__":
    main()
