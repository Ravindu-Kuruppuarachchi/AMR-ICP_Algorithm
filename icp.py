import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    """Euclidean distance between two points."""
    a = np.array(point1)
    b = np.array(point2)
    return np.linalg.norm(a - b, ord=2)

def point_based_matching(point_pairs):
    """Calculate rotation and translation between point pairs."""
    x_mean = y_mean = xp_mean = yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for (x, y), (xp, yp) in point_pairs:
        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = s_y_yp = s_x_yp = s_y_xp = 0
    for (x, y), (xp, yp) in point_pairs:
        s_x_xp += (x - x_mean)*(xp - xp_mean)
        s_y_yp += (y - y_mean)*(yp - yp_mean)
        s_x_yp += (x - x_mean)*(yp - yp_mean)
        s_y_xp += (y - y_mean)*(xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
    translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

    return rot_angle, translation_x, translation_y

def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, 
        convergence_translation_threshold=1e-3, convergence_rotation_threshold=1e-4,
        point_pairs_threshold=10, verbose=False):
    """ICP implementation with proper cumulative transformation tracking."""
    transformation_history = []
    cumulative_rotation = 0.0
    cumulative_translation = np.zeros(2)
    initial_points = points.copy()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print(f'------ iteration {iter_num} ------')

        # Find closest point pairs
        distances, indices = nbrs.kneighbors(points)
        closest_point_pairs = [
            (points[nn_index], reference_points[indices[nn_index][0]])
            for nn_index in range(len(distances)) 
            if distances[nn_index][0] < distance_threshold
        ]

        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # Compute incremental transformation
        rot_angle, trans_x, trans_y = point_based_matching(closest_point_pairs)
        if rot_angle is None:
            if verbose:
                print('No better solution can be found!')
            break

        if verbose:
            print(f'Rotation: {math.degrees(rot_angle):.4f} degrees')
            print(f'Translation: {trans_x:.4f}, {trans_y:.4f}')

        # Update cumulative transformation
        c, s = math.cos(rot_angle), math.sin(rot_angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        
        # New cumulative translation: R_prev * t_new + t_prev
        cumulative_translation = np.dot(
            np.array([[math.cos(cumulative_rotation), -math.sin(cumulative_rotation)],
                     [math.sin(cumulative_rotation), math.cos(cumulative_rotation)]]),
            np.array([trans_x, trans_y])
        ) + cumulative_translation
        
        # New cumulative rotation: θ_prev + θ_new
        cumulative_rotation += rot_angle

        # Transform points for next iteration
        points = np.dot(points, rot_matrix.T) + np.array([trans_x, trans_y])

        transformation_history.append(np.hstack((rot_matrix, np.array([[trans_x], [trans_y]]))))

        # Check convergence
        if (abs(rot_angle) < convergence_rotation_threshold and
            abs(trans_x) < convergence_translation_threshold and
            abs(trans_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    # Calculate final aligned points using cumulative transformation
    final_rotation_matrix = np.array([
        [math.cos(cumulative_rotation), -math.sin(cumulative_rotation)],
        [math.sin(cumulative_rotation), math.cos(cumulative_rotation)]
    ])
    final_aligned_points = np.dot(initial_points, final_rotation_matrix.T) + cumulative_translation

    return {
        'transformation_history': transformation_history,
        'aligned_points': final_aligned_points,
        'total_rotation': cumulative_rotation,
        'total_translation': cumulative_translation,
        'translation_distance': np.linalg.norm(cumulative_translation)
    }

def read_points_from_file(filename):
    """Read points from a text file."""
    return np.loadtxt(filename, usecols=(0, 1))

def plot_results(reference, source, aligned, results):
    """Plot results with transformation information."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before alignment
    ax1.scatter(reference[:, 0], reference[:, 1], s=10, alpha=0.7, label='Reference')
    ax1.scatter(source[:, 0], source[:, 1], s=10, alpha=0.7, label='Source')
    ax1.set_title('Before ICP Alignment')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)
    
    # After alignment
    ax2.scatter(reference[:, 0], reference[:, 1], s=10, alpha=0.7, label='Reference')
    ax2.scatter(aligned[:, 0], aligned[:, 1], s=10, alpha=0.7, label='Aligned')
    ax2.set_title(
        f'After ICP Alignment\n'
        f'Total Rotation: {math.degrees(results["total_rotation"]):.2f}°\n'
        f'Total Translation: [{results["total_translation"][0]:.2f}, {results["total_translation"][1]:.2f}]\n'
        f'Translation Distance: {results["translation_distance"]:.2f}'
    )
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load point clouds
    reference_points = read_points_from_file('scan1.txt')
    points_to_align = read_points_from_file('scan2.txt')

    # Run ICP
    results = icp(reference_points, points_to_align, verbose=True)

    # Print results
    print("\nFinal Transformation Results:")
    print(f"Total Rotation: {math.degrees(results['total_rotation']):.4f} degrees")
    print(f"Total Translation: X = {results['total_translation'][0]:.4f}, Y = {results['total_translation'][1]:.4f}")
    print(f"Translation Distance: {results['translation_distance']:.4f}")

    # Plot results
    plot_results(reference_points, points_to_align, results['aligned_points'], results)