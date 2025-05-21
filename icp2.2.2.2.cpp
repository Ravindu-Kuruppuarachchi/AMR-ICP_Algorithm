#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <flann/flann.hpp>

using namespace Eigen;
using namespace std;

struct ICPResult {
    vector<Matrix3d> transformation_history;
    MatrixXd aligned_points;
    double total_rotation;
    Vector2d total_translation;
    double translation_distance;
};

vector<Vector2d> readPointsFromFile(const string& filename) {
    vector<Vector2d> points;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return points;
    }

    double x, y;
    while (file >> x >> y) {
        points.emplace_back(x, y);
    }

    return points;
}

tuple<double, Vector2d> pointBasedMatching(const vector<pair<Vector2d, Vector2d>>& pointPairs) {
    if (pointPairs.empty()) {
        return {numeric_limits<double>::quiet_NaN(), Vector2d::Constant(numeric_limits<double>::quiet_NaN())};
    }

    Vector2d mean_ref = Vector2d::Zero();
    Vector2d mean_points = Vector2d::Zero();
    for (const auto& pair : pointPairs) {
        mean_ref += pair.first;
        mean_points += pair.second;
    }
    mean_ref /= pointPairs.size();
    mean_points /= pointPairs.size();

    double s_x_xp = 0, s_y_yp = 0, s_x_yp = 0, s_y_xp = 0;
    for (const auto& pair : pointPairs) {
        Vector2d ref_centered = pair.first - mean_ref;
        Vector2d point_centered = pair.second - mean_points;
        s_x_xp += ref_centered.x() * point_centered.x();
        s_y_yp += ref_centered.y() * point_centered.y();
        s_x_yp += ref_centered.x() * point_centered.y();
        s_y_xp += ref_centered.y() * point_centered.x();
    }

    double rot_angle = atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp);
    double c = cos(rot_angle), s = sin(rot_angle);
    Vector2d translation = mean_points - Rotation2Dd(rot_angle) * mean_ref;

    return {rot_angle, translation};
}

ICPResult icp(const MatrixXd& reference, MatrixXd points, 
             int max_iterations = 100, double distance_threshold = 0.3,
             double convergence_trans_thresh = 1e-3, 
             double convergence_rot_thresh = 1e-4,
             int point_pairs_thresh = 10, bool verbose = false) {
    
    ICPResult result;
    result.total_rotation = 0.0;
    result.total_translation = Vector2d::Zero();
    
    // Convert Eigen matrices to FLANN format
    flann::Matrix<double> ref_flann((double*)reference.data(), reference.rows(), reference.cols());
    flann::Matrix<double> points_flann((double*)points.data(), points.rows(), points.cols());
    
    // Build KD-tree for reference points
    flann::Index<flann::L2<double>> index(ref_flann, flann::KDTreeIndexParams(4));
    index.buildIndex();
    
    // For nearest neighbor search
    flann::Matrix<int> indices(new int[points.rows()], points.rows(), 1);
    flann::Matrix<double> distances(new double[points.rows()], points.rows(), 1);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        if (verbose) {
            cout << "------ iteration " << iter << " ------" << endl;
        }
        
        // Find nearest neighbors
        index.knnSearch(points_flann, indices, distances, 1, flann::SearchParams(128));
        
        // Collect valid point pairs
        vector<pair<Vector2d, Vector2d>> pointPairs;
        for (int i = 0; i < points.rows(); ++i) {
            if (distances[i][0] < distance_threshold) {
                pointPairs.emplace_back(
                    reference.row(indices[i][0]).transpose(),
                    points.row(i).transpose()
                );
            }
        }
        
        if (pointPairs.size() < point_pairs_thresh) {
            if (verbose) {
                cout << "No better solution can be found (very few point pairs)!" << endl;
            }
            break;
        }
        
        // Compute transformation
        auto [rot_angle, translation] = pointBasedMatching(pointPairs);
        if (isnan(rot_angle)) {
            if (verbose) {
                cout << "No better solution can be found!" << endl;
            }
            break;
        }
        
        if (verbose) {
            cout << "Rotation: " << rot_angle * 180.0 / M_PI << " degrees" << endl;
            cout << "Translation: " << translation.transpose() << endl;
        }
        
        // Update cumulative transformation
        result.total_rotation += rot_angle;
        result.total_translation = Rotation2Dd(result.total_rotation) * translation + result.total_translation;
        
        // Transform points
        points = (Rotation2Dd(rot_angle) * points.transpose()).transpose();
        points.rowwise() += translation.transpose();
        
        // Store transformation
        Matrix3d transform = Matrix3d::Identity();
        transform.topLeftCorner(2, 2) = Rotation2Dd(rot_angle).toRotationMatrix();
        transform.topRightCorner(2, 1) = translation;
        result.transformation_history.push_back(transform);
        
        // Check convergence
        if (abs(rot_angle) < convergence_rot_thresh && 
            translation.norm() < convergence_trans_thresh) {
            if (verbose) {
                cout << "Converged!" << endl;
            }
            break;
        }
    }
    
    // Calculate final aligned points using cumulative transformation
    result.aligned_points = (Rotation2Dd(result.total_rotation) * points.transpose()).transpose();
    result.aligned_points.rowwise() += result.total_translation.transpose();
    result.translation_distance = result.total_translation.norm();
    
    delete[] indices.ptr();
    delete[] distances.ptr();
    
    return result;
}

void plotResults(const MatrixXd& reference, const MatrixXd& source, 
                const MatrixXd& aligned, const ICPResult& result) {
    // In a real application, you would use a plotting library like matplotlib-cpp
    // Here we just print the results
    
    cout << "\n=== Before Alignment ===" << endl;
    cout << "Reference points: " << reference.rows() << " points" << endl;
    cout << "Source points: " << source.rows() << " points" << endl;
    
    cout << "\n=== After Alignment ===" << endl;
    cout << "Total Rotation: " << result.total_rotation * 180.0 / M_PI << " degrees" << endl;
    cout << "Total Translation: " << result.total_translation.transpose() << endl;
    cout << "Translation Distance: " << result.translation_distance << endl;
}

int main() {
    // Read point clouds
    vector<Vector2d> ref_points = readPointsFromFile("scan1.txt");
    vector<Vector2d> source_points = readPointsFromFile("scan2.txt");
    
    // Convert to Eigen matrices
    MatrixXd reference(ref_points.size(), 2);
    MatrixXd source(source_points.size(), 2);
    
    for (size_t i = 0; i < ref_points.size(); ++i) {
        reference.row(i) = ref_points[i];
    }
    for (size_t i = 0; i < source_points.size(); ++i) {
        source.row(i) = source_points[i];
    }
    
    // Run ICP
    ICPResult result = icp(reference, source, 100, 0.3, 1e-3, 1e-4, 10, true);
    
    // Show results
    plotResults(reference, source, result.aligned_points, result);
    
    return 0;
}