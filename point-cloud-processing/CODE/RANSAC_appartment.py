import numpy as np
import open3d as o3d

def fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2, :]
    d = -np.dot(normal, centroid)
    return np.append(normal, d)

def ransac_plane(points, threshold, iterations):
    best_inliers = []
    best_params = None

    for _ in range(iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        params = fit_plane(sample)
        inliers = []

        for point in points:
            distance = np.abs(np.dot(params[:3], point) + params[3]) / np.linalg.norm(params[:3])
            if distance < threshold:
                inliers.append(point)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = params

    return best_params, np.array(best_inliers)

pcd = o3d.io.read_point_cloud("DATA/ITC_firstfloor.ply")
o3d.visualization.draw_geometries([pcd])
points = np.asarray(pcd.points)
plane_params, inliers = ransac_plane(points, threshold=0.01, iterations=1000)
print("Plane parameters:", plane_params)