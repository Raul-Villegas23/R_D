# -*- coding: utf-8 -*-
"""
Created on Tue April 15 14:15:09 2023

@author: Florent Poux
"""

#%% 1. Library setup
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

#%% 2. Mesh Import

DATANAME = "pijlkruid37-37.glb"
mesh = o3d.io.read_triangle_mesh("../DATA/" + DATANAME)
print(mesh)

#%% Conversion to Point Cloud
pcd = mesh.sample_points_uniformly(number_of_points=1000000)
print(pcd)

#%% 3. Data Pre-Processing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)
# Visualize using draw
o3d.visualization.draw_geometries([pcd])

# 3.1. Sampling
#%% 3.1. Random Sampling Test
retained_ratio = 0.2
sampled_pcd = pcd.random_down_sample(retained_ratio)
o3d.visualization.draw_geometries([sampled_pcd], window_name = "Random Sampling")

#%% 3.2. Statistical outlier filter 
nn = 16
std_multiplier = 10

#The statistical outlier removal filter returns the point cloud and the point indexes
filtered_pcd, filtered_idx = pcd.remove_statistical_outlier(nn, std_multiplier)

#Visualizing the points filtered
outliers = pcd.select_by_index(filtered_idx, invert=True)
outliers.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([filtered_pcd, outliers])

#%% 3.3. Voxel downsampling
voxel_size = 0.05

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])

#%% 3.4. Estimating normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
print(nn_distance)
#setting the radius search to compute normals
radius_normals=nn_distance*4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

# 3.3. Visualizing the point cloud in Python
pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled,outliers])

#%% 4. Extracting and Setting Parameters

# Retrieving the parameters from a visual inspection
'''-- Mouse view control --
  Left button + drag         : Rotate.
  Ctrl + left button + drag  : Translate.
  Wheel button + drag        : Translate.
  Shift + left button + drag : Roll.
  Wheel                      : Zoom in/out.

-- Keyboard view control --
  [/]          : Increase/decrease field of view.
  R            : Reset view point.
  Ctrl/Cmd + C : Copy current view status into the clipboard.
  Ctrl/Cmd + V : Paste view status from clipboard.

-- General control --
  Q, Esc       : Exit window.
  H            : Print help message.
  P, PrtScn    : Take a screen capture.
  D            : Take a depth capture.
  O            : Take a capture of current rendering settings.'''

# # R+1
# front = [ 0.9589353664782414, 0.26005892375915179, 0.1131915150992901 ]
# lookat = [ 255203.95752668029, 473201.8925091463, 35.155325023385068 ]
# up = [ -0.11993448154912173, 0.010154689078208061, 0.99272987384548284 ]
# zoom = 0.25999999999999956

# R0
front = [ -0.99369181161880105, 0.092518793899417626, 0.063378673835467206 ]
lookat = [ -1.043328164965434, 1.1011642697614414, -0.52209011072167588 ]
up = [ 0.062787194896183215, -0.0092928174816590079, 0.99798367306300229 ]
zoom = 0.21999999999999958

#Exterior
# front = [ -0.037058935732263293, 0.98288825505266464, 0.18043645242001422 ]
# lookat = [ -2.1257038105600592, -1.4378742133543285, -2.5282468212747435 ]
# up = [ -0.012485710864744209, -0.1810018036369839, 0.98340350523290332 ]
# zoom = 0.25999999999999956

# draw_positionned_scene(pcd, front, lookat, up, zoom)
pcd = pcd_downsampled
o3d.visualization.draw_geometries([pcd],zoom=zoom, front=front, lookat=lookat,up=up)

pt_to_plane_dist = 0.1
#%% 5. RANSAC Planar Segmentation

distance_threshold = 0.1
ransac_n = 3
num_iterations = 1000

plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,ransac_n=ransac_n,num_iterations=num_iterations)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],zoom=zoom, front=front, lookat=lookat,up=up)

#%% 6. Multi-order RANSAC
max_plane_idx = 10
# pt_to_plane_dist = nn_distance + np.std(pcd.compute_nearest_neighbor_distance())
pt_to_plane_dist = 0.1

segment_models = {}
segments = {}
rest = pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],zoom=zoom, front=front, lookat=lookat,up=up)

#%% 7. Euclidean Considerations with DBSCAN
epsilon = 0.1
min_cluster_points = 10

labels = np.array(pcd.cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd],zoom=zoom, front=front, lookat=lookat,up=up)

#%% 8. Using Euclidean Clustering within the RANSAC Definition
max_plane_idx = 10
pt_to_plane_dist = 0.1

segment_models = {}
segments = {}

rest = pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    
    # labels = np.array(segments[i].cluster_dbscan(eps=0.01, min_points=int(len(inliers)/10)))
    labels = np.array(segments[i].cluster_dbscan(eps=pt_to_plane_dist*3, min_points=10))
    rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=0)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels==0)[0]))
    segments[i].paint_uniform_color(list(colors[:3]))
    print("pass",i,"/",max_plane_idx,"done.")

# o3d.visualization.draw_geometries([segments.values()])

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)],zoom=zoom, front=front, lookat=lookat,up=up)

# o3d.visualization.draw_geometries([rest])


#%% Refined RANSAC with Euclidean clustering AND ORDERING

segment_models = {}

segments = {}

rest = pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    
    # labels = np.array(segments[i].cluster_dbscan(eps=0.01, min_points=int(len(inliers)/10)))
    labels = np.array(segments[i].cluster_dbscan(eps=pt_to_plane_dist*3, min_points=10))
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    best_candidate=int(np.unique(labels)[np.where(candidates==np.max(candidates))[0]])
    print("the best candidate is: ", best_candidate)
    rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels==best_candidate)[0]))
    segments[i].paint_uniform_color(list(colors[:3]))
    print("pass",i+1,"/",max_plane_idx,"done.")


labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],zoom=zoom, front=front, lookat=lookat,up=up)



#%%2 9. DBSCAN sur rest
o3d.visualization.draw_geometries([rest],zoom=zoom, front=front, lookat=lookat,up=up)

epsilon = 0.15
min_cluster_points = 5

rest_db = rest

labels = np.array(rest_db.cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest_db.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([rest_db],zoom=zoom, front=front, lookat=lookat,up=up)
o3d.visualization.draw_geometries([rest],zoom=zoom, front=front, lookat=lookat,up=up)



# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],zoom=zoom, front=front, lookat=lookat,up=up)

#%% Voxelization with Open3D
voxel_size=0.5

min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()

pcd_ransac=o3d.geometry.PointCloud()
for i in segments:
    pcd_ransac += segments[i]

pcd_ransac.paint_uniform_color([0.8, 0.1, 0.1])
rest.paint_uniform_color([0.1, 0.1, 0.8])

voxel_grid_clutter = o3d.geometry.VoxelGrid.create_from_point_cloud(rest, voxel_size=voxel_size)
voxel_grid_plane = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_ransac, voxel_size=voxel_size)

o3d.visualization.draw_geometries([voxel_grid_clutter,voxel_grid_plane])

idx_voxels=[v.grid_index for v in voxel_grid_clutter.get_voxels()]

# color_voxels=[v.color for v in voxel_grid.get_voxels()]
bounds_voxels=[np.min(idx_voxels, axis=0),np.max(idx_voxels, axis=0)]

#%% Creating a voxel grid

min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()

pcd_ransac=o3d.geometry.PointCloud()
for i in segments:
    pcd_ransac += segments[i]

def fit_voxel_grid(point_cloud, voxel_size, min_b=False, max_b=False):

    # Determine the minimum and maximum coordinates of the point cloud
    if type(min_b) == bool or type(max_b) == bool:
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
    else:
        min_coords = min_b
        max_coords = max_b

    # Calculate the dimensions of the voxel grid
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # Create an empty voxel grid
    voxel_grid = np.zeros(grid_dims, dtype=bool)    

    # Calculate the indices of the occupied voxels
    indices = ((point_cloud - min_coords) / voxel_size).astype(int)

    # Mark occupied voxels as True
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    return voxel_grid, indices

voxel_size = 0.3

ransac_voxels, idx_ransac = fit_voxel_grid(pcd_ransac.points, voxel_size, min_bound, max_bound)

rest_voxels, idx_rest = fit_voxel_grid(rest.points, voxel_size, min_bound, max_bound)

filled_ransac = np.transpose(np.nonzero(ransac_voxels))
filled_rest = np.transpose(np.nonzero(rest_voxels))

total = pcd_ransac + rest
total_voxels, idx_total = fit_voxel_grid(total.points, voxel_size, min_bound, max_bound)

empty_indices = np.transpose(np.nonzero(~total_voxels))

#%% 10. Point Cloud Export
xyz_segments=[]
for idx in segments:
    print(idx,segments[idx])
    a = np.asarray(segments[idx].points)
    N = len(a)
    b = idx*np.ones((N,3+1))
    b[:,:-1] = a
    xyz_segments.append(b)

rest_w_segments=np.hstack((np.asarray(rest.points),(labels+max_plane_idx).reshape(-1, 1)))
xyz_segments.append(rest_w_segments)

# np.savetxt("../RESULTS/" + DATANAME.split(".")[0] + ".xyz", np.concatenate(xyz_segments), delimiter=';', fmt='%1.9f')

#%% Creating a 3D mesh voxel model

def cube(c,s,compteur=0):
    v1=c+s/2*np.array([-1,-1,1])
    v2=c+s/2*np.array([1,-1,1])
    v3=c+s/2*np.array([-1,1,1])
    v4=c+s/2*np.array([1,1,1])
    v5=c+s/2*np.array([-1,1,-1])
    v6=c+s/2*np.array([1,1,-1])
    v7=c+s/2*np.array([-1,-1,-1])
    v8=c+s/2*np.array([1,-1,-1])
    f1=np.array([1,2,3])
    f2=np.array([3,2,4])
    f3=np.array([3,4,5])
    f4=np.array([5,4,6])
    f5=np.array([5,6,7])
    f6=np.array([7,6,8])
    f7=np.array([7,8,1])
    f8=np.array([1,8,2])
    f9=np.array([2,8,4])
    f10=np.array([4,8,6])
    f11=np.array([7,1,5])
    f12=np.array([5,1,3])
    vcube=[v1,v2,v3,v4,v5,v6,v7,v8]
    ch=np.empty([8,1],dtype=str)
    ch.fill('v')
    vertice=np.hstack((ch,vcube))
    faces=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
    faces= np.asarray(faces)+ compteur*8
    ch=np.empty([12,1],dtype=str)
    ch.fill('f')
    faces=np.hstack((ch,faces))
    cube=np.append(vertice,faces,axis=0)
    return cube

def voxel_modelling(filename, indices, voxel_size):
    voxel_assembly=[]
    with open(filename, "a") as f:
        cpt = 0
        for idx in indices:
            voxel = cube(idx,voxel_size,cpt)
            # f.write(b"o "+ idx +"\n")
            f.write(f"o {idx}  \n")
            np.savetxt(f, voxel,fmt='%s')
            cpt += 1
            voxel_assembly.append(voxel)
    return voxel_assembly

# vrsac = voxel_modelling("../RESULTS/ransac_vox.obj", filled_ransac, 1)
# vrest = voxel_modelling("../RESULTS/rest_vox.obj", filled_rest, 1)
# voxel_modelling("../RESULTS/empty_vox.obj", empty_indices, 1)

# %%
