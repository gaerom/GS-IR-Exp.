import open3d as o3d
import numpy as np

scene_id = 10
file_path = f'/mnt/storage1/gaussian-splatting/output/rwavs_{scene_id}/point_cloud/iteration_30000/point_cloud.ply'

point_cloud = o3d.io.read_point_cloud(file_path)


num_points = np.asarray(point_cloud.points).shape[0]
light_gray = np.tile([0.8, 0.8, 0.8], (num_points, 1))  
point_cloud.colors = o3d.utility.Vector3dVector(light_gray)


# target_coordinates = np.array([0.9, 1.7, -0.3]) 
target_coordinates = np.array([-3.5, 1.2, -1.4]) 
# target_coordinates = np.array([-0.2, 0.7, 3.9])


target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
target_sphere.translate(target_coordinates)
target_sphere.paint_uniform_color([1, 0, 0]) 

vis = o3d.visualization.Visualizer()
vis.create_window()


vis.add_geometry(point_cloud)
vis.add_geometry(target_sphere)


render_option = vis.get_render_option()
render_option.background_color = np.array([1, 1, 1]) 

vis.run()
vis.destroy_window()
