import numpy as np
import cv2

#importing outpt points and colors

output_points = np.load("output/points0.npy")   
output_colors = np.load("output/colors0.npy")

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y 
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

output_file = 'output/point_cloud_using midas.ply'
create_output(output_points, output_colors, output_file)
