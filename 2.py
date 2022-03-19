import cv2
import argparse
import numpy as np
from DirectLT import normalized_DLT
from utils import vis_img

def argparse2():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_img", type=str,
        help="input document image for rectification"
    )
    parser.add_argument(
        "-o", "--output_img", type=str,
        help="output document image after rectification"
    )
    args = parser.parse_args()
    
    return args

def bilinear_interpolate(img, H, target_size=None):
    def get_valid_region(img, grid):
        #print(grid)
        is_valid = (
            (grid[:,:,0] < img.shape[1]) & \
            (grid[:,:,0] >= 0) & \
            (grid[:,:,1] < img.shape[0]) & \
            (grid[:,:,1] >= 0)
        )
        return is_valid
    backward_H = np.linalg.inv(H)
    u = np.arange(0,img.shape[1])
    v = np.arange(0,img.shape[0])
    uu, vv = np.meshgrid(u, v)
    grid = np.stack([
        vv,uu,
        np.ones_like(uu),
    ],axis=2)
    #grid_computed = grid.copy()
    grid[:,:,[0,1]] = grid[:,:,[1,0]]
    new_grid = np.matmul(backward_H, grid.reshape(-1,3).T).T
    new_grid = new_grid.reshape(-1, grid.shape[1], 3)
    
    #new_grid = np.matmul(backward_H, grid.reshape(-1,3).T).T
    #new_grid = new_grid.reshape(-1,grid.shape[1],3)
    #new_grid[:,:,[0,1]] = new_grid[:,:,[1,0]]
    new_grid = new_grid / new_grid[:,:,2:]
    
    new_grid_uv = new_grid[:,:,:2] ### u, v
    valid_region = get_valid_region(img, np.floor(new_grid_uv).astype(int))
    new_grid_uv[~valid_region] = 0
    new_grid_uv = new_grid_uv.astype(int)
    ret_img = img[(new_grid_uv[:,:,1],new_grid_uv[:,:,0])]
    
    return ret_img

def main(args):
    img = cv2.imread(args.input_img)
    print(img.shape)
    points1 = np.array([[201,222], [625,55], [480,700], [950, 480]])
    up = points1[:,1].min()
    down = points1[:,1].max()
    left = points1[:,0].min()
    right = points1[:,0].max()
    points2 = np.array([[left, up], [right,up],[left,down],[right,down]])

    show_img = img.copy()
    for point in points2:
        u,v = point
        show_img = cv2.circle(show_img, (u,v), radius=5, color=(0, 0, 255), thickness=-1)
    vis_img(cv2.cvtColor(show_img,cv2.COLOR_BGR2RGB))
    H = normalized_DLT(points1, points2, k=points1.shape[0])
    recons = bilinear_interpolate(img, H, target_size=None)
    #recons = cv2.warpPerspective(img, H, dsize=(img.shape[1], img.shape[0]))
    print(recons.shape)
    vis_img(cv2.cvtColor(recons,cv2.COLOR_BGR2RGB))
if __name__ == '__main__':
    args = argparse2()
    main(args)