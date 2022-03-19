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

def get_new_grid(img, H):
    def get_valid_region(img, grid):
        #print(grid)
        is_valid = (
            (grid[:,:,0] <= img.shape[1]-1) & \
            (grid[:,:,0] >= 0) & \
            (grid[:,:,1] <= img.shape[0]-1) & \
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

    grid[:,:,[0,1]] = grid[:,:,[1,0]]
    new_grid = np.matmul(backward_H, grid.reshape(-1,3).T).T
    new_grid = new_grid.reshape(-1, grid.shape[1], 3)
    new_grid = new_grid / new_grid[:,:,2:]
    
    new_grid_uv = new_grid[:,:,:2] ### u, v
    valid_region = get_valid_region(img, new_grid_uv)
    new_grid_uv[~valid_region] = 0

    return new_grid_uv

def nearest_interpolate(img, H, target_size=None):
    new_grid_uv = get_new_grid(img, H)
    new_grid_uv = new_grid_uv.astype(int)
    ret_img = img[(new_grid_uv[:,:,1],new_grid_uv[:,:,0])]
    
    return ret_img

def bilinear_interpolate(img, H, target_size=None):
    new_grid_uv = get_new_grid(img, H)
    us = new_grid_uv[:,:,0]
    vs = new_grid_uv[:,:,1]
    right = np.ceil(us)
    left = np.floor(us)
    down = np.ceil(vs)
    up = np.floor(vs)
    lu = np.stack([left, up], axis=2).astype(int)
    ld = np.stack([left, down], axis=2).astype(int)
    ru = np.stack([right, up], axis=2).astype(int)
    rd = np.stack([right, down], axis=2).astype(int)
    area_lu = np.abs(new_grid_uv - lu).prod(axis=2, keepdims=True)
    area_ld = np.abs(new_grid_uv - ld).prod(axis=2, keepdims=True)
    area_ru = np.abs(new_grid_uv - ru).prod(axis=2, keepdims=True)
    area_rd = np.abs(new_grid_uv - rd).prod(axis=2, keepdims=True)
    
    for area in [area_lu, area_ld, area_ru, area_rd]:
        assert area.max() <= 1.0 and area.min() >= 0.0
    
    weights = [area_rd, area_ru, area_ld, area_lu]
    neighbors = [
        img[lu[:,:,1], lu[:,:,0]],
        img[ld[:,:,1], ld[:,:,0]],
        img[ru[:,:,1], ru[:,:,0]],
        img[rd[:,:,1], rd[:,:,0]]
    ]
    ret_img = np.zeros_like(img)
    for weight, neighbor in zip(weights, neighbors):
        ret_img = ret_img + weight* neighbor
    #ret_img = img[(new_grid_uv[:,:,1],new_grid_uv[:,:,0])]
    #weight_sum = np.stack(weights, axis=2).sum(axis=2)
    #ret_img = ret_img.astype(float)
    #ret_img = ret_img.astype(int)
    #ret_img = ret_img.astype(np.float32)
    
    return ret_img

def main(args):
    img = cv2.imread(args.input_img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
    vis_img(show_img)
    H = normalized_DLT(points1, points2, k=points1.shape[0])
    recons = bilinear_interpolate(img, H, target_size=None)
    #recons = nearest_interpolate(img, H, target_size=None)
    #recons = cv2.warpPerspective(
    #    img, H, dsize=(img.shape[1], img.shape[0]),
    #    flags=cv2.INTER_CUBIC
    #)
    print(recons.shape)
    #vis_img(recons)
    vis_img(recons.astype(int))
if __name__ == '__main__':
    args = argparse2()
    main(args)