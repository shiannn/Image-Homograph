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
    recons = cv2.warpPerspective(img, H, dsize=(img.shape[1], img.shape[0]))
    print(recons.shape)
    vis_img(cv2.cvtColor(recons,cv2.COLOR_BGR2RGB))
if __name__ == '__main__':
    args = argparse2()
    main(args)