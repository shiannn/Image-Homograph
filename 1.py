import sys
import numpy as np
import cv2 as cv
from utils import normalized, to_homogeneous, vis_img
from DirectLT import normalized_DLT, DLT
import argparse

def argparse1():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i1", "--input_img1", type=str,
        help="input document image for rectification"
    )
    parser.add_argument(
        "-i2", "--input_img2", type=str,
        help="output document image after rectification"
    )
    parser.add_argument(
        "-g", "--ground_truth", type=str,
        help="ground truth points"
    )
    parser.add_argument(
        "-n", "--normalize", action='store_true',
        help="whether normalize DLT"
    )
    parser.add_argument(
        "-v", "--visualize", action='store_true',
        help="whether visualize"
    )
    args = parser.parse_args()
    
    return args

def get_sift_correspondences(img1, img2):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    return points1, points2, kp1, kp2, good_matches

def main(args):
    img1 = cv.imread(args.input_img1)
    img2 = cv.imread(args.input_img2)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
    gt_correspondences = np.load(args.ground_truth)
    if args.normalize:
        norm = True
    else:
        norm = False
    
    points1, points2, kp1, kp2, good_matches = get_sift_correspondences(img1, img2)
    for k_sample in [4,8,20]:
        if args.visualize:
            img_draw_match = cv.drawMatches(
                img1, kp1, img2, kp2, 
                good_matches[:k_sample], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            vis_img(img_draw_match, size=(10,10))
        if norm:
            H = normalized_DLT(points1, points2, k=k_sample)
        else:
            H = DLT(points1, points2, k=k_sample)
        #H, _ = cv.findHomography(points1[:k_sample], points2[:k_sample], cv.RANSAC,5.0)

        ps = to_homogeneous(gt_correspondences[0])
        pt = to_homogeneous(gt_correspondences[1])
        Hps = np.matmul(H, ps.T).T
        Hps = Hps / Hps[:,2:]

        if args.visualize:
            rows, cols, chs = img1.shape
            recons = cv.warpPerspective(img1, H, dsize=(img2.shape[1], img2.shape[0]))
            print(recons.shape)
            if recons.shape[0] != img2.shape[0]:
                recons = cv.resize(recons, dsize=(recons.shape[1], img2.shape[0]))
            
            show_img = np.concatenate([recons, img2], axis=1)
            
            vis_img(show_img,size=(10,10))
        verbose = 'k_sample: {} error: {}'.format(
            k_sample,
            np.sqrt(((Hps - pt)**2).sum(axis=1)).mean(axis=0)
        )
        print(verbose)
if __name__ == '__main__':
    args = argparse1()
    main(args)