import sys
import numpy as np
import cv2 as cv
from utils import normalized, to_homogeneous, vis_img
from DirectLT import normalized_DLT, DLT

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
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return points1, points2, img_draw_match

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    if sys.argv[4] == 'norm':
        norm = True
    else:
        norm = False
    
    points1, points2, img_draw_match = get_sift_correspondences(img1, img2)

    vis_img(cv.cvtColor(img_draw_match,cv.COLOR_BGR2RGB))

    for k_sample in [4,8,20]:
        if norm:
            H = normalized_DLT(points1, points2, k=k_sample)
        else:
            H = DLT(points1, points2, k=k_sample)

        ps = to_homogeneous(gt_correspondences[0])
        pt = to_homogeneous(gt_correspondences[1])
        Hps = np.matmul(H, ps.T).T
        Hps = Hps / Hps[:,2:]

        print(np.sqrt(((Hps - pt)**2).sum(axis=1)).mean(axis=0))
        rows, cols, chs = img1.shape
        recons = cv.warpPerspective(img1, H, dsize=(img1.shape[1], img1.shape[0]))
        
        vis_img(
            cv.cvtColor(
            np.concatenate([recons, img2], axis=1),cv.COLOR_BGR2RGB)
        )