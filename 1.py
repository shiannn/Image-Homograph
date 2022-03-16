import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
    
    good_matches = good_matches[:5]
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #print(img_draw_match)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img_draw_match)
    plt.show()
    #cv.imshow('match', img_draw_match)
    #cv.waitKey(0)
    #cv2.destroyAllWindows() 
    return points1, points2

def homography_estimation(points1, points2, k=4):
    points1, points2 =points1[:k], points2[:k]
    def build_matrix(points1, points2):
        A = np.zeros((2*points1.shape[0], 9))
        for idx, (p1, p2) in enumerate(zip(points1, points2)):
            u1, v1 = p1
            u2, v2 = p2
            row1 = np.array([0,0,0,-u1,-v1,-1,v2*u1, v2*v1, v2])
            row2 = np.array([u1, v1, 1, 0,0,0, -u2*u1, -u2*v1, -u2])
            A[2*idx] = row1
            A[2*idx+1] = row2
        return A
    def solve_rayleigh(A):
        eigval, eigvec = np.linalg.eig(np.matmul(A.T, A))
        idx = eigval.real.argsort()[::-1]   
        eigval = eigval[idx]
        eigvec = eigvec[idx]
        #print(eigval)
        #print(eigvec)
        return eigvec[-1]
    
    def solve_rayleigh_svd(A):
        u, s, v = np.linalg.svd(A)
        #print(v[-1])
        return v[-1]
        
    A = build_matrix(points1, points2)
    h = solve_rayleigh_svd(A)
    res = np.matmul(A, h).mean()
    assert res < 1e-4
    H = h.reshape(3,3)
    #print(H)
    return H

def to_homogeneous(points_uv):
    homo_points = np.concatenate([points_uv, np.ones((points_uv.shape[0], 1))], axis=1)
    return homo_points

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    
    points1, points2 = get_sift_correspondences(img1, img2)
    
    for k_sample in [4,8,20,1000]:
        H = homography_estimation(points1, points2, k=k_sample)
        ps = to_homogeneous(gt_correspondences[0])
        pt = to_homogeneous(gt_correspondences[1])
        #print(ps.shape)
        #pt = gt_correspondences[1]
        Hps = np.matmul(H, ps.T)
        Hps = Hps.T / Hps.T[:,2:]
        #print(Hps.T)
        #print(Hps.T / Hps.T[:,2:])
        print(np.sqrt(((Hps - pt)**2).sum(axis=1)).mean(axis=0))
        rows, cols, chs = img1.shape
        recons = cv.warpPerspective(img1, H, dsize=(img1.shape[1], img1.shape[0]))
        print(recons.shape)

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(121)
        ax.imshow(recons)
        ax = fig.add_subplot(122)
        ax.imshow(img2)
        plt.show()