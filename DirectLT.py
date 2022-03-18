import numpy as np
from utils import normalized

def normalized_DLT(points1, points2, k=4):
    normalized_points1, transform1, _ = normalized(points1)
    normalized_points2, _, inv_transform2 = normalized(points2)
    #print(normalized_points1)
    H = DLT(
        normalized_points1,
        normalized_points2, k=k
    )
    H2 = np.matmul(
        inv_transform2,
        np.matmul(H, transform1)
    )
    return H2

def DLT(points1, points2, k=4):
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