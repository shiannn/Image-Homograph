import numpy as np

def vis_img(img, size=(10,10)):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

def normalized(points):
    mean, std = points.mean(axis=0), points.std()
    inv_transform = np.array([
        [std/np.sqrt(2), 0, mean[0]],
        [0, std/np.sqrt(2), mean[1]],
        [0, 0, 1]
    ])
    transform = np.linalg.inv(inv_transform)
    homo_points = to_homogeneous(points)
    normalized_points = np.matmul(transform, homo_points.T)
    normalized_points = normalized_points.T[:,:2]
    
    return normalized_points, transform, inv_transform

def to_homogeneous(points_uv):
    homo_points = np.concatenate([points_uv, np.ones((points_uv.shape[0], 1))], axis=1)
    return homo_points