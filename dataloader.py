import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        label : per point label ,[N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    # label = label[centroids.astype(np.int32)]
    return point  # ,label

def Minkowski_distance(src, dst, p):
    """
    Calculate Minkowski distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point Minkowski distance, [B, N, M]
    """
    return torch.cdist(src,dst,p=p)

def knn_idx(xyz, new_xyz,K=3,p=2):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        (group_dist,group_idx): (grouped points distance, [B, S, K], grouped points index, [B, S, K])
    """
    if new_xyz==None:
        new_xyz=xyz
    new_xyz = new_xyz.permute(0,2,1)
    xyz = xyz.permute(0,2,1)
    sqrdists = Minkowski_distance(src=new_xyz, dst=xyz, p=p)
    group_dist, group_idx = torch.topk(sqrdists, K, dim=-1, largest=False, sorted=True)
    return group_dist, group_idx[:,:,1:]

def gather_idx(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNet1(Dataset):
    # def __init__(self, path="/workspace/VAE/chair_data", train_folder=False, npoints=1024):
    def __init__(self, path="/Users/surajreddy/Documents/3DVSS/VAE/chair_data", train_folder=False, npoints=1024):
        super(ModelNet1, self).__init__()
        if train_folder:
            self.data = np.load(os.path.join(path,"data_train_889.npy"))
        else:
            self.data = np.load(os.path.join(path,"data_test_100.npy"))
        self.npoints = npoints
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pc = self.data[index]
        pc = farthest_point_sample(pc, self.npoints)
        pc = pc_normalize(pc)
        return pc