import os
from dataloader import ModelNet10
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from datasets.modelnet40 import ModelNet40_NPY
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.cluster import MeanShift
from Model import PointNet

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
    new_xyz = new_xyz.permute(0,2,1)
    xyz = xyz.permute(0,2,1)
    sqrdists = Minkowski_distance(src=new_xyz, dst=xyz, p=p)
    group_dist, group_idx = torch.topk(sqrdists, K, dim=-1, largest=False, sorted=True)
    return group_dist,group_idx

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
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

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

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


model = PointNet()
checkpoint = torch.load("PointNet_max_LOGS/Model/best_model.pth", map_location='cpu')

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

path = "../"
file = "chair_1.ply"

with open(os.path.join(path, file), 'rb') as f:
    plydata = PlyData.read(f)

x = np.expand_dims(plydata.elements[0].data['x'], axis=-1)
y = np.expand_dims(plydata.elements[0].data['y'], axis=-1)
z = np.expand_dims(plydata.elements[0].data['z'], axis=-1)
pc = np.hstack([x, y, z])

pc = np.array(pc)
pc=farthest_point_sample(pc,1024)
pc = pc_normalize(pc)

pc = torch.from_numpy(pc).unsqueeze(0)

pc = pc.permute(0,2,1)
print(pc.shape)
pred, feat, _,_ = model(pc)
k=[]
y=max(pred[0])
for i in pred[0]:
    if i==y:
        k.append(1)
    else:
        k.append(0)
print(k)
with torch.no_grad():
    # PC = torch.from_numpy(pc).unsqueeze(0).permute(0,2,1)
    PC = pc
    _,ID = knn_idx(PC,PC, K=128) 
    newP = gather_idx(PC.permute(0,2,1), ID[:,:,1:])
    print("NEWP",newP.shape)
    meanP = torch.mean(newP, dim=2)
    print("mean",meanP.shape)
    print(PC.shape)
    query = ((torch.sum((meanP - PC.permute(0,2,1))**2, dim=-1))**0.5)

    dis,ID = query.topk(k=1024)
    new_PC = gather_idx(PC.permute(0,2,1), ID)  # B N C


# _, feat, _,_ = model(pc)
# _,idx = knn_idx(xyz=feat, new_xyz=feat, K=250)
# t = ID[0][350].detach().numpy()
# id = idx[0][t][:].detach().cpu().numpy()
# pc = pc.permute(0,2,1)
# pc = pc[0].detach().cpu().numpy()
# ax = plt.axes(projection="3d")
# ax.scatter(pc[:,0],pc[:,1],pc[:,2],s=0.2)
# ax.scatter(pc[id,0],pc[id,1],pc[id,2],s=1, c="red")
# ax.scatter(pc[t,0],pc[t,1],pc[t,2],s=10,c="black", marker="X")
# ax.axis(False)
# # print("Show")
# plt.show()
