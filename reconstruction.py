import os

import numpy as np
import trimesh
from scipy.spatial import KDTree
from skimage import measure
import torch
import torch.nn.functional as F
from torch import einsum, nn

from tqdm import tqdm

import argparse

def generate_grid(point_cloud, res):
    """Generate grid over the point cloud with given resolution
    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    coords = np.mgrid[:res, :res, :res]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    length += length / res
    coords_matrix[0, 0] = length[0] / res
    coords_matrix[1, 1] = length[1] / res
    coords_matrix[2, 2] = length[2] / res
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.T

    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples):
    """Predict occupancy of values batch-wise
    Args:
        points (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_samples (int): number of points to evaluate at once
    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """

    num_pts = points.shape[0]
    occ = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        occ[i * num_samples : i * num_samples + num_samples] = eval_func(
            points[i * num_samples : i * num_samples + num_samples]
        ).detach().numpy()
    if num_pts % num_samples:
        occ[num_batches * num_samples :] = eval_func(
            points[num_batches * num_samples :]
        ).detach().numpy()

    return occ


def eval_grid(coords, eval_func, num_per_sample=1024):
    """Predict occupancy of values on a grid
    Args:
        coords (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_per_sample (int): number of points to evaluate at once

    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """
    coords = coords.reshape([-1, 3])
    occ = batch_eval(coords, eval_func, num_samples=num_per_sample)
    return occ


def reconstruct(model, grid, res, transform):
    """Reconstruct mesh by predicting occupancy values on a grid
    Args:
        model (function): function that takes a batch of points and returns occupancy values
        grid (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
        transform (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]

    Returns:
        verts (np.array, [M, 3]): 3D coordinates of M vertices
        faces (np.array, [K, 3]): indices of K faces
    """

    occ = eval_grid(grid, model)
    occ = occ.reshape([res, res, res])

    verts, faces, normals, values = measure.marching_cubes(occ, 0.0)
    verts = np.matmul(transform[:3, :3], verts.T) + transform[:3, 3:4]
    verts = verts.T

    return verts, faces


def compute_metrics(reconstr_path, gt_path, num_samples=1000000):
    """Compute chamfer and hausdorff distances between the reconstructed mesh and the ground truth mesh
    Args:
        reconstr_path (str): path to the reconstructed mesh
        gt_path (str): path to the ground truth mesh
        num_samples (int): number of points to sample from each mesh

    Returns:
        chamfer_dist (float): chamfer distance between the two meshes
        hausdorff_dist (float): hausdorff distance between the two meshes
    """
    reconstr = trimesh.load(reconstr_path)
    gt = trimesh.load(gt_path)

    # sample points on the mesh surfaces using trimesh
    reconstr_pts = reconstr.sample(num_samples)
    gt_pts = gt.sample(num_samples)

    # compute chamfer distance between the two point clouds
    reconstr_tree = KDTree(reconstr_pts)
    gt_tree = KDTree(gt_pts)
    dist1, _ = reconstr_tree.query(gt_pts)
    dist2, _ = gt_tree.query(reconstr_pts)
    chamfer_dist = (dist1.mean() + dist2.mean()) / 2
    hausdorff_dist = max(dist1.max(), dist2.max())

    return chamfer_dist, hausdorff_dist


if __name__ == "__main__":
    #from model import Baseline
    from model import Baseline, DenseGrid, SimpleModel, HashGrid

    parser = argparse.ArgumentParser()

    parser.add_argument('--GridType', type = str, required = True)
    parser.add_argument('--Resolution', type = int)
    parser.add_argument('--CoarseResolution', type = int)
    parser.add_argument('--FineResolution', type = int)
    parser.add_argument('--LoD', type = int)
    parser.add_argument('--BandWidth', type = int)
    parser.add_argument('--in_dim', type = int)
    parser.add_argument('--hidden_dim', type = int)
    parser.add_argument('--hidden_layers', type = int)
    parser.add_argument('--ReconstructionResolution', type = int)

    args=parser.parse_args()

    for cur_obj in os.listdir("processed"):
        pc = trimesh.load(f"processed/{cur_obj}")
        verts = np.array(pc.vertices)
        gt_occ = np.array(pc.visual.vertex_colors)[:, 0]
        gt_occ = (gt_occ == 0).astype("float32") * -2 + 1
        
        if(args.GridType=='Single'):
            grid = DenseGrid(args.Resolution, 1, 'trilinear', args.in_dim)  
        elif(args.GridType=='Multi'):
            grid = DenseGrid(args.CoarseResolution, args.LoD, 'trilinear', args.in_dim)
        elif(args.GridType=='Hash'):
            grid = HashGrid(args.CoarseResolution, args.FineResolution, args.LoD, args.BandWidth, args.in_dim)
        else:
            raise NotImplementedError('Wrong Grid Type.')
        model = SimpleModel(grid, args.in_dim, args.hidden_dim, 1, args.hidden_layers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(model)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        prev_loss=100000000.0
        loop = tqdm(range(1500))
        for epoch in loop:
            output = model(torch.from_numpy(verts).to(device))

            loss_gt = loss_fn(output, torch.from_numpy(gt_occ).to(device))
            # Sample random points in [-1, 1]^3. The number of points: N_reg_points (to be tuned)
            
            N_reg_points = 10000
            shift_scale = 0.15
            
            reg_coords = torch.rand(N_reg_points, 3) * 2 - 1
            reg_coords = reg_coords.to(device)

            # Sample random shifts for the points
            shift = torch.randn(N_reg_points, 1) * shift_scale
            shift = shift.to(device)

            # Compute the registration loss
            reg_pred = model(reg_coords)
            shift_reg_pred = model(torch.clip(reg_coords + shift, -1, 1))
            reg_loss = torch.mean(torch.abs(reg_pred - shift_reg_pred))

            # Compute the total loss
            reg_weight = 0.02
            loss = (
                loss_gt + reg_loss * reg_weight
            )  # loss_gt - the loss over the ground truth points
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch: {epoch}")
            loop.set_postfix_str(
                f"Loss: {loss.item():.5f}"
            )
            if (prev_loss - loss < 0.00001 and loss < 0.05) or loss < 0.01:
                break
            prev_loss = loss

        model = model.to('cpu')
        resolution = args.ReconstructionResolution
        grid, transform = generate_grid(verts, res=resolution)
        rec_verts, rec_faces = reconstruct(model, torch.from_numpy(grid), resolution, transform)
        reconstr_path = f"reconstructions/{cur_obj}"
        os.makedirs(os.path.dirname(reconstr_path), exist_ok=True)
        trimesh.Trimesh(rec_verts, rec_faces).export(reconstr_path)

        gt_path = f"data/{cur_obj}"

        chamfer_dist, hausdorff_dist = compute_metrics(
            reconstr_path, gt_path, num_samples=1000000
        )

        print(cur_obj)
        print(f"Chamfer distance: {chamfer_dist:.4f}")
        print(f"Hausdorff distance: {hausdorff_dist:.4f}")
        print("##################")
