from scipy.spatial import KDTree
import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn

class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))

def trilinear_interpolation(res, grid, points, grid_type):
    """
    Performs bilinear interpolation of points with respect to a grid.

    Parameters:
        grid (numpy.ndarray): A 2D numpy array representing the grid.
        points (numpy.ndarray): A 2D numpy array of shape (n, 2) representing
            the points to interpolate.

    Returns:
        numpy.ndarray: A 1D numpy array of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = (points[:, :, 0] + 1.0) * 0.5 * (res - 1) # point coordination is initially in [-1, 1]. First shift and rescale it to [0, 1], then convert to the grid index
    y = (points[:, :, 1] + 1.0) * 0.5 * (res - 1)
    z = (points[:, :, 2] + 1.0) * 0.5 * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
    z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()
    z2 = torch.clip(z1 + 1, 0, res - 1).int()

    # Compute the weights for each of the four points
    w1 = (x2 - x) * (y2 - y) * (z2 - z)
    w2 = (x - x1) * (y2 - y) * (z2 - z)
    w3 = (x2 - x) * (y - y1) * (z2 - z)
    w4 = (x - x1) * (y - y1) * (z2 - z)
    w5 = (x2 - x) * (y2 - y) * (z - z1)
    w6 = (x - x1) * (y2 - y) * (z - z1)
    w7 = (x2 - x) * (y - y1) * (z - z1)
    w8 = (x - x1) * (y - y1) * (z - z1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (z1 * res * res + y1 * res + x1).long()
        id2 = (z1 * res * res + y1 * res + x2).long()
        id3 = (z1 * res * res + y2 * res + x1).long()
        id4 = (z1 * res * res + y2 * res + x2).long()
        id5 = (z2 * res * res + y1 * res + x1).long()
        id6 = (z2 * res * res + y1 * res + x2).long()
        id7 = (z2 * res * res + y2 * res + x1).long()
        id8 = (z2 * res * res + y2 * res + x2).long()

    elif grid_type == "HASH":
        npts = res**3
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id5 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id6 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id7 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id8 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
        else:
            id1 = (z1 * res * res + y1 * res + x1).long()
            id2 = (z1 * res * res + y1 * res + x2).long()
            id3 = (z1 * res * res + y2 * res + x1).long()
            id4 = (z1 * res * res + y2 * res + x2).long()
            id5 = (z2 * res * res + y1 * res + x1).long()
            id6 = (z2 * res * res + y1 * res + x2).long()
            id7 = (z2 * res * res + y2 * res + x1).long()
            id8 = (z2 * res * res + y2 * res + x2).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
        + torch.einsum("ab,abc->abc", w5, grid[(id5).long()])
        + torch.einsum("ab,abc->abc", w6, grid[(id6).long()])
        + torch.einsum("ab,abc->abc", w7, grid[(id7).long()])
        + torch.einsum("ab,abc->abc", w8, grid[(id8).long()])
    )
    return values[0]

class DenseGrid(nn.Module):
    def __init__(self, base_lod=4, num_lod=5, interpolation_type="closest", feat_dim=3):
        super().__init__()
        self.feat_dim = feat_dim  # feature dim size
        self.codebook = nn.ParameterList([])
        self.interpolation_type = interpolation_type  # bilinear

        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]
        print("LODS:", self.LODS)
        self.init_feature_structure()
        self.output_layer=nn.ReLU()

    def init_feature_structure(self):
        for LOD in self.LODS:
            ############ TODO: YOUR CODE HERE ############
            # TASK: Construct the feature grid for each level of detail
            # HINTS:

            # 1. Create a feature grid of size (LOD*LOD, 1) with self.feat_dim channels
            # 2. Initialize the feature grid with normal distribution and variance 0.01
            # 3. Append the feature grid to self.codebook. Use nn.Parameter to wrap the feature grid

            mean = torch.zeros(LOD**3, self.feat_dim) # or (LOD**2, 1, self.feat_dim)?
            fts = nn.Parameter(torch.normal(mean,0.01)) # variance = 0.01 => standard deviation = 0.1

            # Dummy code
            #fts = nn.Parameter(torch.zeros(LOD**2, self.feat_dim),requires_grad=True)

            ############ END OF YOUR CODE ############
            self.codebook.append(fts)

    def forward(self, pts):
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):

            if self.interpolation_type == "closest":

                ############ TODO: YOUR CODE HERE ############
                # TASK: For each point in pts, find the closest feature in the feature grig
                # HINTS:

                # 1. Find x,y index of closest grid node. You can use torch.floor to round down
                # E.g point coordinate is [0.09, 0.23] and resolution is 16, then x = 1, y = 3

                x = torch.floor((pts[:,0] + 1.0) * 0.5 * (res - 1) + 0.5)
                y = torch.floor((pts[:,1] + 1.0) * 0.5 * (res - 1) + 0.5)
                z = torch.floor((pts[:,2] + 1.0) * 0.5 * (res - 1) + 0.5)

                # Dummy code
                #x = torch.zeros(pts.shape[0])
                #y = torch.zeros(pts.shape[0])
                ############ END OF YOUR CODE ############

                features = self.codebook[i][(x + y * res + z * res * res).long()]
            elif self.interpolation_type == "trilinear":

                ############ TODO: YOUR CODE HERE ############
                # TASK: Apply bilinear interpolation to get the feature for each point in pts
                # HINTS:

                # 1. Use bilinear_interpolation function from utils.py to get the feature for each point in pts
                # Note, use "NGLOD" as the interpolation type

                features = trilinear_interpolation(res,self.codebook[i],pts,"NGLOD")

                # Dummy code
                #features = torch.zeros(pts.shape[0], self.feat_dim)
                ############ END OF YOUR CODE ############

            else:
                raise NotImplementedError

            features=self.output_layer(features)
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)


class SimpleModel(nn.Module):
    def __init__(
        self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers=1
    ):
        super().__init__()
        ############ TODO: YOUR CODE HERE ############
        # TASK: Implement a simple MLP with `num_hidden_layers` hidden layers
        # HINTS: - Use torch.nn.Linear to create linear layers
        #        - You can use either torch.nn.ReLU as activation functions

        # Dummy code
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Linear(input_dim, hidden_dim, bias=True))
        for layer in range(num_hidden_layers):
            self.module_list.append(torch.nn.ReLU())
            if(layer < num_hidden_layers - 1):
                self.module_list.append(torch.nn.Linear(hidden_dim, hidden_dim, bias=True))
            else:
                self.module_list.append(torch.nn.Linear(hidden_dim, output_dim, bias=True))
        #self.module_list.append(torch.nn.Linear(hidden_dim,output_dim,bias=True))
        #self.module_list.append(torch.nn.Linear(input_dim, output_dim, bias=True))
        ############ END OF YOUR CODE ############
        self.model = torch.nn.Sequential(*self.module_list)
        self.grid_structure = grid_structure

    def forward(self, coords):
        ############ TODO: YOUR CODE HERE ############
        # TASK: Implement forward pass of the model
        # HINTS:

        # 1. Transform points in a right format for the grid structure
        # * Note that coords is a tensor of shape (H, W, 2) where H and W are the height and width of the image
        # * You need to reshape it to a tensor of shape (H*W, 2)
        # coords = ...
        
        coords=coords.reshape(-1,3)
        coords=coords.to(torch.float32)

        # 2. Pass the reshaped tensor to the grid structure to get points features
        # feat = ...

        feat=self.grid_structure(coords)

        # 3. Pass the features to the model to get the output (prediction of color values)
        # out = ...
        out=self.model(feat)

        # 4. Reshape the output back to size [H, W, C]

        out=out.squeeze(-1)

        # Dummy code
        #h, w, c = coords.shape
        #out = torch.zeros(h, w, 3)
        ############ END OF YOUR CODE ############

        return out

class HashGrid(nn.Module):
    def __init__(self, min_grid_res=6, max_grid_res=64, num_LOD=10, band_width=19, feat_dim=3):
        super().__init__()
        self.feat_dim = feat_dim  # feature dim size
        self.codebook = nn.ParameterList([])
        self.codebook_size = 2**band_width

        b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_LOD - 1))
        self.LODS = [int(1 + np.floor(min_grid_res * (b**l))) for l in range(num_LOD)]
        print("LODS:", self.LODS)
        self.init_hash_structure()
        self.output_layer=nn.ReLU()

    def init_hash_structure(self):
        for LOD in self.LODS:
            ############ TODO: YOUR CODE HERE ############
            # TASK: Construct the feature grid for each level of detail
            # HINTS:

            # 1. Create a feature grid of size (min(LOD*LOD, self.codebook_size), 1) with self.feat_dim channels
            # 2. Initialize the feature grid with normal distribution and variance 0.01
            # 3. Append the feature grid to self.codebook. Use nn.Parameter to wrap the feature grid

            mean = torch.zeros(min(LOD**3,self.codebook_size), self.feat_dim)
            fts = nn.Parameter(torch.normal(mean,0.01))

            # Dummy code
            #fts = nn.Parameter(torch.zeros(LOD**2, self.feat_dim))

            ############ END OF YOUR CODE ############
            self.codebook.append(fts)

    def forward(self, pts):
        _, feat_dim = self.codebook[0].shape
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):
            ############ TODO: YOUR CODE HERE ############
            # TASK: Apply bilinear interpolation to get the feature for each point in pts
            # HINTS:

            # 1. Use bilinear_interpolation function from utils.py to get the feature for each point in pts
            # Note, use "HASH" as the interpolation type

            features = trilinear_interpolation(res,self.codebook[i],pts,"HASH")

            # Dummy code
            #features = torch.zeros(pts.shape[0], self.feat_dim)
            ############ END OF YOUR CODE ############
            features = self.output_layer(features) # ReLU()
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)