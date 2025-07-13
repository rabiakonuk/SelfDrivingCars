"""
Code written by Joey Wilson, 2023.
"""

from sys import _xoptions
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset


class PointLoader(Dataset):
  def __init__(self, root, remap_array, data_split="Train",
              device="cpu", max_size=40000):
    self.root = root
    self.velo_dir = os.path.join(root, "velodyne_ds")
    self.velo_files = sorted(os.listdir(self.velo_dir))
    self.split = data_split
    if not self.split == "Test":
      self.label_dir = os.path.join(root, "labels_ds")
      self.label_files = sorted(os.listdir(self.label_dir))
    self.device = device
    self.remap_array = remap_array
    self.max_size = max_size
  
  def __len__(self):
    return len(self.velo_files)

  def __getitem__(self, index):
    # Load point cloud
    velo_file = os.path.join(self.velo_dir, self.velo_files[index])
    # TODO: Fetch the point cloud
    pc = None
    # Test set has no available ground truth
    if self.split == "Test":
      return pc, None
    # Load labels
    label_file = os.path.join(self.label_dir, self.label_files[index])
    # TODO: Fetch labels from label_file
    label = None
    # TODO: Mask the label with a mask of 0xFFFF
    # TODO: Use the masked label to index the remap array
    label = None
    # Downsample the points
    if self.split == "Train":
      indices = np.random.permutation(pc.shape[0])[:self.max_size]
      pc = pc[indices, :]
      label = label[indices]
    return pc, label

  def collate_fn(self, data):
    B = len(data)
    pc_numpy = [data[i][0] for i in range(B)]
    torch_pc = torch.tensor(np.array(pc_numpy), device=self.device)
    label_numpy = [data[i][1] for i in range(B)]
    torch_label = torch.tensor(np.array(label_numpy), device=self.device)
    return torch_pc, torch_label 


class PointNetEncoder(nn.Module):
  def __init__(self, cs, linear_out=None):
    super().__init__()
    # cs is a list of channel sizes e.g. [3, 64, num_classes]
    # Each layer i contains a linear layer from 
    # cs [i-1] to cs[i]
    # and a ReLU nonlinearity 
    # linear_out is in the case this is the last layer of the network
    self.net = torch.nn.Sequential()
    for i in range(1, len(cs)):
      # TODO: Replace None with a linear layer from cs[i-1] to cs[i]
      self.net.add_module("Lin" + str(i), None)
      # TODO: Replace None with a batchnorm layer of size cs[i]
      self.net.add_module("Bn" + str(i), None)
       #TODO: Replace None with a ReLU Layer
      self.net.add_module("ReLU" + str(i), None)
    if linear_out is not None:
      # TODO: Replace None with a linear layer from cs[i] to linear_out
      self.net.add_module("LinFinal", nn.Linear(cs[i], linear_out))

  def forward(self, x):
    # Input x is a BxNxC matrix where N is number of points
    B, N, C = x.shape
    # TODO: Use x.view() to reshape x into (B*N, C)
    x = None
    # TODO: Feed x through your network
    x = None
    # TODO: Reshape x into shape (B, N, C1) where C1 is the new channel dim
    x = None
    return x
    
# This module learns to combines global and local point features
class PointNetModule(nn.Module):
  def __init__(self, cs_en, cs_dec, num_classes=20):
    super().__init__()
    # TODO: Create a PointNetEncoder with cs=cs_en
    self.enc = None
    # TODO: Create a PointNetEncoder decoder module with cs=cs_dec
    # and linear_out=num_classes
    self.dec = None

  def forward(self, x):
    B, N, C = x.shape
    # Encoder
    # TODO: Feed x through the PointNetEncoder
    point_feats = None
    # Max across points
    # TODO: Use max pooling across the point dimension to create
    # a global_feats tensor of shape (B, C1)
    # We used the torch.max
    global_feats = None
    # TODO: Reshape global_feats from (B, C1) to (B, 1, C1)
    # And repeat along the middle dimension N times using repeat
    # End tensor should be (B, N, C1)
    global_feats = None
    # TODO: Concatenate local and global features along channel dimension (2)
    joint_feats = None
    # TODO: Feed joint_feats through the decoder module
    out = None
    return out
  
'''
## T-Net
Next, we will add the transformation network or T-Net to our PointNet module. The T-Net operates very similarly to the PointNet module, however it only operates on the global features to create a global 3x3 transformation matrix.

In the initialization function, create a `PointNetEncoder` MLP to encode and decode the transformation. Also create a `PointNetModule` for the joint encoding operation. In the forward pass:


1.   Pass the input through the transformation encoder
2.   Apply max pooling to obtain global features.
3. Pass the global features through the transformation decoder to obtain the transformation.
4. Reshape the transfomation to Bx3x3  and add an identity matrix. The identity matrix creates a possible skip connection.
5. Apply the transformation to the input points to obtain transformed points, and feed the transformed points through the `PointNetModule` joint encoder.

Fill in `PointNetFull`. If the T-Net is implemented correctly, the following cell will return `True`.
'''
    
# This module adds a T-Net
class PointNetFull(nn.Module):
  def __init__(self, cs_en, cs_dec, cs_t_en, cs_t_dec):
    super().__init__()
    # TODO: Create a PointNetEncoder with cs=cs_t_en
    self.t_enc = None
    # TODO: Create a PointNetEncoder with cs=cs_t_dec and linear_out = 9
    # Note that 9 comes from the product 3x3
    self.t_dec = None
    # TODO: Create a PointNetModule with cs_en, and cs_dec
    self.joint_enc = None

  def forward(self, x):
    B, N, C = x.shape
    # T-Net
    # TODO: Feed x through the t-net encoder
    t_feats = None
    # TODO: Max pool across the point dimension of t_feats
    t_feats = None
    # TODO: Reshape t_feats to shape (B, 1, C1)
    t_feats = None
    # TODO: Feed t_feats through the T-Net Decoder
    t_feats = None
    # TODO: Reshape t_feats to (B, 3, 3)
    point_trans = None
    # TODO: Compute the transformation (B, 3, 3) matrices
    # As a summation of point_trans and an identity matrix
    point_trans = None
    # Apply transform
    # TODO: Perform batched matrix multiplication between x and point_trans
    # torch.bmm() may be helpful
    transformed_points = None
    # Joint Encoder
    # TODO: Feed the transformed_points through the joint encoder
    output_preds = None
    return output_preds


# Compute the per-class iou and miou
def IoU(targets, predictions, num_classes, ignore_index=0):
  intersections = torch.zeros(num_classes, device=targets.device)
  unions = torch.zeros_like(intersections)
  counts = torch.zeros_like(intersections)
  # TODO: Discard ignored points
  valid_mask = None
  targets = targets[valid_mask]
  predictions = predictions[valid_mask]
  # Loop over classes and update the counts, unions, and intersections
  for c in range(num_classes):
    # TODO: Fill in computation
    # Add small value to avoid division by 0
    # Make sure to keep the small smoothing constant to match the autograder
    unions[c] = unions[c] + 0.00001
  # Per-class IoU
  # Make sure to set iou for classes with no points to 1
  iou = None
  # Calculate mean, ignoring ignore index
  miou = None
  return iou, miou

    

