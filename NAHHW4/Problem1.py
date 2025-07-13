"""
Code written by Joey Wilson, 2023.
"""

from sys import _xoptions
import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn.modules.activation import F
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

    # Load point cloud -- update the file name
    velo_file = os.path.join(self.velo_dir, self.velo_files[index])

    # Adjust this line based on the actual file format
    # pc = np.load(velo_file, allow_pickle=True)
    # check the shape that matches with data format
    pc = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    # TODO: Fetch the point cloud
    # Test set has no available ground truth
    if self.split == "Test":
      return pc, None

    # Load labels
    label_file = os.path.join(self.label_dir, self.label_files[index])
    # label = np.load(label_file)
    # ensure the dtype matches the label file format
    label = np.fromfile(label_file, dtype=np.int32)  

    # TODO: Fetch labels from label_file
    # TODO: Mask the label with a mask of 0xFFFF
    # TODO: Use the masked label to index the remap array
    label = label & 0xFFFF
    label = self.remap_array[label]

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
      # add linear layer
      self.net.add_module("Lin" + str(i), nn.Linear(cs[i-1], cs[i]))
      # TODO: Replace None with a batchnorm layer of size cs[i]
      # add batch normalization layer
      self.net.add_module("Bn" + str(i), nn.BatchNorm1d(cs[i]))
      #TODO: Replace None with a ReLU Layer
      # add ReLu activation layer
      self.net.add_module("ReLU" + str(i), nn.ReLU())

    if linear_out is not None:
      # add final lin layer if lin_out is specified
      self.net.add_module("LinFinal", nn.Linear(cs[-1], linear_out))

  def forward(self, x):
    if x.ndim == 2:
        # If there are only two dimensions, assume x is of shape (N, C) and unsqueeze a batch dimension.
        x = x.unsqueeze(0)  # Now x is of shape (1, N, C)
    B, N, C = x.shape
    x = x.reshape(B * N, C)  # Use reshape instead of view

    for layer in self.net:
      x = layer(x)
      # print(f"output of {layer}: {x}")

    # x = self.net(x)
    C1 = x.shape[-1]
    x = x.reshape(B, N, C1)

    return x


    
# This module learns to combines global and local point features
class PointNetModule(nn.Module):
  def __init__(self, cs_en, cs_dec, num_classes=20):
    super().__init__()
    # TODO: Create a PointNetEncoder with cs=cs_en
    self.enc = PointNetEncoder(cs_en)
    # TODO: Create a PointNetEncoder decoder module with cs=cs_dec
    # and linear_out=num_classes
    self.dec = PointNetEncoder(cs_dec, linear_out=num_classes)

  def forward(self, x):
    B, N, C = x.shape
    # Encoder
    # TODO: Feed x through the PointNetEncoder
    point_feats = self.enc(x)
    # Max across points
    # TODO: Use max pooling across the point dimension to create
    # a global_feats tensor of shape (B, C1)
    # We used the torch.max
    global_feats, _ = torch.max(point_feats, dim=1, keepdim=True)  # (B, 1, C1)
    # TODO: Reshape global_feats from (B, C1) to (B, 1, C1)
    # And repeat along the middle dimension N times using repeat
    # End tensor should be (B, N, C1)
    global_feats = global_feats.repeat(1, N, 1)  # (B, N, C1)
    # TODO: Concatenate local and global features along channel dimension (2)
    joint_feats = torch.cat([point_feats, global_feats], dim=2)  # (B, N, C1+C2)
    # TODO: Feed joint_feats through the decoder module
    out = self.dec(joint_feats)

    return out

# This module adds a T-Net

class PointNetFull(nn.Module):
  def __init__(self, cs_en, cs_dec, cs_t_en, cs_t_dec):
      super().__init__()
      # Create a PointNetEncoder with cs=cs_t_en
      self.t_enc = PointNetEncoder(cs_t_en)
      # Create a PointNetEncoder with cs=cs_t_dec and linear_out=9
      self.t_dec = PointNetEncoder(cs_t_dec, linear_out=9)
      # Create a PointNetModule with cs_en, and cs_dec
      self.joint_enc = PointNetModule(cs_en, cs_dec)

  def forward(self, x):
    B, N, C = x.shape
    # print("Input shape:", x.shape)

    # T-Net Encoder
    t_feats = self.t_enc(x)
    # print("T-Net Encoder Output:", t_feats)

    # Max pool across the point dimension of t_feats
    global_t_feats, _ = torch.max(t_feats, dim=1, keepdim=True)

    # Feed global features through the T-Net Decoder
    t_feats = self.t_dec(global_t_feats.squeeze(1))
    # print("T-Net Decoder Output:", t_feats)

    # Reshape t_feats to (B, 3, 3)
    point_trans = t_feats.view(B, 3, 3)
    # print("Transformation matrix:", point_trans)

    # Add an identity matrix to the transformation
    identity = torch.eye(3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
    point_trans += identity

    # Apply the transformation to the input points
    x = x.transpose(1, 2)  # (B, C, N)
    transformed_points = torch.bmm(point_trans, x)
    transformed_points = transformed_points.transpose(1, 2)  # (B, N, C)
    # print("Transformed points shape:", transformed_points.shape)

    # Feed the transformed points through the joint encoder
    output_preds = self.joint_enc(transformed_points)
    # print("Output shape:", output_preds.shape)

    return output_preds

'''
# Function to initialize network weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Example of creating and initializing the network
cs_t_en = [3, 32, 64]
cs_t_dec = [64, 32, 9]
cs_enc = [3, 32, 64, 128]
cs_dec = [256, 128, 64, 20]

net = PointNetFull(cs_enc, cs_dec, cs_t_en, cs_t_dec)
net.apply(initialize_weights)

# Set T-Net to evaluation mode
net.t_enc.eval()
net.t_dec.eval()
'''

def IoU(targets, predictions, num_classes, ignore_index=None):
    
    intersections = torch.zeros(num_classes, device=targets.device)
    unions = torch.zeros_like(intersections)
    counts = torch.zeros_like(intersections)

    # Discard ignored points if ignore_index is specified
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        targets = targets[valid_mask]
        predictions = predictions[valid_mask]

    # Loop over classes and update the counts, unions, and intersections
    for c in range(num_classes):
        intersection = ((predictions == c) & (targets == c)).sum()
        union = ((predictions == c) | (targets == c)).sum()

        intersections[c] = intersection
        unions[c] = union
        counts[c] = (targets == c).sum()

    # Add small value to unions to avoid division by zero
    unions += 0.00001

    # Per-class IoU
    iou = intersections / unions

    # Set IoU for classes with no points to 1
    iou[counts == 0] = 1.0

    # Calculate mean, ignoring ignore index
    if ignore_index is not None and ignore_index < num_classes:
        valid_iou = iou[torch.arange(num_classes) != ignore_index]
        miou = valid_iou.mean()
    else:
        miou = iou.mean()

    # print("miou: ", miou)
    # print("iou: ", iou)
    return iou, miou

'''
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


# Compute the per-class iou and miou
def IoU(targets, predictions, num_classes, ignore_index=0):
    # Initialize IoU for each class
    iou_per_class = torch.zeros(num_classes, device=targets.device)

    # Loop over classes and update the counts, unions, and intersections
    for c in range(num_classes):
        # Skip the ignored index
        if c == ignore_index:
            iou_per_class[c] = 1.0
            continue

        # Calculate intersection and union for each class
        intersection = ((predictions == c) & (targets == c)).sum().item()
        union = ((predictions == c) | (targets == c)).sum().item()

        # Compute the IoU for this class, add a small epsilon to avoid division by zero
        iou = intersection / (union + 1e-6) if union > 0 else 1.0
        iou_per_class[c] = iou

    # Only calculate mIoU for classes not equal to ignore_index
    valid_indices = torch.arange(num_classes) != ignore_index
    valid_iou = iou_per_class[valid_indices]
    miou = valid_iou.mean() if len(valid_iou) > 0 else 1.0

    print("miou: ",miou)
    print("iou_per_class: ", iou_per_class)
    return iou_per_class, miou

'''
