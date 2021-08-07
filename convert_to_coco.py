# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of visualizing object and hand pose of one image sample."""

import numpy as np
import cv2
import matplotlib.pyplot as plt


from dex_ycb_toolkit.factory import get_dataset


def create_scene(sample, obj_file):
  """Creates the pyrender scene of an image sample.

  Args:
    sample: A dictionary holding an image sample.
    obj_file: A dictionary holding the paths to YCB OBJ files.

  Returns:
    A pyrender scene object.
  """
  # Create pyrender scene.
  scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

  # Add camera.
  fx = sample['intrinsics']['fx']
  fy = sample['intrinsics']['fy']
  cx = sample['intrinsics']['ppx']
  cy = sample['intrinsics']['ppy']
  cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
  scene.add(cam, pose=np.eye(4))

  # Load poses.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']
  pose_m = label['pose_m']

  # Load YCB meshes.
  mesh_y = []
  for i in sample['ycb_ids']:
    mesh = trimesh.load(obj_file[i])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_y.append(mesh)

  # Add YCB meshes.
  for o in range(len(pose_y)):
    if np.all(pose_y[o] == 0.0):
      continue
    pose = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose[1] *= -1
    pose[2] *= -1
    node = scene.add(mesh_y[o], pose=pose)

  # Load MANO layer.
  mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side=sample['mano_side'],
                         mano_root='manopth/mano/models',
                         use_pca=True)
  faces = mano_layer.th_faces.numpy()
  betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)

  # Add MANO meshes.
  if not np.all(pose_m == 0.0):
    pose = torch.from_numpy(pose_m)
    vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
    vert /= 1000
    vert = vert.view(778, 3)
    vert = vert.numpy()
    vert[:, 1] *= -1
    vert[:, 2] *= -1
    mesh = trimesh.Trimesh(vertices=vert, faces=faces)
    mesh1 = pyrender.Mesh.from_trimesh(mesh)
    mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
    mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
    mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
    node1 = scene.add(mesh1)
    node2 = scene.add(mesh2)

  return scene

def reproject_to_3d(im_coords, K, z):
  im_coords = np.stack([im_coords[:,0], im_coords[:,1]],axis=1)
  im_coords = np.hstack((im_coords, np.ones((im_coords.shape[0],1))))
  projected = np.dot(np.linalg.inv(K), im_coords.T).T
  projected[:, 0] *= z
  projected[:, 1] *= z
  projected[:, 2] *= z
  return projected

def make_dirs(path):
  try:
    base = os.path.dirname(path)
    os.makedirs(base)
  except Exception as e:
    pass

def crop_and_center(imgInOrg, gtIn):
  shape = imgInOrg.shape
  box_size = min(imgInOrg.shape[0], imgInOrg.shape[1])
  center = get_center(gtIn)
  x_min_v = center[0] - box_size/2
  y_min_v = center[1] - box_size/2
  x_max_v = center[0] + box_size/2
  y_max_v = center[1] + box_size/2
  
  x_min_n = int(max(0, -x_min_v))
  y_min_n = int(max(0, -y_min_v))

  x_min_o = int(max(0, x_min_v))
  y_min_o = int(max(0, y_min_v))
  x_max_o = int(min(imgInOrg.shape[1], x_max_v))
  y_max_o = int(min(imgInOrg.shape[0], y_max_v))

  w = int(x_max_o - x_min_o)
  h = int(y_max_o - y_min_o)
  
  new_img = np.zeros((box_size, box_size, 3))
  new_img[y_min_n:y_min_n+h, x_min_n:x_min_n+w] = \
          imgInOrg[y_min_o:y_max_o, x_min_o:x_max_o]
  new_img = cv2.resize(new_img, (256, 256))
  x_min_v *= float(256)/480
  y_min_v *= float(256)/480
  return new_img, x_min_v, y_min_v

def main():
  name = 's0_train'
  dataset = get_dataset(name)
  count = 0
  for idx in range(len(dataset)):
    sample = dataset[idx]
    label = np.load(sample['label_file'])
    fx = sample['intrinsics']['fx']
    fy = sample['intrinsics']['fy']
    cx = sample['intrinsics']['ppx']
    cy = sample['intrinsics']['ppy']

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    joint_3d = label["joint_3d"]
    joint_2d = label["joint_2d"]

    if (not np.all(joint_3d[0] == np.ones(3) * -1)):
      continue
    
   
    img = cv2.imread(sample["color_file"])
    processed_img, x_offset, y_offset = crop_and_center(img, joint_2d)
    joint_2d[:, 0] -= x_offset
    joint_2d[:, 1] -= y_offset
    output_path = sample["color_file"].replace("DexYCB", "DexYCBOutput")
    make_dirs(output_path)
    cv2.imwrite(output_path, processed_img)
    joint_3d = reproject_to_3d(joint_2d, K, joint_3d[:, 2])
    input("? ")
  print(f"Total count: {count}")
if __name__ == '__main__':
  main()
