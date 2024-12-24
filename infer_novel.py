import os
import time

import nibabel as nib
import numpy as np

import torch
from medpy import metric
from torch import nn

from Network.FFMNet import MixAttNet as Net
from Utils import AvgMeter, check_dir, dice_score

from DataOp import get_list
from pathlib import Path


# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES']='0'

# define the model loading path
ckpt_path = os.path.join('FetalCPSeg-Program/Test/ckpt_save/')

# get the data path list
test_list = get_list(dir_path='Input')

# here we define the network
net = Net().cuda()
# I trained this network with DataParallel mode, so we need open this mode in the test phase as well.
net = nn.DataParallel(net)

# load the trained model
net.load_state_dict(torch.load(ckpt_path+'/best_test.pth.gz'))

patch_size = 64
# patch sampling spacing
spacing = 4

net.eval()

test_meter = AvgMeter()

for idx, data_dict in enumerate(test_list):
  image_path = data_dict['image_path']
  print("Input file is: " + image_path)
  image_folder = Path(image_path).parent.absolute()
  predict_path = str(image_folder) + "/" + "predict.nii.gz"

  if not os.path.isfile(predict_path):
    print("Output predict not found")

    image = nib.load(image_path).get_fdata()

    w, h, d = image.shape

    pre_count = np.zeros_like(image, dtype=np.float32)
    predict = np.zeros_like(image, dtype=np.float32)

    # here we generate the patch sampling coordinate index of x,y,z
    x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size, patch_size // spacing)[:, np.newaxis],np.array([w - patch_size])[:, np.newaxis])).astype(int))
    y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size, patch_size // spacing)[:, np.newaxis],np.array([h - patch_size])[:, np.newaxis])).astype(int))
    z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size, patch_size // spacing)[:, np.newaxis],np.array([d - patch_size])[:, np.newaxis])).astype(int))
    start_time = time.time()

    for x in x_list:
      for y in y_list:
        for z in z_list:
          image_patch = image[x:x + patch_size, y:y + patch_size, z:z + patch_size].astype(np.float32)
          patch_tensor = torch.from_numpy(image_patch[np.newaxis, np.newaxis, ...]).cuda()
          predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += net(patch_tensor).squeeze().cpu().data.numpy()
          pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1

    # get the final prediction
    predict /= pre_count

    predict = np.squeeze(predict)
    image = np.squeeze(image)
    # mask = np.squeeze(mask)

    # threshold the prediction with 0.5
    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0

    # here we saved the prediction into the subject folder.
    predict_nii = nib.Nifti1Image(predict, affine=np.eye(4))
    save_path = data_dict['save_path']
    nib.save(predict_nii, os.path.join(save_path, 'predict.nii.gz'.format(idx)))

    print("[{}] Testing Finished, Cost {:.2f}s".format(idx, time.time()-start_time))

  else:
    print(predict_path + ' already found')
