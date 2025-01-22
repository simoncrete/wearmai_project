# Code taken in-part from coco_fit.py from older repo in SPIN
import sys
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Adjust Python path to include the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from smplify import SMPLify  
from models.smpl import SMPL
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_wearmai():
    # 1 Load the wearmai .npz
    wearmai_data = np.load('data/dataset_extras/wearmai_train.npz')
    imgnames = wearmai_data['imgname']
    centers = wearmai_data['center']
    scales = wearmai_data['scale']
    keypoints_openpose = wearmai_data['openpose']  # (N,25,3)
    keypoints_gt = wearmai_data['part']      # (N,24,3)
    keypoints_2d = np.concatenate([keypoints_openpose, keypoints_gt], axis=1) #(N,49,3)


    N = len(imgnames)
    print("Number of images:", N)

    # 2 Initialize the SMPLify object
    smplify = SMPLify(
        step_size=1e-2,
        batch_size=16,         
        num_iters=100,
        focal_length=5000, # CHANGE??
        device=device
    )

    # 3 Create a numpy array to store final SMPL params:
    #    - 72D pose
    #    - 10D shape (betas)
    #    - 3D global translation
    # --> total 85 parameters
    final_fits = np.zeros((N, 85), dtype=np.float32)

    # 4 Loop and fit SMPLi
    for i in tqdm(range(N)):
        img_path = os.path.join('/home/simon/Desktop/wearMAI_data/coco/', imgnames[i])
        # Read the image if you need height/width:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
        h, w, _ = img.shape

        # Keypoints for this sample
        kp_2d = torch.from_numpy(keypoints_2d[i][None]).float().to(device)
        
        # Initial guesses
        init_pose = torch.zeros((1, 72), dtype=torch.float32, device=device)
        init_betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
        init_cam_t = torch.tensor([[0, 0, 2.5]], dtype=torch.float32, device=device)
        
        # Camera center CHANGE??
        camera_center = torch.tensor([[w / 2.0, h / 2.0]], dtype=torch.float32, device=device)
        
        # 5 Run SMPLify
        
        vertices, joints, pose, betas, cam_t, reprojection_loss = smplify(
            init_pose,
            init_betas,
            init_cam_t,
            camera_center,
            kp_2d
        )
        
        # 6 Save [pose, betas, translation] in final_fits
        final_fits[i, :72]   = pose.cpu().numpy().reshape(-1)
        final_fits[i, 72:82] = betas.cpu().numpy().reshape(-1)
        final_fits[i, 82:85] = cam_t.cpu().numpy().reshape(-1)
    
    np.save('wearmai_fits.npy', final_fits)
    print("Saved SMPL fits to wearmai_fits.npy")


if __name__ == "__main__":
    fit_wearmai()

#python fit_wearmai.py to run

