import os
from os.path import join
import sys
import json
import numpy as np
from .read_openpose import read_openpose

def wearmai_data_extract(dataset_path, openpose_path, out_path):

    # convert joints to global order with wearmai mapping
    #joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
    joints_idx = [1, 10, 3, 11, 0, 4, 14, 2, 8, 16, 6, 13, 5, 7, 9]

    '''
    *0 right knee inner = Rk_i --> 1
    1 left elbow = Le --> 10
    2 left hip = Lh --> 3
    3 left wrist = Lw --> 11
    *4 right ankle outer = Ra_o --> 0
    *5 right ankle inner = Ra_i --> 0
    *6 left knee outer = Lk_o --> 4
    *7 right knee outer = Rk_o --> 1
    8 pelvis = P --> 14
    9 right hip = Rh --> 2
    10 right shoulder = Rs --> 8
    11 spine = S --> 16
    *12 left knee inner = Lk_i --> 4
    13 right wrist = Rw --> 6
    14 head = H --> 13
    *15 left ankle inner = La_i --> 5
    16 right elbow = Re --> 7
    *17 left ankle outer = La_o --> 5
    18 left shoulder = Ls --> 9

    possible mapping with merging
    0 right knee --> 1
    1 left elbow --> 10
    2 left hip --> 3
    3 left wrist --> 11
    4 right ankle --> 1
    5 left knee  --> 4
    6 pelvis  --> 14
    7 right hip  --> 2
    8 right shoulder --> 8
    9 spine --> 16
    10 right wrist --> 6
    11 head --> 13
    12 left ankle  --> 5
    13 right elbow --> 7
    14 left shoulder --> 9
    '''


    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 
                             'annotations', 
                             'person_keypoints_train2017.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        #change to right amount of keypoints
        keypoints = np.array(keypoints, dtype=np.float32)
        keypoints = keypoints.reshape(-1, 3)
        #keypoints = np.reshape(keypoints, (15,3))
        #keypoints[keypoints[:,2]>0,2] = 1
        
        #get rid of outer and inner (could use middle point if both exist, or just default to one)
        #come back if time permits to add extra logic
        filtered_keypoints = np.zeros((15,3))
        filtered_keypoints[0]  = keypoints[0] #right knee (inner)
        filtered_keypoints[1]  = keypoints[1] #left elbow
        filtered_keypoints[2]  = keypoints[2] #left hip
        filtered_keypoints[3]  = keypoints[3] #left wrist
        filtered_keypoints[4]  = keypoints[4] #right ankle
        filtered_keypoints[5]  = keypoints[6] #left knee
        filtered_keypoints[6]  = keypoints[8] #pelvis
        filtered_keypoints[7]  = keypoints[9] #right hip
        filtered_keypoints[8]  = keypoints[10] #right shoulder
        filtered_keypoints[9]  = keypoints[11] #spine
        filtered_keypoints[10] = keypoints[13] #right wrist
        filtered_keypoints[11] = keypoints[14] #head
        filtered_keypoints[12] = keypoints[15] #left ankle
        filtered_keypoints[13] = keypoints[16] #right elbow
        filtered_keypoints[14] = keypoints[18] #left shoulder
        
        filtered_keypoints[filtered_keypoints[:,2]>0,2] = 1

        # check if all major body joints are annotated                   
        #if sum(keypoints[5:,2]>0) < 12:
        #    print('not all major joints are annoted')
        
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('train2017', img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = filtered_keypoints
        #print(part)
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200
        
        # read openpose detections
        #json_file = os.path.join(openpose_path, 'coco',
         #   img_name.replace('.jpg', '_keypoints.json'))
        #openpose = read_openpose(json_file, part, 'coco')
        openpose = np.zeros([25, 3])
       
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'wearmai_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_)
    print('Succesfully generated wearmai_train.npz')