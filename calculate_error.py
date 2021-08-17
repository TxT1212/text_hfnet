# All conditions: (0.1m, 1°) / (0.25m, 2°) / (1m, 5°)
# f.write(f'{name} {qvec} {tvec}\n')
from operator import gt
import os
import sys
import numpy as np
import pickle
import glob2
from numpy.core.records import recarray
import colmap_model.read_write_model
import math
# read estimate result
results_q_pose = "/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_netvlad20_.txt"
results_q_pose = "/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_ours.txt"
results_q_pose = "/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_hfnet_org1.txt"


def read_pose_hloc(results_q_pose):
    fq = open(results_q_pose, "r")
    poses_q_t = fq.readlines()
    poses = {}
    for i in poses_q_t:
        nowline = i.split()
        

        poses[nowline[0]] = ([float(i) for i in nowline[1:5]], [float(i) for i in nowline[5:]])
    return poses


estimate_pose = read_pose_hloc(results_q_pose)
for p in estimate_pose:
    print(p, estimate_pose[p])

# read gt pose
gt_query = "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_gt/"
ws = "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/"


def read_pose_baidu_ibl(gt_query):
    ibl_gt = glob2.glob(os.path.join(gt_query, '*.camera'))
    ibl_gt.sort()
    poses = {}
    for ibl_gt_i in ibl_gt:
        ibl_gt_i_ = ibl_gt_i.replace(gt_query, "").replace(".camera", ".jpg")
        f = open(ibl_gt_i, "r")
        # 1410.864258 0.000000 794.641113
        # 0.000000 1408.981201 603.413025 (K)
        # 0.000000 0.000000 1.000000
        # 0.000000 0.000000 0.000000 (0)
        # -0.807854 -0.008790 0.589317
        # -0.589321 -0.002360 -0.807895 (R)
        # 0.008492 -0.999959 -0.003274
        # -6.638618 12.360680 0.033681 (t)
        # 1632 1224
        lines = f.readlines()
        K_R_tvec = np.zeros([8, 3])
        for i in range(0, 8):
            nowline = lines[i].split()
            K_R_tvec[i][0] = float(nowline[0])
            K_R_tvec[i][1] = float(nowline[1])
            K_R_tvec[i][2] = float(nowline[2])
        # nowline = lines[8].split()
        # width = int(nowline[0])
        # height = int(nowline[1])
        # K = K_R_tvec[0:3, :]
        R = K_R_tvec[4:7, :]
        t = K_R_tvec[7, :]
        q = colmap_model.read_write_model.rotmat2qvec(R)
        poses[ibl_gt_i_] = (q, t)

        # params = np.array([K[0][0], K[1][1], K[0][2], K[1][2]])
        # model="PINHOLE"
        # print(ibl_gt_i.replace(gt_query, "").replace(".camera", ".jpg"), model, width, height, *params)
        # >> /home/ezxr/Documents/ibl_dataset_cvpr17_3852/queries_with_intrinsics.txt
    return poses
pose_gt = read_pose_baidu_ibl(gt_query)
for i in pose_gt:
    print(i, pose_gt[i])
def calculate_error(images, images_gt):
    distances = []
    angles = []
    for name in images:
        qvec = images[name][0]
        rvc = colmap_model.read_write_model.qvec2rotmat(qvec)
        tvec = images[name][1]
        tvec = -(rvc.T).dot(tvec)
        # print("T\n", T_img)

        qvec_gt = images_gt[name][0]
        rvec_gt = colmap_model.read_write_model.qvec2rotmat(qvec_gt)
        tvec_gt = images_gt[name][1]
        # tvec_gt = -(rvec_gt.T).dot(tvec_gt)
        # print("gt\n", rvec_gt, tvec_gt)
        distance = np.linalg.norm(tvec_gt - tvec.squeeze())
        rotation = rvc.dot(rvec_gt.T)
        # print("test: T*T': \n", rvc.dot(rvec_gt.T))
        angle = (rotation.dot(np.array([[0, 0, 1]]).T))[2][0]
        angle = round(angle, 10)
        # print("angle ", angle)
        if angle > 1:
            angle =1
        angle = math.acos(angle)
        distances.append(distance)
        angles.append(angle)

    return np.array(angles), np.array(distances)
angles, distances = calculate_error(estimate_pose, pose_gt)
cout_excelent = 0
cout_good = 0
cout_ok = 0
# print(angles.size) (0.25m, 2°) / (1m, 5°)
dist_thrd = [0.25, 0.5, 5]
angle_thrd = [2, 5, 10]
# dist_thrd = [0.1, 0.25, 1]
# angle_thrd = [1, 2, 5]
for i in range(angles.size):
    if (distances[i] < dist_thrd[0] and angles[i] < angle_thrd[0]):
        cout_excelent+=1 
    if (distances[i] < dist_thrd[1] and angles[i] < angle_thrd[1]):
        cout_good+=1 
    if (distances[i] < dist_thrd[2] and angles[i] < angle_thrd[2]):
        cout_ok+=1 
print("0.25m, 2°:",cout_excelent, "/", angles.size, " ", cout_excelent*1.0/angles.size)
print("0.5m, 5°:",cout_good, "/", angles.size, " ",  cout_good*1.0/angles.size)
print("5m, 10°:",cout_ok, "/", angles.size, " ",  cout_ok*1.0/angles.size)