import os
import sys
import numpy as np
import pickle
import glob2
import colmap_model.read_write_model
import re
import kapture


def find_recursive(root_dir, ext='.jpg'):
    path = os.path.join(root_dir, '**/*'+ext)
    files = glob2.glob(path, recursive=True)
    files.sort()
    return files


query_image_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_undistort/'
# query_image_path = '/media/ezxr/data/nevar/HyundaiDepartmentStore/1F/mapping/'
query_orc_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_ocr/'
# query_orc_path = '/media/ezxr/data/nevar/HyundaiDepartmentStore_ocr/1F/mapping/'
query_image_with_text_list = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/queries.txt'
imgs = find_recursive(query_image_path)
# print(imgs)


def get_query_list():
    # 去除文字检测失败的图片
    ocrs = find_recursive(query_orc_path, 'jpg*expand_4.jpg')

    # image_suc = set()
    for ocr in ocrs:
        print(ocr)
        # ocr = re.sub('0_txt.*', '', ocr)
        # image_suc.add(ocr)
    # for ocr in image_suc:
    #     print(ocr.replace(query_orc_path, ""))
    # for img in imgs:
    #     img = img.replace(query_image_path, "")
    #     if(len(ocr)):
    #         print(img) # >> query_image_with_text_list
# get_query_list()


def get_query_with_intrinsics(gt_query="/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_gt/"):
    ibl_gt = glob2.glob(os.path.join(gt_query, '*.camera'))
    ibl_gt.sort()
    fq = open(query_image_with_text_list, "r")
    query_images_with_text = fq.readlines()
    for ibl_gt_i in ibl_gt:
        ibl_gt_i_ = ibl_gt_i.replace(gt_query, "").replace(".camera", ".jpg\n")
        if ibl_gt_i_ in query_images_with_text:
            f = open(ibl_gt_i, "r")
            lines = f.readlines()
            K_R_tvec = np.zeros([8, 3])
            for i in range(0, 8):
                nowline = lines[i].split()
                K_R_tvec[i][0] = float(nowline[0])
                K_R_tvec[i][1] = float(nowline[1])
                K_R_tvec[i][2] = float(nowline[2])
            nowline = lines[8].split()
            width = int(nowline[0])
            height = int(nowline[1])
            K = K_R_tvec[0:3, :]
            params = np.array([K[0][0], K[1][1], K[0][2], K[1][2]])
            model = "PINHOLE"
            print(ibl_gt_i.replace(gt_query, "").replace(
                ".camera", ".jpg"), model, width, height, *params)
            # >> /home/ezxr/Documents/ibl_dataset_cvpr17_3852/queries_with_intrinsics.txt
# get_query_with_intrinsics()


def get_query_with_intrinsics_from_kapture(camera_path, image_path):
    # kapture.
    f1 = open(camera_path, "r")
    cameras = {}
    lines_1 = f1.readlines()
    for line in lines_1:
        if(len(line) == 0 or line[0] == '#'):
            continue
        else:
            now_line = line.strip('\n').split(', ')
            print(now_line)
            cameras[now_line[0]] = now_line[3:]
    f2 = open(image_path, "r")
    lines_2 = f2.readlines()
    for line in lines_2:
        if(len(line) == 0 or line[0] == '#'):
            continue
        else:
            now_line = line.strip('\n').split(', ')
            print(now_line[2], *cameras[now_line[1]])


camera_path = '/media/ezxr/data/nevar/HyundaiDepartmentStore/1F/test/sensors.txt'
image_path = '/media/ezxr/data/nevar/HyundaiDepartmentStore/1F/test/records_camera.txt'
get_query_with_intrinsics_from_kapture(camera_path, image_path)


def shink_query_list_with_another_list(l1, l2):
    f1 = open(l1, "r")
    lines_1 = f1.readlines()
    names = set()
    for line in lines_1:
        names.add(line.split()[0])
    f2 = open(l2, "r")
    lines_2 = f2.readlines()
    for line in lines_2:
        if(line.split()[0] in names):
            print(line[:-1])
# shink_query_list_with_another_list("/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_ours.txt", '/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_netvlad100.txt')
