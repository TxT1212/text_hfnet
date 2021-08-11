import os, sys
import numpy as np
import pickle
import glob2
import colmap_model.read_write_model
def find_recursive(root_dir, ext='.jpg'):
    path = os.path.join(root_dir, '*'+ext)
    files = glob2.glob(path)
    files.sort()
    return files

query_image_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_undistort/'
query_orc_path = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_ocr/'
query_image_with_text_list = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/queries.txt'
imgs = find_recursive(query_image_path)
# print(imgs)

def get_query_list():
    # 去除文字检测失败的图片
    for img in imgs:
        img = img.replace(query_image_path, "")
        ocr = find_recursive(query_orc_path, img + '*.jpg')
        if(len(ocr)):
            print(img) # >> query_image_with_text_list
# get_query_list()
gt_query = "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_gt/"
ws = "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/"
ibl_gt = glob2.glob(os.path.join(gt_query, '*.camera'))
ibl_gt.sort()

fq = open(query_image_with_text_list, "r")
query_images_with_text =fq.readlines()

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
        model="PINHOLE"
        print(ibl_gt_i.replace(gt_query, "").replace(".camera", ".jpg"), model, width, height, *params)
