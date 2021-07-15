import numpy as np
from numpy.core.defchararray import array
import open3d as o3d
from read_write_model import *

areas = np.load('/home/ezxr/Documents/wxc/pic_ocr_flip/dict_test2xyz.npy', allow_pickle=True)
ocr_outpath = '/home/ezxr/Documents/wxc/pic_ocr_flip/'
model_path = '/home/ezxr/Documents/wxc/f2_aligned_to_f1_gz/0/'
dict_test2xyz = areas.item()
images = read_images_binary(os.path.join(model_path, "images.bin"))
# print(type(dict_test2xyz))
items_3d = []
d_threshold = 1
count_point_nums = np.zeros(20)
for image_name in dict_test2xyz:
    xyz = dict_test2xyz[image_name]
    points = np.load(ocr_outpath+image_name+"_cloud3d.npy")
    # print(points.shape[0])
    if(points.shape[0]<5):
        count_point_nums[points.shape[0]-1] += 1
    else:
        count_point_nums[19] += 1


    # print(type(xyz))
    flag_found_same = False
    for item_3d in items_3d:
        dist = np.linalg.norm(xyz - item_3d[0]) 
        if(dist > d_threshold):
            continue
        else:
            flag_found_same = True
            item_3d[1].append(image_name)
            break
    if(not flag_found_same):
        items_3d.append([xyz, [image_name]])
    # print(count_point_nums)
for item_3d in items_3d:
    for orc2d_name in item_3d[1]:
        print(ocr_outpath + orc2d_name, ocr_outpath + item_3d[1][0])