import numpy as np
import pickle
import argparse
import glob2
import os
import re
from numpy.lib.npyio import load
from numpy.lib.type_check import imag
import math
def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext), recursive=True)
    files.sort
    return files

def main(ocr_output_path):
    all_properties = find_recursive(ocr_output_path, '.all_property.bin')
    items_3d = [] # [3d_location, norm, [img1, img2, ...]]
    d_threshold = 0.6 # max distance = 1m
    cos_threshold = 0.5 # max angle ~= 1/3 pi 
    for all_property_path in all_properties:
        with open(all_property_path, 'rb') as fb:
            all_property = pickle.load(fb)
            # property = [boxes[box_id], txts[box_id], scores[box_id], n_p, points_inlier, images[image_id].qvec, tvec]
        image_name = re.sub('.all_property.bin', '.jpg', all_property_path)
        # print(image_name)  
        n = all_property[3]
        if(np.isnan(n[0]) or np.linalg.norm(n) < 0.99 or np.linalg.norm(n)>1.01):
            continue
        n = n / np.linalg.norm(n)
        # print(n)
        points = all_property[4]
        xyz = np.average(points, axis=0)
        # print(xyz)
        flag_found_same = False
        for item_3d in items_3d:
            if(len(item_3d) < 1):
                continue
            dist = np.linalg.norm(xyz- item_3d[0]) 
            angle = np.dot(n, item_3d[1])
            if(dist > d_threshold or angle < cos_threshold):
                continue
            elif(not flag_found_same):
                flag_found_same = True
                found_item = item_3d
                item_3d[2].append(image_name)
                img_nums = len(item_3d[2])
                item_3d[0] = item_3d[0] * (img_nums - 1) / img_nums + xyz / img_nums
                item_3d[1] = item_3d[1] * (img_nums - 1) + n
                item_3d[1] /= np.linalg.norm(item_3d[1])
            else: 
                for img_names in item_3d[2]:
                    found_item[2].append(img_names)
                img_nums = len(found_item[2])
                new_img_nums = len(item_3d[2])
                found_item[0] = found_item[0] * (img_nums - new_img_nums) / img_nums + new_img_nums * item_3d[0] / img_nums
                found_item[1] = found_item[1] * (img_nums - new_img_nums) + new_img_nums * item_3d[1]
                found_item[1] /= np.linalg.norm(found_item[1])
                item_3d.clear()
        if(not flag_found_same):
            items_3d.append([xyz, n, [image_name]])
    for item_3d in items_3d:
        if(len(item_3d) < 1):
            continue        
        print(item_3d[0], item_3d[1])
        for img_3d in item_3d[2]:
            print("**", img_3d)
    with open(ocr_output_path + "/items_3d.bin", 'wb') as fp:
        pickle.dump(items_3d, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocr_output_path",
        required=True,
        type=str,
        help="visualized output paths"
    )
    args = parser.parse_args()
    main(args.ocr_output_path)
    