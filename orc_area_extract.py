from os import sysconf
import pickle
from typing import Pattern
import numpy as np
import argparse
import math
from cv2 import cv2
from tqdm import tqdm
import glob2
import os
import sys


def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext), recursive=True)
    return files


def ocr_images_cut(input_path, ocr_results, output_path):
    for i in range(0, len(ocr_results)):
        print(ocr_results[i])
        result = np.load(ocr_results[i], allow_pickle=True)
        if(result.size < 1):
            continue
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        # chop text area
        box_num = len(boxes)
        drop_score = 0.5
        img_path = ocr_results[i].replace(".ocr_result.npy", "").replace(output_path, input_path)
        print(img_path)
        img = cv2.imread(img_path, 1)
        expand_ratios = [0, 1, 2, 4]
        for i in range(box_num):
            if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
                continue
            box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
            left = np.min(box[:, 0, 0])
            right = np.max(box[:, 0, 0])
            width_ = right - left
            down = np.min(box[:, 0, 1])
            up = np.max(box[:, 0, 1])
            height_ = up - down
            for expand_ratio in expand_ratios:
                left_expand = max(left - int(width_/2)*expand_ratio, 0)
                down_expand = max(down - int(height_/2)*expand_ratio, 0)
                right_expand = min(right + int(width_/2) *
                                   expand_ratio, img.shape[1])
                up_expand = min(up + int(height_/2)*expand_ratio, img.shape[0])
                image_chop = img[down_expand:up_expand +
                                 1, left_expand:right_expand+1]

                # save result
                ratio_w_h = int((right-left)/(up-down)*100)/100
                if ratio_w_h < 0.8:
                    image_chop = image_chop.transpose(1, 0, 2)
                txt = txts[i]
                save_name = output_path + str.replace(img_path, input_path, "") + str(i) + "_isFlip_0" + "_txt_" + txt.replace(
                    "/", "") + "_wh-ratio_" + str(ratio_w_h) + "_confidence_" + str(int(scores[i]*1000)/1000.) + "_expand_" + str(expand_ratio) + ".jpg"
                print(save_name)
                cv2.imwrite(save_name, image_chop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ocr"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="input picture paths"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="visualized output paths"
    )
    parser.add_argument(
        "--chop_charactor",
        type=bool,
        default=True,
        help="????????????????????????????????????????????????????????????"
    )
    args = parser.parse_args()
    ocr_results = find_recursive(args.input_path, ".ocr_result.npy")
    ocr_results = find_recursive(args.output_path, ".ocr_result.npy")
    assert(len(ocr_results))

    ocr_images_cut(args.input_path, ocr_results, args.output_path)
