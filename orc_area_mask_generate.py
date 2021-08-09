from os import sysconf
import pickle
from typing import Pattern
import numpy as np
import argparse
import math
import cv2
from tqdm import tqdm
import glob2
import os
import sys

def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext), recursive=True)
    return files

def ocr_area_mask_generate(input_path, ocr_results, output_path):
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
        img_path = ocr_results[i].replace(
            ".ocr_result.npy", "").replace(output_path, input_path)
        print(img_path)
        img = cv2.imread(img_path, 0)
        mask = np.zeros(img.shape)
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
            mask[down:up, left:right] = np.ones([height_, width_]) *255

        save_name = output_path + \
            str.replace(img_path, input_path, "") + "mask.jpg"
        print(save_name)
        cv2.imwrite(save_name, mask)

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
        help="是否切割有文字的区域，并且保存成新的图片"
    )
    args = parser.parse_args()
    ocr_results = find_recursive(args.output_path, ".ocr_result.npy")
    assert(len(ocr_results))

    ocr_area_mask_generate(args.input_path, ocr_results, args.output_path)
