from os import sysconf
import pickle
from typing import Pattern
from joblib.logger import PrintTime
import numpy as np
import argparse
import math
from cv2 import cv2
from tqdm import tqdm
import glob2
import os
import sys

# ** /home/ezxr/Documents/wxc/pic_ocr_flip/f1/./wxc_f1_20201214_f1n_route8_0003/00000180.jpg2_isFlip_0_txt_CHAUMET_wh-ratio_6.77_confidence_0.954_expand_4.jpg 0.901420440186386


def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext), recursive=True)
    files.sort
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
        img_path = ocr_results[i].replace(
            ".ocr_result.npy", "").replace(output_path, input_path)
        print(img_path)
        img = cv2.imread(img_path, 1)
        expand_ratios = [0, 4, 8]
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
            center_u = (left + right) / 2
            center_v = (up + down) / 2
            exapnd_max = min(img.shape[0], img.shape[1])
            for expand_ratio in expand_ratios:
                exapnd_expect = max(width_, height_)*(expand_ratio + 1)
                exapnd_actual = min(exapnd_expect, exapnd_max)
                exapnd_actual = int(exapnd_actual)
                chop_length = exapnd_actual
                left_expect = center_u - chop_length/2
                right_expect = center_u + chop_length/2
                down_expect = center_v - chop_length/2
                up_expect = center_v + chop_length/2
                if(down_expect < 0):
                    up_expand = up_expect - down_expect
                    down_expand = 0
                elif(up_expect > img.shape[0]):
                    down_expand = down_expect + (img.shape[0] - up_expect)
                    up_expand = img.shape[0]
                else:
                    down_expand = down_expect
                    up_expand = up_expect

                if(left_expect < 0):
                    right_expand = right_expect - left_expect
                    left_expand = 0
                elif(right_expect > img.shape[1]):
                    left_expand = left_expect + (img.shape[1] - right_expect)
                    right_expand = img.shape[1]
                else:
                    left_expand = left_expect
                    right_expand = right_expect
                down_expand = int(down_expand)
                left_expand = int(left_expand)
                up_expand = int(up_expand)
                right_expand = int(right_expand)
                image_chop = img[down_expand:up_expand, left_expand:right_expand]

                # save result
                txt = txts[i]
                save_name = output_path + str.replace(img_path, input_path, "") + str(i) + "_txt_" + txt.replace(
                    "/", "").replace(" ", "") + "_confidence_" + str(int(scores[i]*1000)/1000.) + "_expand_" + str(expand_ratio) + ".jpg"
                print(save_name)
                # print(image_chop.shape)
                # print(up_expand, down_expand, up_expect, down_expect)
                # print(right_expand, left_expand, right_expect, left_expect)
                # print(chop_length, right_expect - left_expect,up_expect - down_expect )
                # print(img.shape[0], img.shape[1])
                assert(abs(image_chop.shape[0] - image_chop.shape[1]) < 3)
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
        help="是否切割有文字的区域，并且保存成新的图片"
    )
    args = parser.parse_args()
    # ocr_results = find_recursive(args.input_path, ".ocr_result.npy")
    ocr_results = find_recursive(args.output_path, ".ocr_result.npy")
    assert(len(ocr_results))

    ocr_images_cut(args.input_path, ocr_results, args.output_path)
