import os,sys
sys.path.append('/home/ezxr/Downloads/ocr/PaddleOCR')
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import glob2
from tqdm import tqdm
import argparse
import math
import cv2
import numpy as np
import re

ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_limit_side_len=10000) # need to run only once to download and load model into memory


def ocr_images(input_path, imgs, output_path, chop_charactor):
    for img_path in tqdm(imgs):
        img_name = img_path.split('/')[-1]
        # img_mat = cv2.imread(img)    
        print(img_path)
        img = cv2.imread(img_path, 1)

        #### begin gray scale test  
        # cv.IMREAD_GRAYSCALE = 0
        # img = cv2.imread(img_path, 0)
        print("img.shape: " ,img.shape)
        #### end gray scale test

        ## begin test flip
        # img = img[:][:][::-1]
        # cv2.imshow("1", img)
        # cv2.waitKey()
        ## end test filp

        result = ocr.ocr(img, cls=True)
        # for line in result:
        #     print(line)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        np.save(output_path + str.replace(img_path, input_path, "")+".ocr_result.npy", np.array(result))


        # chop text area
        box_num = len(boxes)
        drop_score = 0.5
        for i in range(box_num):
            if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
                continue
            box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
            left = np.min(box[:, 0, 0])
            right = np.max(box[:, 0, 0])
            width_ = right -left
            down = np.min(box[:, 0, 1])
            up = np.max(box[:, 0, 1])
            height_ = up - down

            left_expand = max(left - int(width_/2), 0)
            down_expand = max(down - int(height_/2), 0)
            right_expand = min(right + int(width_/2), img.shape[1])
            up_expand = min(up + int(height_/2), img.shape[0])
            image_chop = img[down_expand:up_expand+1, left_expand:right_expand+1]

            # save result
            ratio_w_h = int((right-left)/(up-down)*100)/100
            if ratio_w_h < 0.8:
                image_chop = image_chop.transpose(1,0,2)
            txt = txts[i]
            save_name = output_path + str.replace(img_path, input_path, "") + str(i) + "_isFlip:0" + "_txt:" + txt.replace("/", "").replace(" ", "") +"_wh-ratio:"  +str(ratio_w_h) + "_confidence:" + str(int(scores[i]*1000)/1000.)+ ".jpg"
            print(save_name)
            cv2.imwrite(save_name, image_chop)

            # flip and ocr
            isFlip = 0
            if(scores[i] < 0.7):
                image_chop_flip = img[down:up+1, left:right+1]
                image_chop_flip = image_chop_flip[:][:][::-1]
                result_flip = ocr.ocr(image_chop_flip, cls=True, det=False)
                scores_flip = [line[1] for line in result_flip]
                if(scores_flip[0] > scores[i]+0.1):
                    ### 镜像文字
                    scores[i] = scores_flip[0]
                    txts_flip = [line[0] for line in result_flip]
                    if(len(txts_flip[0]) > 2):
                        print(txts[i], " -> ", txts_flip[0])
                        txts[i] = txts_flip[0]
                        isFlip = 1
                        save_name = output_path + str.replace(img_path, input_path, "") + str(i) + "_isFlip:1" + "_txt:" + txts[i].replace("/", "").replace(" ", "") +"_wh-ratio:"  +str(ratio_w_h) + "_confidence:" + str(int(scores[i]*1000)/1000.)+ ".jpg"
                        cv2.imwrite(save_name, image_chop)


            
        # draw result
        # im_show = draw_ocr(img, boxes, txts, scores, font_path='./doc/fonts/simfang.ttf')
        # im_show = Image.fromarray(im_show)
        # print(im_show.size)
        # c = str.replace(img_path, input_path, output_path)
        # im_show.save(c)


def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext))
    return files

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

    # generate testing img list
    if os.path.isdir(args.input_path):
        for dirpath, dirnames, filenames in os.walk(args.input_path):
            imgs = find_recursive(args.input_path)
            for dirname in dirnames:
                # print(os.path.join(dirpath, dirname))  
                # print(dirname)
                path = os.path.join(args.output_path, dirname)
                print(path)
                os.makedirs(path, exist_ok=True)    
    else:
        print("Error! input_path is not a path")
        sys.exit()
    assert len(imgs), "imgs should be a path to img (.jpg) or directory."
    ocr_images(args.input_path, imgs, args.output_path, args.chop_charactor)

