###################################################
#  input: colmap points.bin images.bin
#         paddle ocr imageXXX.jpg_ocr_result.npy
# output: xyz \in R3 质心
#         xyz*4 \in R12 包围圈
####################################################
import numpy as np
import argparse

from numpy.core.numeric import cross
from read_write_model import *
import math


def load_ocr_result(result_path):
    result = np.load(result_path, allow_pickle=True)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    boxes = [line[0] for line in result]
    # for line in result:
    #     print(line)
    return boxes, txts, scores


#
def if_point_in_box(point, box, box_diff):
    cross_result = np.cross(box - point, box_diff) 
    # print("box ", box)
    # print("point ", point)
    return np.all(cross_result > 0) or np.all(cross_result < 0)

def choose_text_points2d(img, box):
    # left = np.min(box[:, 0, 0]) * expand
    # right = np.max(box[:, 0, 0]) * expand
    # down = np.min(box[:, 0, 1]) * expand
    # up = np.max(box[:, 0, 1]) * expand
    box_diff = np.diff(box,axis = 0,prepend=box[-1:,:])
    point3D_ids_in = []
    for xy, point3D_id in zip(img.xys, img.point3D_ids):
        if(point3D_id == -1 ):
            continue
        if(if_point_in_box(xy, box, box_diff)):
            point3D_ids_in.append(point3D_id)
    return point3D_ids_in


def main(model_path, ocr_output_path):

    images = read_images_binary(os.path.join(model_path, "images.bin"))
    points3D = read_points3d_binary(os.path.join(model_path, "points3D.bin"))
    # print(len(images))
    dict_test2xyz = {}
    for image_id in images:
        # print(images[image_id].name)
        boxes, txts, scores = load_ocr_result(
            ocr_output_path + "/" + images[image_id].name + ".ocr_result.npy")
        if(len(boxes) == 0):
            continue
        for box_id in range(len(boxes)):
            if scores is not None and (scores[box_id] < 0.5 or math.isnan(scores[box_id])):
                continue
            box = np.reshape(
                np.array(boxes[box_id]), [-1, 1, 2]).astype(np.int64).squeeze()
            points3d_ids_in = choose_text_points2d(images[image_id], box)
            xyzs = []
            for points3d_id_in in points3d_ids_in:
                xyz = points3D[points3d_id_in].xyz
                xyzs.append(xyz)
            if(len(xyzs) > 0):
                xyzs = np.array(xyzs).reshape(-1, 3)
                np.save(ocr_output_path + "/" +
                        images[image_id].name + str(box_id) + "_cloud3d_1.npy", xyzs)
                xyz_average = np.average(xyzs, axis=0)
                dict_test2xyz[images[image_id].name +
                              str(box_id)] = xyz_average
            else:
                print(ocr_output_path + "/" +
                      images[image_id].name + str(box_id) + ".jpg")
    np.save(ocr_output_path + "/dict_test2xyz_f1_1.npy", dict_test2xyz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--input_path",
    #     required=True,
    #     type=str,
    #     help="input picture paths"
    # )
    parser.add_argument(
        "--ocr_output_path",
        required=True,
        type=str,
        help="visualized output paths"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="input colmap model paths"
    )
    args = parser.parse_args()
    main(args.model_path, args.ocr_output_path)
