###################################################
#  input: colmap points.bin images.bin
#         paddle ocr imageXXX.jpg_ocr_result.npy
# output: 
# bbox 二维图片中bbox
# text  文字识别结果
# score  文字识别置信度
# xyz 三维质心
# n_p 三维法向
# points 所有inlier三维点
# tvec 
# rvec
####################################################
from token import EXACT_TOKEN_TYPES
import numpy as np
import argparse
from read_write_model import *
import math
import open3d as o3d
import pickle


def load_ocr_result(result_path):
    result = np.load(result_path, allow_pickle=True)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    boxes = [line[0] for line in result]
    for line in result:
        # print(line)
        pass
    return boxes, txts, scores


#
def choose_text_points2d(img, box, expand=1):
    left = np.min(box[:, 0, 0]) * expand
    right = np.max(box[:, 0, 0]) * expand
    down = np.min(box[:, 0, 1]) * expand
    up = np.max(box[:, 0, 1]) * expand
    point3D_ids_in = []
    for xy, point3D_id in zip(img.xys, img.point3D_ids):
        if(point3D_id != -1 and xy[0] > left and xy[0] < right and xy[1] > down and xy[1] < up):
            point3D_ids_in.append(point3D_id)
    return point3D_ids_in

def get_plane_norm(points, Rvec, tvec):
    if(points.shape[0]<3):
        # return (tvec - points[0]).norm()
        return [np.nan]*3, points
    elif(points.shape[0]<4):
        n_w = np.cross(points[0]-points[1], points[0]-points[2])
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=4,
                                                    std_ratio=1.5)
        points = np.asarray(pcd.points)
        if(points.shape[0]<4):
            return [np.nan]*3, points 
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
        n_w = plane_model[0:3]
        # print("n_w ", n_w)
        # print("Rvec ", Rvec)
        points_inlier = pcd.select_by_index(inliers)
        points = np.asarray(points_inlier.points)
    n_c = np.dot(Rvec, n_w.T)
    # print("n_c:", n_c)
    if(n_c[2] > 0):
        return -n_w, points
    return n_w, points

def main(model_path, ocr_output_path):

    images = read_images_binary(os.path.join(model_path, "images.bin"))
    # print(len(images))
    for image_id in images:
        print(images[image_id].name)
        image_name = images[image_id].name
        Rvec = images[image_id].qvec2rotmat()
        tvec = images[image_id].tvec
        # print(Rvec)
        boxes, txts, scores = load_ocr_result(
            ocr_output_path + "/" + image_name + ".ocr_result.npy")
        if(len(boxes) == 0):
            continue
        for box_id in range(len(boxes)):
            if scores is not None and (scores[box_id] < 0.5 or math.isnan(scores[box_id])):
                continue
            box = np.reshape(
                np.array(boxes[box_id]), [-1, 1, 2]).astype(np.int64)
            cloud_path = ocr_output_path + image_name + str(box_id) + "_cloud3d_1.npy"
            if(os.path.exists(cloud_path)):
                points = np.load(cloud_path)
                n_p, points_inlier = get_plane_norm(points, Rvec, tvec)
                if(math.isnan(n_p[0])):
                    continue
            else:
                n_p = [np.nan]*3
                continue
            # flag_found_same = False
            # xyz = np.average(points_inlier, axis=0)
            # print("points_inlier ", points_inlier)
            property = [boxes[box_id], txts[box_id], scores[box_id], n_p, points_inlier, images[image_id].qvec, tvec]
            # for item in property:
            #     print(item)
            with open(ocr_output_path + "/" + image_name + str(box_id) + ".all_property.bin", "wb") as fp:
                pickle.dump(property, fp)

            

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
        # required=True,
        default='/home/ezxr/Documents/wxc/pic_ocr_flip/',
        type=str,
        help="visualized output paths"
    )
    parser.add_argument(
        "--model_path",
        # required=True,
        default='/home/ezxr/Documents/wxc/f1_gz/0',
        type=str,
        help="input colmap model paths"
    )
    args = parser.parse_args()
    main(args.model_path, args.ocr_output_path)
