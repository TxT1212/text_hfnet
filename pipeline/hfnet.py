# -*-coding: utf-8 -*-
from __future__ import print_function
from sklearn.decomposition import PCA
import heapq
import tensorflow as tf
#import imageio
import numpy as np
#from create_tf_record import *
# from tensorflow.python.framework import graph_util
import cv2
import scipy.io as sio
import os
import re
import glob2
import pickle
import time
tf.contrib.resampler
# from inference_landmarkar_pb import *
# from read_write_model import *

np.set_printoptions(threshold=np.inf)

h = 256  # input h for posenet
w = 320  # input w for posenet
depths = 3
USE_DENSEXYZ = False
database_BY_POSENET = True
database_BY_HFNET = True
USE_HFLOCAL = False
Reduce_PCA_hfnet = True
Reduce_PCA_posenet = True
use_pose_refine_database = True
USE_COLMAP_SIFT = False

outputs = ['global_descriptor', 'keypoints', 'local_descriptors']


def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')
    assert points_2D.shape[0] == points_3D.shape[0], 'points 3D and points 2D must have same number of vertices'
    #print ('3d points: ', points_3D)
    #print ('2d points: ', np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)))
    # SIGN, R_exp, t = cv2.solvePnP(points_3D,
    #                           # points_2D,
    #                           np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
    #                           cameraMatrix,
    #                           distCoeffs)
    #                           # , None, None, False, cv2.SOLVEPNP_UPNP)

    SIGN, R_exp, t, inliers = cv2.solvePnPRansac(points_3D,
                                                 points_2D,
                                                 cameraMatrix,
                                                 distCoeffs,
                                                 reprojectionError=6.0)

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    print('!!!!!!!!!!solvePnP sign:  ,', SIGN, '    inliers: ', len(inliers))
    return R, t  # ,R_exp


def distancelessthresh(chose_2d, db_cand_p2d, thres):
    dif_max = np.inf
    indx = -1
    for i in range(len(db_cand_p2d)):
        cand2d = db_cand_p2d[i]
        diff = np.sqrt((chose_2d[0] - cand2d[0])**2 +
                       (chose_2d[1] - cand2d[1])**2)
        if (diff < dif_max):
            dif_max = diff
            indx = i
    if(dif_max < thres):
        return indx
    return -1


def baseline_sift_matching(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)

    good = [[m] for m, n in matches if m.distance < 0.7*n.distance]
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                              # (0, 255, 0)
                              matchColor=(-1, -1, -1), matchesMask=None,
                              singlePointColor=(255, 0, 0), flags=0)
    return img3


def debug_matching(frame1, frame2, path_image1, path_image2, result_dir, base_name1, base_name2, save_name, matches,
                   matches_mask, num_points, use_ratio_test, if_resize=True):
    img1 = cv2.imread(path_image1, 0)
    img2 = cv2.imread(path_image2, 0)

    if if_resize:
        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))

    kp1 = get_ocv_kpts_from_np(frame1['keypoints'][:num_points, :])
    kp2 = get_ocv_kpts_from_np(frame2['keypoints'][:num_points, :])

    if use_ratio_test:
        img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                 matchColor=(-1, -1, -1),  # (0, 255, 0)
                                 matchesMask=matches_mask,
                                 singlePointColor=(255, 0, 0), flags=0)
    else:
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              matchColor=(-1, -1, -1),  # (0, 255, 0)
                              singlePointColor=(255, 0, 0), flags=0)

    img_sift = baseline_sift_matching(img1, img2)

    '''
    fig = plt.figure(figsize=(2, 1))
    fig.add_subplot(2, 1, 1)
    plt.imshow(img)
    plt.title('Custom features')
    fig.add_subplot(2, 1, 2)
    plt.imshow(img_sift)
    plt.title('SIFT')
    plt.show()
    '''

    h, w = img.shape[:2]
    dst_h, dst_w = 480, 640*2

    if dst_h != h or dst_w != w:
        img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        img_sift = cv2.resize(img_sift, (dst_w, dst_h),
                              interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(result_dir, base_name1 + '_' +
                             base_name2 + '_' + save_name + '_features.png'), img)
    cv2.imwrite(os.path.join(result_dir, base_name1 +
                             '_' + base_name2 + '_sift.png'), img_sift)

    print(os.path.join(result_dir, base_name1 + '_' +
                       base_name2 + '_' + save_name + '_features.png'))


def get_ocv_kpts_from_np(keypoints_np):
    return [cv2.KeyPoint(x=x, y=y, _size=1) for x, y in keypoints_np]


def match_frames(des1, des2, path_image1, path_image2, result_dir, base_name1, base_name2, save_name, num_points,
                 use_ratio_test, ratio_test_values, debug):
    print(des1.shape, des2.shape)
    if use_ratio_test:
        keypoint_matches = [[] for _ in ratio_test_values]
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)
        smallest_distances = [dict() for _ in ratio_test_values]
        matches_mask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            for ratio_idx, ratio in enumerate(ratio_test_values):
                if m.distance < ratio * n.distance:
                    if m.trainIdx not in smallest_distances[ratio_idx]:
                        smallest_distances[ratio_idx][m.trainIdx] = (
                            m.distance, m.databaseIdx)
                        matches_mask[i] = [1, 0]
                        keypoint_matches[ratio_idx].append(
                            (m.databaseIdx, m.trainIdx))
                    else:
                        old_dist, old_databaseIdx = smallest_distances[
                            ratio_idx][m.trainIdx]
                        if m.distance < old_dist:
                            old_distance, old_databaseIdx = smallest_distances[
                                ratio_idx][m.trainIdx]
                            smallest_distances[ratio_idx][m.trainIdx] = (
                                m.distance, m.databaseIdx)
                            matches_mask[i] = [1, 0]
                            keypoint_matches[ratio_idx].remove(
                                (old_databaseIdx, m.trainIdx))
                            keypoint_matches[ratio_idx].append(
                                (m.databaseIdx, m.trainIdx))
    else:
        keypoint_matches = [[]]
        matches_mask = []
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)

        # Matches are already cross-checked.
        for match in matches:
            # match.trainIdx belongs to des2.
            keypoint_matches[0].append((match.databaseIdx, match.trainIdx))

    if debug:
        # debug_matching(frame1, frame2, path_image1, path_image2, result_dir, base_name1, base_name2, save_name, matches, matches_mask, num_points, use_ratio_test, if_resize=True)
        pass

    return keypoint_matches


def freeze_graph_images(pb_path, image_path, save_dir, prefix, model):
    '''
    : run this func to save the db globaldesc in npy. 
    : posenet or hfnet 
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    if (model == 'hfnet'):
        pb_path = yolo_pb_path_
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            # with open(pb_path, "rb") as f:
            with tf.gfile.GFile(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())

                tf.import_graph_def(output_graph_def, name="")
                tensor_name = [tensor.name for tensor in output_graph_def.node]
                print(tensor_name)
                print('---------------------------')
                # for op in graph.get_operations():
                #    # print出tensor的name和值
                #    print(op.name, op.values())
            #config = tf.ConfigProto()
            config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                # 定义输入的张量名称,对应网络结构的输入张量
                # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
                input_image_tensor = sess.graph.get_tensor_by_name("image:0")
                #input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

                # 定义输出的张量名称
                output_tensor_name = {
                    n: sess.graph.get_tensor_by_name(n+':0')[0] for n in outputs}
                #hf_keypoints_mat = []
                #hf_global_desc_mat = []
                #hf_global_index_dict = []
                #hf_local_desc_mat = []
                print('start read imgs')
                if os.path.isfile(image_path) and '.txt' in image_path:
                    file = open(image_path, 'r')
                    lines = file.readlines()
                    lines = sorted(lines)
                elif os.path.isdir(image_path):
                    lines = glob2.glob(os.path.join(
                        image_path, './**/*'+'.jpg'))
                        # image_path, './**/*'+'expand_4.jpg'))
                    lines = sorted(lines)
                else:
                    fd = open(image_path, 'rb')
                    gt_pkl = pickle.load(fd)
                    whole_sample_nums = len(gt_pkl)
                    keys = list(gt_pkl.keys())
                    lines = sorted(keys)
                # pathlist = os.listdir(image_path)
                # for pppath in pathlist:
                #     pppath = os.path.join(image_path, pppath)
                #     if os.path.isdir(pppath):
                #         lines = glob.glob(os.path.join(pppath, '*.jpg'))
                #         lines=sorted(lines)
                #         print('pppath ',pppath)
                # print("lines:", lines)
                run_index = -1
                for line in lines:
                    run_index = run_index + 1
                    database_img_n = line.strip()
                    image = cv2.imread(database_img_n, 3)[:, :, ::-1]
                    image = cv2.resize(image, (640, 480),
                                       interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow("1", image)
                    # cv2.waitKey()
                    net_input = np.expand_dims(
                        image[..., ::-1].astype(np.float), axis=0)
                    input_map = {input_image_tensor: net_input}
                    hfout = sess.run(output_tensor_name, feed_dict=input_map)
                    global_descriptor = hfout['global_descriptor']
                    keypoints = hfout['keypoints']
                    local_descriptors = hfout['local_descriptors']
                    hf_global_desc_mat.append(global_descriptor)
                    hf_global_index_dict.append(
                        database_img_n.split('data/')[-1])
                    hf_local_desc_mat.append(local_descriptors)
                    hf_keypoints_mat.append(keypoints)
                np.save(save_dir+prefix+'_'+'hfnet'+'_globaldesc.npy',
                        np.array(hf_global_desc_mat, np.float32))
                np.save(save_dir+prefix+'_'+'hfnet' +
                        '_globalindex.npy', hf_global_index_dict)
                print('saved global!')
                # np.save(save_dir+prefix+'_'+'hfnet' +
                #         '_localdesc.npy', hf_local_desc_mat)
                # np.save(save_dir+prefix+'_'+'hfnet' +
                #         '_keypoints.npy', hf_keypoints_mat)
                print('saved all!')


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    # model path for posenet and hfnet
    yolo_pb_path_ = "/home/mm/ARWorkspace/ARResearch/text_hfnet/hfnet_total.pb"  # hfnet_c11_50000
    # mv3_pb_path_ = "./models/posenet/270000_frozen.pb"
    # image info of database data
    image_path_ = '/data/largescene/B1_lining/images/wxc_b1_1214test_lining_route1_0001'
  #  image_path_ = '/media/ezxr/data/nevar/HyundaiDepartmentStore_ocr/1F/mapping/mapping_image_list_3.txt'
  #  image_path_ = "/media/txt/data2/naver/ocr/4F/release/"
    # image_path_ = '/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_image_ocr/'
    # image_path_ = '/home/ezxr/Documents/wxc/pic_ocr_flip/f2'
    # image_path_ = '/home/ezxr/Documents/tem/'
    # image_path_ = '/home/ezxr/Documents/wxc/query_ocr'
    # image_path_ = './data/xrayforest/list.txt'
    # image_camera_path = './data/xrayforest/cameras.bin'
    # parameter used for global database using posenet or hfnet in db data
    # db_posenet = './saved/database_posenet_globaldesc.npy'
    # db_posenet_index = './saved/database_posenet_globalindex.npy'
    # db_hfnet_global = './saved/database_hfnet_globaldesc.npy'
    # db_hfnet_local = './saved/database_hfnet_localdesc.npy'
    # db_hfnet_keypoint = './saved/database_hfnet_keypoints.npy'
    # db_hfnet_index = './saved/database_hfnet_globalindex.npy'
    # output dir , saved results
    save_dir = './saved/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global_desc_mat = []
    global_index_dict = []
    xyz_pre_gt = []
    q_pre_gt = []
    hf_keypoints_mat = []
    hf_global_desc_mat = []
    hf_global_index_dict = []
    hf_local_desc_mat = []

    # freeze_graph_images(pb_path=yolo_pb_path_, image_path=image_path_,
    #                     save_dir=save_dir, prefix='4F', model='hfnet')
 #   image_path_ = "/media/txt/data2/naver/ocr/B1/release/"
  #  image_path_ =  "/media/txt/data2/naver/HyundaiDepartmentStore/1F/release"
    freeze_graph_images(pb_path=yolo_pb_path_, image_path=image_path_,
                        save_dir=save_dir, prefix='F1_org', model='hfnet')
