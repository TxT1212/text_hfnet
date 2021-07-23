import pickle
from tokenize import Floatnumber
import numpy as np
import re
# def SR():
#     pass
with open("/home/ezxr/Documents/wxc/pic_ocr_flip/items_3d.bin", 'rb') as fb:
    items_3d = pickle.load(fb)
# ocr_output_path = '/home/ezxr/Documents/wxc/pic_ocr_flip/'
# print(len(items_3d))
dict_img_3d = {}
for id in range(0, len(items_3d)):
    # print(items_3d)

    item_3d = items_3d[id]
    if(len(item_3d) < 1):
        continue
    # print(item_3d[0])
    # break
    for img in item_3d[2]:
        dict_img_3d[img] = id
for item in dict_img_3d:
    # print(item in dict_img_3d, item, dict_img_3d[item])
    pass
# print(len(dict_img_3d))
# with open(ocr_output_path + "dict_2dto3dindex.bin", "wb") as fp:
#     pickle.dump(dict_img_3d, fp)


def filter_hf_candidate(item_3d_now):
    pass


def read_nn_log(log_path='logs/log_pyr_4.txt'):
    f = open(log_path, "r")
    lines = f.readlines()
    dict = {}
    for line in lines:
        if(line[0] == '*'):
            tempstr = ''
            seq = line.split(' ')
            img_name = tempstr.join(seq[1:-1])
            img_name = re.sub('_isFlip.*', '.jpg', img_name)

            print(img_name)
            dict[query_image].append((img_name, seq[-1][:-1]))
            pass
        else:
            tempstr = ''
            query_image = line.split(" ")[1:]
            query_image = tempstr.join(query_image[:])[:-1]
            # query_image = re.sub('_isFlip.*', '.jpg', query_image)

            # print(query_image)
            dict[query_image] = []
    # for item in dict:
    #     print(item)
    #     for nn in dict[item]:
    #         print("**", nn)
    return dict

log_hfnet_nn = 'logs/log_pyr_4.txt'
dict_hfnet_nn = read_nn_log(log_hfnet_nn)


log_paddle_clas_ = 'logs/paddle_clas_.txt'
dict_paddle_clas_nn = read_nn_log(log_paddle_clas_)

item_3d_now = []
for query_img in dict_paddle_clas_nn:
    print(query_img)
    # print(dict_paddle_clas_nn[item])
    item_3d_now.clear()
    for nn_result, difference in dict_paddle_clas_nn[query_img]:
        if(float(difference) > 0.855 or nn_result not in dict_img_3d):
            # print("bad\t", nn_result, difference)
            continue
        print("good\t", nn_result, difference, dict_img_3d[nn_result])
        item_3d_now.append(dict_img_3d[nn_result])
    for nn_result, difference in dict_hfnet_nn[query_img]:
        if(nn_result not in dict_img_3d):
            continue
        dict_img_3d[nn_result]



# f = open(log_paddle_clas_, "r")
# lines = f.readlines()
# item_3d_now = []
# for line in lines:
    # if(line[0] == '*'):
    #     tempstr = ''
    #     seq = line.split(' ')
    #     img_name = tempstr.join(seq[1:-1])
    #     img_name = re.sub('_isFlip.*', '.jpg', img_name)

    #     difference = seq[-1][:-1]

    #     if(float(difference) > 0.855):
    #         # print(seq[0], img_name, seq[-1][:-1])
    #         continue
    #     else:
    #         # print(img_name, img_name in dict_img_3d)
    #         if(img_name in dict_img_3d):
    #             item_3d_now.append(dict_img_3d[img_name])
    #         pass
    # else:  # query image
    #     # filter_hf_candidate(item_3d_now)
    #     # print(item_3d_now)

    #     # print("query image:", line[:-1])
    #     item_3d_now.clear()

    #     pass
