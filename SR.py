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
        img = re.sub('/./', '/', img)
        dict_img_3d[img] = id
for item in dict_img_3d:
    # print(item in dict_img_3d, item, dict_img_3d[item])
    pass
# print(len(dict_img_3d))
# with open(ocr_output_path + "dict_2dto3dindex.bin", "wb") as fp:
#     pickle.dump(dict_img_3d, fp)


def item_3d_diff(item1, item2):
    xyz1 = item1[0]
    xyz2 = item2[0]
    n1 = item1[1]
    n2 = item2[1]
    distance = np.linalg.norm(xyz1- xyz2) 
    angle = np.dot(n1, n2)
    return angle, distance


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
            img_name = re.sub('/./', '/', img_name)

            # print(img_name)
            dict[query_image].append((img_name, seq[-1][:-1]))
            pass
        else:
            tempstr = ''
            query_image = line.split(" ")[1:]
            query_image = tempstr.join(query_image[:])[:-1]
            query_image = re.sub('_isFlip.*', '.jpg', query_image)
            query_image = re.sub('/./', '/', query_image)

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

items_3d_now = set()
paddle_clas_item=[]
for query_img in dict_paddle_clas_nn:
    print()
    print(query_img)
    # print(dict_paddle_clas_nn[item])
    items_3d_now.clear()
    paddle_clas_item.clear()
    for nn_result, difference in dict_paddle_clas_nn[query_img]:
        if(float(difference) > 0.85 or nn_result not in dict_img_3d):
            # print("bad\t", nn_result, difference)
            continue
        # print("good\t", nn_result, difference, dict_img_3d[nn_result])
        paddle_clas_item.append(nn_result)
        items_3d_now.add(dict_img_3d[nn_result])
    print("items_3d_now: ", items_3d_now, paddle_clas_item)
    
    if(len(items_3d_now)==0):
        print("Failed: no paddle_clas candidate, most likely candidate:")
        for nn_result, difference in dict_hfnet_nn[query_img]:
            if(nn_result not in dict_img_3d):
                continue
            print("**", nn_result, dict_img_3d[nn_result])
            break
        continue
    elif(len(items_3d_now)==1 and len(paddle_clas_item) > 3):
        print("Success: no duplicated logo: ")
        for nn_result, difference in dict_paddle_clas_nn[query_img]:
            if(float(difference) > 0.85 or nn_result not in dict_img_3d):
                continue
            print("**", nn_result, dict_img_3d[nn_result])
        continue     
    else:
        if(len(items_3d_now)<3 and len(paddle_clas_item) > 3):
            items_3d_now_list = list(items_3d_now)
            item1 = items_3d[items_3d_now_list[0]]
            item2 = items_3d[items_3d_now_list[1]]
            angle, distance = item_3d_diff(item1, item2) 
            if(angle > 0.5 and distance < 10):
                print("Success: Few duplicated logo:")
                for nn_result in paddle_clas_item:
                    print("**", nn_result, dict_img_3d[nn_result])
            else:
                print("Duplicated logos: ")
                for nn_result, difference in dict_hfnet_nn[query_img]:
                    if(nn_result not in dict_img_3d):
                        continue
                    if(dict_img_3d[nn_result] not in items_3d_now):
                        # print(dict_img_3d[nn_result])
                        continue
                    print("**", nn_result, dict_img_3d[nn_result])
        else:
            print("Duplicated logos: ")
            for nn_result, difference in dict_hfnet_nn[query_img]:
                # print(nn_result, nn_result not in dict_img_3d)
                if(nn_result not in dict_img_3d):
                    continue
                if(dict_img_3d[nn_result] not in items_3d_now):
                    # print(dict_img_3d[nn_result])
                    continue
                print("**", nn_result, dict_img_3d[nn_result])