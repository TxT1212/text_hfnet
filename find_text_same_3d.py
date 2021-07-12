import numpy as np
from numpy.core.defchararray import array

areas = np.load('/home/ezxr/Documents/wxc/pic_ocr_flip/dict_test2xyz.npy', allow_pickle=True)

dict_test2xyz = areas.item()
# print(type(dict_test2xyz))
items_3d = []
d_threshold = 0.4
for test_xyz in dict_test2xyz:
    xyz = dict_test2xyz[test_xyz]
    test = test_xyz
    # print(type(xyz))
    flag_found_same = False
    for item_3d in items_3d:
        dist = np.linalg.norm(xyz - item_3d[0]) 
        if(dist > d_threshold):
            continue
        else:
            flag_found_same = True
            item_3d[1].append(test)
            break
    if(not flag_found_same):
        items_3d.append([xyz, [test]])
        
for item_3d in items_3d:
    print(item_3d)