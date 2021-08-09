import numpy as np
import math
# data = np.arange(6).reshape((3, 2))
# print(np.average(data, axis=0))
# print(data)
# aa = []
# for a in range(0,4):
#     b = np.ones([1,4])*a
#     aa.append(b)
#     print(aa)
# aa = np.array(aa)
# print(aa.squeeze())
# print(aa)
# def func1(a):
#     a = np.zeros(a.shape).copy()
# a = np.ones([4,3])
# b= np.zeros(a.shape)
# # print(a, b)

# func1(a)
# print(a)

# a = np.ones(4)
# b = np.ones(4) * 3
# c= a*b
# print(c)
# print(math.acos(0.85)/math.pi*180)
from collections import defaultdict
md = defaultdict(list)
md['a'] = 1
md['b'] = 2
md['c'] = 1
# print(md)
# print('c' in md)

# print([np.nan]*3)
md = defaultdict(list)
md['a'] = 1
md['b'] = 2
md['c'] = 1
# print(md)
# print('c' in md)

# print([np.nan]*3)

# import pickle
# with open('/home/ezxr/Documents/wxc/pic_ocr_flip/wxc_f1_20201214_f1n_route1_0001/00000020.jpg0.bin', "rb") as fp:   # Unpickling

#     f = pickle.load(fp)
#     print(f)

import glob
# for name in glob.glob('/home/ezxr/Documents/wxc/pic_ocr_flip/./wxc_f2_20201214_f2n_route4_0007/00000380.jpg0*.jpg'):
#     print(name)

# import pickle
# with open("/home/ezxr/Documents/wxc/pic_ocr_flip/wxc_f1_20201214_f1n_route3_0001/00000020.jpg5.all_property.bin", 'rb') as fb:
#     all_property = pickle.load(fb)
# print(all_property)
# box = np.array([[0, 0], [0, 4], [4, 4], [4, 0]])
# point = np.array([3  ,  2])
# box_diff = np.diff(box,axis = 0,prepend=box[-1:,:])
# cross_result = np.cross(box - point, box_diff) 
# print("box ", box)
# print("point ", point)
# print(cross_result)
# print( np.all(cross_result > 0) or np.all(cross_result < 0))

# [-11.37630485  10.93872276   7.58615937] [0.04460036 0.12256441 0.9914579 ]
# [-11.33171422  10.85702518   7.51468241] [ 0.11107713 -0.33811095  0.93452815]
# a=np.array([0.04460036, 0.12256441, 0.9914579 ])
# b=np.array([0.11107713, -0.33811095,  0.93452815 ])
# c = np.dot(a, b )
# print(c)
# [-42.79544779  -3.09540775   0.09234229] [ 0.01289239 -0.05016401 -0.99865778]
# [-42.71792618  -3.05539222   0.09327215] [ 0.02284204 -0.99956384  0.01871836]
# [-42.71792618  -3.05539222   0.09327215] [ 0.02284204 -0.99956384  0.01871836]
# #  0.13076392 -0.21808918  0.96712869
# a = []
# for i in range(10):
#     a.append([i] * 3)
#     a[i].append('d' * i)

# for item in a:
#     print(item)
#     if(2 in item):
#         a.remove(item)
#     print(item)

# print(a)

test_set = set()
test_set.add(1)
test_set.clear()
a =1
# print(a in test_set)
assert False

a = np.array([1, 2, 3])
b = np.array([2, 4, 5])
# print((a != a).any())

