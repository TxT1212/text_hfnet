import os
import sys
import numpy as np
import glob2
import re

hloc_trac = '/media/txt/data2/naver/outputs/4F/hloc_superpoint+superglue_netvlad50.txt'
cameras = '/media/txt/data2/naver/HyundaiDepartmentStore/4F/release/test/sensors/records_camera.txt'
f = open(cameras, 'r')
img_name_full = {}
lines_c = open(cameras, 'r')
for line in lines_c:
    if(len(line) == 0 or line[0] == '#'):
        continue
    now_name = line.split(', ')[-1]
    # print(now_name.split("/")[-1].replace('\n', ''))
    img_name_full[now_name.split("/")[-1].replace('\n', '')] = now_name.replace('\n', '')



f = open(hloc_trac, 'r')
lines = f.readlines()
for line in lines:
    if(len(line) == 0 or line[0] == '#'):
        continue
    now_l = line.strip('\n').split(' ')
    # print(now_l)
    print(img_name_full[now_l[0]], *now_l[1:])
    