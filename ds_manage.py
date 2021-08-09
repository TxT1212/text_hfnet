import glob2
import numpy as np
import sys, os
import shutil
import re
def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext), recursive=True)
    files.sort
    return files

def change_file_name(path='/home/ezxr/Documents/wxc/pic_ocr_flip/'):
    image_names = find_recursive(path, "expand_4.jpg")
    for img in image_names:
        # if('expand' not in img and 'mask' not in img and 'isFlip_1' not in img):
        #     img_new = re.sub('_isFlip.*', '.jpg', img)
        # if(1):
        #     img_new = re.sub('all_property.jpg', '.all_property_old.bin', img)        
        #     print(img)
        #     print(img_new)
        #     os.rename(img, img_new)
        # if(" " in img):
        #     img_new = re.sub(' ', '', img)        
        #     print(img)
        #     print(img_new)
        #     os.rename(img, img_new)
        img_new = re.sub('_isFlip.*', '.jpg', img)
        print(img_new)
        shutil.copyfile(img, img_new)
        
change_file_name('/home/ezxr/Documents/ibl_dataset_cvpr17_3852/query_images_ocr')
# change_file_name('/home/ezxr/Documents/wxc/query_ocr')
# change_file_name()
