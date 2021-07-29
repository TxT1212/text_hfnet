import numpy as np
import argparse
from read_write_model import *
import glob2
from database import *
# 1410.864258 0.000000 794.641113
# 0.000000 1408.981201 603.413025 (K)
# 0.000000 0.000000 1.000000
# 0.000000 0.000000 0.000000 (0)
# -0.807854 -0.008790 0.589317
# -0.589321 -0.002360 -0.807895 (R)
# 0.008492 -0.999959 -0.003274
# -6.638618 12.360680 0.033681 (t)
# 1632 1224
def find_history_k(K, Ks):
    for i in range(len(Ks)):
        K_i = Ks[i]
        if((K != K_i).any()):
            continue
        else:
            return i + 1
    return 0
db = COLMAPDatabase.connect("/home/ezxr/Documents/ibl_dataset_cvpr17_3852/colmap/database.db")
db.create_tables()

gt_query = "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/training_gt/"
ibl_gt = glob2.glob(os.path.join(gt_query, '*.camera'))
ibl_gt.sort()
camera_id = 0
image_id = 1
cameras = {}
images = {}
K_last = np.zeros([3,3])
Ks = []
for ibl_gt_i in ibl_gt:
    print(ibl_gt_i)
    f = open(ibl_gt_i, "r")
    lines = f.readlines()
    K_R_tvec = np.zeros([8, 3])
    for i in range(0, 8):
        nowline = lines[i].split()
        K_R_tvec[i][0] = float(nowline[0])
        K_R_tvec[i][1] = float(nowline[1])
        K_R_tvec[i][2] = float(nowline[2])
    nowline = lines[8].split()
    width = int(nowline[0])
    height = int(nowline[1])
    K = K_R_tvec[0:3, :]
    find_result = find_history_k(K, Ks)
    if(find_result == 0):
        Ks.append(K)
        camera_id=len(Ks)
        params = np.array([K[0][0], K[1][1], K[0][2], K[1][2]])

        cameras[camera_id] = Camera(id=camera_id,
                            model="PINHOLE",
                            width=width,
                            height=height,
                            params=np.array(params))
        db.add_camera(1, width, height, params)
    
    else:
        camera_id = find_result
    R = K_R_tvec[4:7, :].T
    tvec = -np.dot(R, K_R_tvec[7, :])
    img_name = ibl_gt_i.replace(gt_query, "").replace("camera", "jpg")
    images[image_id] = Image(
                    id=image_id, qvec=rotmat2qvec(R), tvec=tvec,
                    camera_id=camera_id, name=img_name,
                    xys=np.array([]), point3D_ids=np.array([]))   
    db.add_image(img_name, camera_id, rotmat2qvec(R), tvec, image_id)
    print(K)
    print(R)
    print(tvec)
    image_id += 1
write_images_text(images, "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/colmap/db/images.txt")
write_cameras_text(cameras, "/home/ezxr/Documents/ibl_dataset_cvpr17_3852/colmap/db/cameras.txt")




# For convenience, try creating all the tables upfront.

db.commit()
db.close()