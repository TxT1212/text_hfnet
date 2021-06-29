###################################################
#  input: colmap points.bin images.bin
#         paddle ocr imageXXX.jpg_ocr_result.npy
# output: xyz \in R3 质心
#         xyz*4 \in R12 包围圈
####################################################
import numpy as np
import argparse

def load_ocr_result(result_path):
    result = np.load(result_path)
    
    boxes = [line[0] for line in result]
    return boxes

def load_images(images_bin):
    pass

# 
def choose_text_points2d(image, box):
    pass

def load_points(points_bin):
    pass


def main():
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="input picture paths"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="visualized output paths"
    )    
    main()