# cp it to paddleClas/deploy/python
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import copy
import cv2
import numpy as np
from tqdm import tqdm

from python.predict_rec import RecPredictor
from vector_search import Graph_Index

from utils import logger
from utils import config
import glob2

def find_recursive(root_dir, ext='.jpg'):
    files = glob2.glob(os.path.join(root_dir, './**/*'+ext), recursive=True)
    files.sort()
    return files


class GalleryBuilder(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.build(config['IndexProcess'])

    def build(self, config):
        '''
            build index from scratch
        '''
        gallery_images = find_recursive(config['image_root'], 'expand_0.jpg')

        # extract gallery features
        gallery_features = np.zeros(
            [len(gallery_images), config['embedding_size']], dtype=np.float32)

        for i, image_file in enumerate(tqdm(gallery_images)):
            img = cv2.imread(image_file)
            if img is None:
                logger.error("img empty, please check {}".format(image_file))
                exit()
            img = img[:, :, ::-1]
            rec_feat = self.rec_predictor.predict(img)
            gallery_features[i, :] = rec_feat
        np.save(config['gallery_features_outpath'], gallery_features)
        np.save(config['gallery_images_outpath'], gallery_images)
        for img in gallery_images:
            print(img)



def main(config):
    system_builder = GalleryBuilder(config)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
