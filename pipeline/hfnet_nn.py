import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 
import re
import argparse

parser = argparse.ArgumentParser(
        description="nn"
    )
parser.add_argument(
    "--input_path",
    required=True,
    type=str,
    help="描述子的保存路径",
    default='./saved/'
)
parser.add_argument(
    "--save_query_prefix",
    required=True,
    type=str,
    help="保存文件名的前缀",
    default=''
)
parser.add_argument(
    "--save_db_prefix",
    required=True,
    type=str,
    help="db保存文件名的前缀",
    default=''
)
parser.add_argument(
    "--db_image_token",
    type=str,
    help="把db图片和query区分开来的特征字符串，用于db和guery一同提取时",
    default=''
)
args = parser.parse_args()
globaldesc_path = args.input_path + args.save_db_prefix + 'hfnet_globaldesc.npy'
image_names_path = args.input_path + args.save_db_prefix + 'hfnet_globalindex.npy'
globaldesc = np.load(globaldesc_path)
image_names = np.load(image_names_path)

if(args.save_db_prefix == args.save_query_prefix):
    db_index = []
    q_index = []
    for i in range(0, len(image_names)):
        name = image_names[i]
        if(args.db_image_token in name):
        # if('2019-04-16_15-35-46/images/' in name or '2019-04-16_16-14-48/images/' in name):
            db_index.append(i)
        elif(True):
        # elif('09-49-05' in name):
            q_index.append(i)
            print(name)
    name_db = image_names[db_index]
    desc_db = globaldesc[db_index]
    image_names_query = image_names[q_index]
    globaldesc_query = globaldesc[q_index]
else:
    name_db = image_names
    desc_db = globaldesc
    globaldesc_path = args.input_path + args.save_query_prefix + 'hfnet_globaldesc.npy'
    image_names_path = args.input_path + args.save_query_prefix + 'hfnet_globalindex.npy'
    globaldesc_query = np.load(globaldesc_path)
    image_names_query = np.load(image_names_path)
# print(desc_db.shape, name_db.shape, image_names_query.shape, globaldesc_query.shape)

# ## fit
nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(desc_db)
knnPickle = open('saved/knn_model_F1_org', 'wb') 
pickle.dump(nbrs, knnPickle)


# ### load
# # nbrs = pickle.load(open('saved/knn_model_F1_org', 'rb'))

#### predict
distances, indices = nbrs.kneighbors(globaldesc_query) 



for i in range(0,image_names_query.size):
    pass
    names = [name_db[indices_now] for indices_now in indices[i]][0:20]
    for ii in range(0, 20):
        q = re.sub('[0-9]*_txt.*', '', image_names_query[i])
        d = re.sub('[0-9]*_txt.*', '', names[ii])
        print(q, d)
