import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pickle 
import re

globaldesc = np.load('/home/txt/Downloads/ezxr_loc/text_hfnet/saved/F1_org_hfnet_globaldesc.npy')
image_names = np.load('/home/txt/Downloads/ezxr_loc/text_hfnet/saved/F1_org_hfnet_globalindex.npy')

db_index = []
q_index = []
for i in range(0, len(image_names)):
    name = image_names[i]
    if('images/' in name or '2019-04-16_16-14-48/images/' in name):
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

