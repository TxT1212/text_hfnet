import numpy as np
import cv2 
image_names = np.load('saved/database_hfnet_globalindex.npy')
image_names_query = np.load('saved/query_hfnet_globalindex.npy')

globaldesc = np.load('saved/database_hfnet_globaldesc.npy')
globaldesc_query = np.load('saved/query_hfnet_globaldesc.npy')
print(image_names_query[115])
print(image_names_query[116])
print(image_names_query[117])
print(image_names[14952])

d1 = globaldesc[14952]
d2 = globaldesc_query[116]
d4 = globaldesc_query[117]
d = np.linalg.norm(d1 - d2, ord=2) 
print(d)

d3 = globaldesc[35166]
print(image_names[35165])
print(image_names[35166])
print(image_names[35167])

d = np.linalg.norm(d3 - d2, ord=2) 
print(d)

d = np.linalg.norm(d3 - d1, ord=2) 
print(d)

d = np.linalg.norm(d2 - d4, ord=2) 
print(d)


image_names_test = np.load('saved/test_hfnet_globalindex.npy')
globaldesc_test = np.load('saved/test_hfnet_globaldesc.npy')

print(image_names_test)
d_t0 = globaldesc_test[0]
d_t1 = globaldesc_test[1]
d_t2 = globaldesc_test[2]

print(np.linalg.norm(d_t0 - d_t1, ord=2))
print(np.linalg.norm(d_t0 - d_t2, ord=2))
print(np.linalg.norm(d_t1 - d_t2, ord=2))

print(np.linalg.norm(d_t1 - d2, ord=2))

# p0 = cv2.imread(image_names_test[0], -1)
# p1 = cv2.imread(image_names_test[1], -1)
# p2 = cv2.imread(image_names_test[2], -1)

# p0 = cv2.resize(p0, (640, 480))
# p1 = cv2.resize(p1, (640, 480))
# p2 = cv2.resize(p2, (640, 480))

# cv2.imwrite(image_names_test[0], p0)
# cv2.imwrite(image_names_test[2], p2)
# cv2.imwrite(image_names_test[1], p1)

print(np.linalg.norm(d_t0 - d1, ord=2))
