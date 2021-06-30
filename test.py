import numpy as np
data = np.arange(6).reshape((3,2))
print(np.average(data, axis=0))
print(data)