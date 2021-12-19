import pickle
pickle_in = open("dict.pickle" , "rb")
example_dict = pickle.load(pickle_in)

from sklearn import preprocessing
from numpy.linalg import eigh
import numpy as np
data = preprocessing.normalize(example_dict)
covariance = (data.T @ data)/3600
values, vectors = eigh(covariance)
p = vectors[np.argsort(values)[-50:]]
x_cap = data @ p.T

numpy_image = np.asarray(p)
example = numpy_image
pickle_out = open("dict2.pickle","wb")
pickle.dump(example,pickle_out)
pickle_out.close()

numpy_image = np.asarray(x_cap)
example = numpy_image
pickle_out = open("dict5.pickle","wb")
pickle.dump(example,pickle_out)
pickle_out.close()
