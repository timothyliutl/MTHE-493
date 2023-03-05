#demonstration of execution time for c and python functions
from ctypes import *
import time
import numpy as np
my_functions = CDLL('cosq_class.so')
centroids = [3,4]
cent_len = len(centroids)
centroids = (c_float * len(centroids))(*centroids)

arr = [np.random.rand()*10 for i in range(100)]
print(arr)
arr_len = len(arr)
arr = (c_float * len(arr))(*arr)




#assert that the array len integer matches the length of the array
my_functions.calc_transition_probabilities.argtypes = (c_int, c_int, c_float, c_int)
my_functions.calc_transition_probabilities.restype = c_float

my_functions.expected_distortion.argtypes = (POINTER(c_float), c_int, c_float, c_int, c_float, c_int)
my_functions.expected_distortion.restype = c_float

print('expected distortion',my_functions.expected_distortion(centroids, cent_len, 0, 0, 0, 1))


my_functions.calc_partitions.argtypes = (POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, c_int, POINTER(c_int))
my_functions.calc_partitions.restype = POINTER(c_int)
return_arr = (c_int * arr_len)()
buffer = my_functions.calc_partitions(arr, arr_len, centroids, cent_len, 0.005, 1, return_arr)
arr1 = np.fromiter(buffer, dtype=np.int, count=len(arr))


arr_len = len(arr1)
arr1 = (c_int * len(arr1))(*arr1)
return_arr2 = (c_float * len(centroids))()
my_functions.calc_centroids.argtypes = (POINTER(c_int), c_int, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, c_int, POINTER(c_float))
my_functions.calc_centroids.restype = POINTER(c_int)
#return_iter = my_functions.calc_centroids(arr1, len(arr1), centroids, cent_len, arr, len(arr), 0.005, 1, return_arr2)
#print(np.fromiter(return_iter, dtype=np.float, count=len(centroids)))

my_functions.iteration.argtypes = (POINTER(c_float), c_int, POINTER(c_float), c_int, c_int, c_float, c_int)
my_functions.iteration.restype = POINTER(c_float)

my_functions.iteration(centroids, cent_len, arr, arr_len, 0, 0.005, 1)