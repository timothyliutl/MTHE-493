#demonstration of execution time for c and python functions
from ctypes import *
import time
st = time.time()
my_functions = CDLL('cosq_class.so')
et = time.time()

st2 = time.time()
for i in range(100000):
    print(i)
et2 = time.time()



print(et-st, et2-st2)