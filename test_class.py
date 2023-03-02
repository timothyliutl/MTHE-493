import numpy as np
from matplotlib import pyplot as plt
from train import quantizer 
import math
import random
import os

# Test quantizer class

samples = np.random.normal(0, 1, 10000) 
rate = 4  
starting_centroids = np.sort(np.random.uniform(low=-5, high=5, size=(2**rate,)))
epsilon = 0.1

quant = quantizer(samples, rate, starting_centroids, epsilon)
# Run quantizer
quant.train()

#encode sample
encoded_sample = quant.encode(0.1)
print(encoded_sample)

#decode sample
decoded_sample = quant.decode(encoded_sample)
print(decoded_sample)


