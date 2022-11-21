import numpy as np
from matplotlib import pyplot as plt
import math
import random

R = 1         # Code Rate
N = 2**R
epsilon = 0.7 # Transition Probability (BSC)
bsc = [[1-epsilon, epsilon], [epsilon, 1-epsilon]]

mean, sd = 0, 1
numSamples = 10**5

def partitionDistortion(partition, centroid):
    distortion = 0
    for i in range(0, N):
        distortion = distortion + (partition[i] - centroid)**2
    return distortion/len(partition)

def calcDistortion(partitions, centroids):
    distortion = 0
    for i in range(0, N):
        for j in range(0, N):
            distortion = distortion + bsc[i][j] * partitionDistortion(partitions[i], centroids[j])
    return distortion

def calcPartitions(samples, centroids):
    partitions = [[] for i in range(0, N)] 
    for sample in samples:
        distortion = -1 
        partition = 0
        for i in range(0, N):
            newDistortion = 0
            for j in range(0, N):
                newDistortion = newDistortion + bsc[i][j] * (sample - centroids[j])**2
            if newDistortion < distortion or distortion == -1:
                distortion = newDistortion
                partition = i
        partitions[partition].append(sample)
    return partitions

def calcCentroids(partitions):
    centroids = [1] * N 
    numerator = 0
    denominator = 0
    for j in range(0, N):
        for i in range(0, N):
            numerator = numerator + bsc[i][j] * sum(partitions[i])/len(partitions[i])  
            denominator = denominator + bsc[i][j] * len(partitions[i])/numSamples 
        centroids[j] = numerator/denominator

    return centroids

def lloydAlgorithm(samples):
    for i in range(0,10):
        partitions = calcPartitions(samples, [-1, 1])
        centroids = calcCentroids(partitions)
        distortion = calcDistortion(partitions, centroids)
        #print(partitions)
        print(centroids)
        print([min(partitions[0]), max(partitions[0])])
        print([min(partitions[1]), max(partitions[1])])
        #print(distortion)
    return centroids, partitions

samples = np.random.normal(mean, sd, numSamples)
samples.sort()
lloydAlgorithm(samples)
