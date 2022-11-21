import numpy as np
from matplotlib import pyplot as plt
import math
import random

R = 1         # Code Rate
N = 2**R
epsilon = 0.1 # Transition Probability (BSC)
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
    centroids = [0] * N 
    for j in range(0, N):
        numerator = 0
        denominator = 0
        for i in range(0, N):
            numerator = numerator + bsc[i][j] * sum(partitions[i])  
            denominator = denominator + bsc[i][j] * len(partitions[i])
        centroids[j] = numerator/denominator
    return centroids

def lloydAlgorithm(samples):
    distortion = [] 
    centroidMovement = [[] for i in range(0,N)]
    centroids = [-0.5,1]
    centroidMovement[0].append(centroids[0])
    centroidMovement[1].append(centroids[1])
    for i in range(0,10):
        partitions = calcPartitions(samples, centroids)
        centroids = calcCentroids(partitions)
        centroidMovement[0].append(centroids[0])
        centroidMovement[1].append(centroids[1])
        distortion.append(calcDistortion(partitions, centroids))
        print(centroids)
    plt.plot(distortion)
    plt.plot(centroidMovement[0])
    plt.plot(centroidMovement[1])
    plt.show()
    plt.savefig("distortion.png")
    return centroids, partitions

samples = np.random.normal(mean, sd, numSamples)
samples.sort()
lloydAlgorithm(samples)
