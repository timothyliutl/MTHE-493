import numpy as np
from matplotlib import pyplot as plt
import math
import random
import os

R = 5         # Code Rate
N = 2**R 
num_samples = 10240

def read_pgm():
    pixel_vals = np.zeros(num_samples)
    error = 0 
    num_pics = 0
    i = 0
    for path in os.listdir("./org"):
        f = open("./org/" + path, 'rb')
        line = f.readline()
        # Check header is P5
        if line != str.encode('P5\n'):
            f.close() 
            continue
        f.readline()
        try: 
            (width, height) = [int(i) for i in f.readline().split()] 
            if width <= 32 or height <= 32:
                print("Error height {} width {}".format(height, width))
            rand_width = random.randint(0,width-32)
            rand_height = random.randint(0,height-32)
            
            if f.readline() != str.encode('255\n'):
                f.close()
                continue
            # only take 10x10 random sample from image
            for y in range(height):
                if y < rand_height or y >= rand_height + 32:
                    continue
                for x in range(width):
                    if x < rand_width or x >= rand_width + 32:
                        continue
                    i = i + 1
                    pixel_vals[i] = ord(f.read(1))
            f.close()
            num_pics = num_pics + 1
            if num_pics >= 10:
                break
        except:
            error = error + 1
            print("{} not formatted correctly\n".format(path))
            f.close()
            continue
    print(pixel_vals)
    print("Num pics: {}".format(num_pics))
    print("Num Errors: {}\n".format(error))
    print("pixel_vals length: {}\n".format(len(pixel_vals)))
    return pixel_vals

def transitionProb(i, j, bsc):
    # i and j converted to bit representation
    iBits = '{0:0{len}b}'.format(i, len=R)
    jBits = '{0:0{len}b}'.format(j, len=R)
    prob = 1
    # Calculate prob for memoryless channel
    for k in range(0, R):
        prob = prob*bsc[int(iBits[k])][int(jBits[k])]
    return prob

def partitionDistortion(partition, centroid):
    distortion = 0
    for sample in partition:
        distortion = distortion + (sample - centroid)**2
    return distortion

def calcDistortion(partitions, centroids, bsc):
    distortion = 0
    for i in range(0, N):
        for j in range(0, N):
            distortion = distortion + transitionProb(i, j, bsc) * partitionDistortion(partitions[i], centroids[j])
    return distortion/len(samples)

def calcPartitions(samples, centroids, bsc):
    partitions = [[] for i in range(0, N)]
    for sample in samples:
        distortion = -1
        partition = 0
        for i in range(0, N):
            newDistortion = 0
            for j in range(0, N):
                newDistortion = newDistortion + transitionProb(i, j, bsc) * (sample - centroids[j])**2
            if newDistortion < distortion or distortion == -1:
                distortion = newDistortion
                partition = i
        partitions[partition].append(sample)
    return partitions

def calcCentroids(partitions, bsc):
    centroids = [0] * N
    for j in range(0, N):
        numerator = 0
        denominator = 0
        for i in range(0, N):
            numerator = numerator + transitionProb(i, j, bsc) * sum(partitions[i])
            denominator = denominator + transitionProb(i, j, bsc) * len(partitions[i])
        if denominator != 0:
            centroids[j] = numerator/denominator
    return centroids

def lloydAlgorithm(samples, bsc, startingCentroids):
    distortion = []
    centroids = startingCentroids
    partitions = calcPartitions(samples, centroids, bsc)
    distortion.append(calcDistortion(partitions, centroids, bsc))

    print("Starting Centroids: {}".format(centroids))
    print("Starting Distortion: {}".format(distortion[0]))

    centroidMovement = [[] for i in range(0,N)]
    for j in range(0,N):
        centroidMovement[j].append(centroids[j])

    i = 0
    while True:
        i = i + 1
        partitions = calcPartitions(samples, centroids, bsc)
        centroids = calcCentroids(partitions, bsc)
        for j in range(0,N):
            centroidMovement[j].append(centroids[j])
        distortion.append(calcDistortion(partitions, centroids, bsc))
        if abs((distortion[i]-distortion[i-1])/distortion[i-1]) < 0.0001 or i >= 10:
            break

    for i in range(0,N):
        partitions[i] = list(set(partitions[i]))
        partitions[i] = list(map(round, partitions[i]))

    centroids = list(map(round, centroids))
    print(type(centroids[0]))

    print("Iterations: %d" % i)
    print("Final Centroids: {}".format(centroids))
    print("Final Distortion: {:.4f}".format(distortion[-1]))
    
    return [centroids, partitions, distortion]

# RUN
distortion = []
samples = read_pgm()
print(max(samples))
print(min(samples))
for i, epsilon in enumerate(np.arange(0, 0.2, 0.1)):
    bsc = [[1-epsilon, epsilon], [epsilon, 1-epsilon]]
    print("\nRunning Lloyds for Epsilon = %f" % epsilon)
    starting = np.sort(np.random.uniform(low=5, high=100, size=(N,)))
    print(starting)
    [centroids, partitions, distortion] =  lloydAlgorithm(samples, bsc, starting)
    np.savetxt("centroids_{}.txt".format(i), np.array(centroids, dtype=int), fmt='%i')
    np.savetxt("partitions_{}.txt".format(i), np.array([xi+[-1]*(255-len(xi)) for xi in partitions], dtype=int), fmt='%i')


