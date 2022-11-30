import numpy as np
from matplotlib import pyplot as plt
import math
import random

R = 1         # Code Rate
N = 2**R

mean, sd = 0, 1
numSamples = 10**5

def partitionDistortion(partition, centroid):
    distortion = 0
    for sample in partition:
        distortion = distortion + (sample - centroid)**2
    return distortion

def calcDistortion(partitions, centroids, bsc):
    distortion = 0
    for i in range(0, N):
        for j in range(0, N):
            distortion = distortion + bsc[i][j] * partitionDistortion(partitions[i], centroids[j])
    return distortion/numSamples

def calcPartitions(samples, centroids, bsc):
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

def calcCentroids(partitions, bsc):
    centroids = [0] * N 
    for j in range(0, N):
        numerator = 0
        denominator = 0
        for i in range(0, N):
            numerator = numerator + bsc[i][j] * sum(partitions[i])  
            denominator = denominator + bsc[i][j] * len(partitions[i])
        centroids[j] = numerator/denominator
    return centroids

def lloydAlgorithm(samples, bsc):
    distortion = [] 
    centroids = [-4, 1]
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
        if abs((distortion[i]-distortion[i-1])/distortion[i-1]) < 0.0001:
            break

    print("Iterations: %d" % i)
    print("Final Centroids: {}".format(centroids))
    print("Final Distortion: \033[92m%.4f\033[0m" % distortion[-1])

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(distortion, label="Distortion", color=color)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Distortion", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(centroidMovement[0], label="y0", color=color)
    ax2.plot(centroidMovement[1], label="y1", color=color)
    ax2.set_ylabel("Centroid Position", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.show()
    plt.savefig("distortion.png")
    return [centroids, partitions]

##
# Testing actual encoder/decoder (prototype)
#def encoder(partitions, message):
#    for i in range(0, N):
#        for sample in partitions[i]:
#            if message == sample:
#                return i
#    print("message not found")
#
#def decoder(centroids, codeword):
#    return centroids[codeword]
#
samples = np.random.normal(mean, sd, numSamples)
epsilon = 0.0 # Transition Probability (BSC)
bsc = [[1-epsilon, epsilon], [epsilon, 1-epsilon]]
print("Running Lloyds for Epsilon = %f" % epsilon)
[centroids0, partitions0] = lloydAlgorithm(samples, bsc)
epsilon = 0.1
bsc = [[1-epsilon, epsilon], [epsilon, 1-epsilon]]
print("\n\nRunning Lloyds for Epsilon = %f" % epsilon)
[centroids1, partitions1] = lloydAlgorithm(samples, bsc)
print("\n\nDistortion of noiseless encoder under noisy channel: \033[92m%.4f\033[0m" % calcDistortion(partitions0, centroids0, bsc))

#distortion = 0
#for msg in samples[0:1000]:
#    codeword = encoder(partitions, msg)
#    noise = np.random.binomial(n = 1, p = epsilon)
#    receivedCodeword = (codeword + noise) % N 
#    decodedMsg = decoder(centroids, receivedCodeword)
#    distortion = distortion + (decodedMsg-msg)**2
#
#distortion = distortion/1000
#print("Empirical Distortion: %f" % distortion)
#
#
#distortion = 0
#for msg in samples[0:1000]:
#    if(msg > 0):
#        codeword = 1
#    else:
#        codeword = 0
#    noise = np.random.binomial(n = 1, p = epsilon)
#    receivedCodeword = (codeword + noise) % N 
#    if(receivedCodeword > 0):
#        decodedMsg = math.sqrt(2/math.pi)
#    else:
#        decodeMsg = -1*math.sqrt(2/math.pi)
#    distortion = distortion + (decodedMsg-msg)**2
#
#distortion = distortion/1000
#print("Empirical Distortion of noiseless encoder: %f" % distortion)