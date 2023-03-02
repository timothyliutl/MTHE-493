import numpy as np
from matplotlib import pyplot as plt
import math
import random

class quantizer:
    def __init__(self, samples, rate, startingCentroids, epsilon):
        self.samples = samples # Training data
        self.rate = rate       # Code Rate
        self.N = 2**rate
        self.epsilon = epsilon # BSC transition probability
        self.bsc = [[1-epsilon, epsilon], [epsilon, 1-epsilon]]
        self.centroids = startingCentroids
        self.relativeDiffLim = 0.001

    def _transitionProb(self, i, j):
        # i and j converted to bit representation
        iBits = '{0:0{len}b}'.format(i, len=self.rate)
        jBits = '{0:0{len}b}'.format(j, len=self.rate)
        prob = 1
        # Calculate prob for memoryless channel
        for k in range(0, self.rate):
            prob = prob*self.bsc[int(iBits[k])][int(jBits[k])]
        return prob

    def _partitionDistortion(self, partition, centroid):
        distortion = 0
        for sample in partition:
            distortion = distortion + (sample - centroid)**2
        return distortion

    def _calcDistortion(self, partitions):
        distortion = 0
        for i in range(0, self.N):
            for j in range(0, self.N):
                distortion = distortion + self._transitionProb(i, j) * self._partitionDistortion(partitions[i], self.centroids[j]) 
        return distortion/len(self.samples)

    def _calcPartitions(self):
        partitions = [[] for i in range(0, self.N)]
        for sample in self.samples:
            distortion = -1
            partition = 0
            for i in range(0, self.N):
                newDistortion = 0
                for j in range(0, self.N):
                    newDistortion = newDistortion + self._transitionProb(i, j) * (sample - self.centroids[j])**2
                if newDistortion < distortion or distortion == -1:
                    distortion = newDistortion
                    partition = i
            partitions[partition].append(sample)
        return partitions

    def _calcCentroids(self, partitions):
        for j in range(0, self.N):
            numerator = 0
            denominator = 0
            for i in range(0, self.N):
                numerator = numerator + self._transitionProb(i, j) * sum(partitions[i])
                denominator = denominator + self._transitionProb(i, j) * len(partitions[i])
            if denominator != 0:
                self.centroids[j] = numerator/denominator

    # Train the quantizer -- final centroids stored in self.centroids
    def train(self):
        print("Training scalar quantizer...")
        distortion = []
        partitions = self._calcPartitions()
        distortion.append(self._calcDistortion(partitions))

        print("Starting Centroids: {}".format(self.centroids))
        print("Starting Distortion: {}".format(distortion[0]))

        i = 0
        while True:
            i = i + 1
            partitions = self._calcPartitions()
            self._calcCentroids(partitions)
            distortion.append(self._calcDistortion(partitions)) 
            if abs((distortion[i]-distortion[i-1])/distortion[i-1]) < self.relativeDiffLim:
                break

        print("Iterations: %d" % i)
        print("Final Centroids: {}".format(self.centroids))
        print("Final Distortion: {:.4f}".format(distortion[-1]))

    # Provide value to encode -- returns index of partition with lowest distortion
    def encode(self, val):
        distortion = -1
        partition = 0
        for i in range(0, self.N):
            newDistortion = 0
            for j in range(0, self.N):
                newDistortion = newDistortion + self._transitionProb(i, j) * (val - self.centroids[j])**2
            if newDistortion < distortion or distortion == -1:
                distortion = newDistortion
                partition = i
        return partition

    def decode(self, index):
        return self.centroids[index]
