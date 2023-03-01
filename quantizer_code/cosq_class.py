import numpy as np

class CoSQ:
    def __init__(self, epsilon, bits):
        self.epsilon = epsilon
        self.bits = bits

    def training_set(self, training_input):
        self.training_set = training_input

    def fit(self):

        if not self.training_set:
            raise Exception("No training set has been imported")

        #train quantizer using lloyd max algorithm and memoryless channel
        num_centroids = 2**self.bits
        centroids = [np.random.randint(0,255) for i in range(num_centroids)]
        partition_list = {}
        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))

        def calc_transition_prob(input, output):
            #expecting input and output as numbers and uses epsilon defined in initialization
            input_bin = str(bin(input))[2:]
            output_bin = str(bin(output))[2:]

            prob = 1

            if len(input_bin)> len(output_bin):
                difference = len(input_bin) - len(output_bin)
                output_bin = '0'*difference + output_bin
            if len(output_bin) > len(input_bin):
                difference = len(output_bin) - len(input_bin)
                input_bin = '0'*difference + input_bin

            for i in range(len(input_bin)):
                if input_bin[i] == output_bin[i]:
                    prob = prob * (1-self.epsilon)
                else:
                    prob = prob * self.epsilon
            return prob



        def calc_partitions(points, centroids):
            partition_output = {centroid:[] for centroid in centroids}
            print(partition_output)
            for index, point in enumerate(points):
                closest_num = takeClosest(point, centroids)
                partition_output[closest_num].append(point)
            return partition_output
        
        def calc_centroids(partitions, centroids):
            new_centroids = []
            for centroid in centroids:
                new_centroids.append(np.array(partitions[centroid]).mean())
            return new_centroids
        
        
        
        def iteration():
            pass

        return calc_transition_prob(10,12)
            

    def quantize(self, value):
        # quantize a given input value
        pass


