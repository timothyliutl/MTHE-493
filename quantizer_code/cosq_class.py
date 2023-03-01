import numpy as np

class CoSQ:
    def __init__(self, epsilon, bits):
        self.epsilon = epsilon
        self.bits = bits
        self.centroids = []

    def training_set(self, training_input):
        self.training_set = training_input

    def fit(self):

        if not self.training_set:
            raise Exception("No training set has been imported you idiot sandwich")

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

        def calc_expected_distortion(centroids, index, point):
            distortion = 0
            for i in range(len(centroids)):
                trans_prob = calc_transition_prob(index, i)
                distortion = distortion + trans_prob*((centroids[i] - point)**2)
            return distortion

        def calc_partitions(points, centroids):
            partition_output = {centroid:[] for centroid in centroids}
            print(partition_output)
            for index, point in enumerate(points):
                closest_num = takeClosest(point, centroids)
                # centroid the quantizer would quantize to without a noisy channel
                # convert it to a number
                # maybe we can reorganize it so the items with the highest distortion have the lowest probabilities
                
                #index/number the point is quantized to
                index_centroid = centroids.index(closest_num)
                #array that calculates all the mean square distances for each centroid
                mse_array = []
                for input_index in range(len(centroids)):
                    #calculating expected distortion for a given quantization, input_index
                    mse_array.append(calc_expected_distortion(centroids, input_index, point))
                mse_array = np.array(mse_array)
                #find quantization with smallest distortion and append quantization to corresponding centroid in partition output
                index_smallest_distortion = np.argmin(mse_array)
                partition_output[centroids[index_smallest_distortion]].append(point)
            return partition_output
        
        def calc_centroids(partitions, centroids):
            new_centroids = []
            for index, centroid in enumerate(centroids):
                weighted_average = 0
                for i in range(len(centroids)):
                    trans_prob = calc_transition_prob(index, i)
                    weighted_average = weighted_average + trans_prob*((np.array(partitions[centroid])).mean())
                new_centroids.append(weighted_average)
            return new_centroids
        
        def iteration():
            pass
        #return calc_partitions(self.training_set, centroids)
        return calc_centroids(calc_partitions(self.training_set, centroids), centroids)
            

    def quantize(self, value):
        # quantize a given input value
        pass


