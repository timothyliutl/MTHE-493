import numpy as np

class CoSQ:
    def __init__(self, epsilon, bits):
        self.epsilon = epsilon
        self.bits = bits
        self.centroids = []
        self.centroid_map = {}

    def training_set(self, training_input):
        self.training_set = training_input

    def __calc_transition_prob(self, input, output):
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
    
    def __calc_expected_distortion(self, centroids, index, point):
            distortion = 0
            for i in range(len(centroids)):
                trans_prob = self.__calc_transition_prob(index, i)
                distortion = distortion + trans_prob*((centroids[i] - point)**2)
            return distortion
    
    def __calc_partitions(self, points, centroids):
            partition_output = {centroid:[] for centroid in centroids}
            for index, point in enumerate(points):
                # centroid the quantizer would quantize to without a noisy channel
                # convert it to a number
                # maybe we can reorganize it so the items with the highest distortion have the lowest probabilities
                #index/number the point is quantized to
                #array that calculates all the mean square distances for each centroid
                mse_array = []
                for input_index in range(len(centroids)):
                    #calculating expected distortion for a given quantization, input_index
                    mse_array.append(self.__calc_expected_distortion(centroids, input_index, point))
                mse_array = np.array(mse_array)
                #find quantization with smallest distortion and append quantization to corresponding centroid in partition output
                index_smallest_distortion = np.argmin(mse_array)
                partition_output[centroids[index_smallest_distortion]].append(point)
            return partition_output
    
    

    def fit(self):

        if len(self.training_set)==0:
            raise Exception("No training set has been imported you idiot sandwich")

        #train quantizer using lloyd max algorithm and memoryless channel
        num_centroids = 2**self.bits
        max_val = np.array(self.training_set).max()
        centroids = [np.random.randint(0, max_val) for i in range(num_centroids)]
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
            for index, point in enumerate(points):

                # centroid the quantizer would quantize to without a noisy channel
                # convert it to a number
                # maybe we can reorganize it so the items with the highest distortion have the lowest probabilities
                
                #index/number the point is quantized to

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
                    if len(partitions[centroid])>0:
                        weighted_average = weighted_average + trans_prob*((np.array(partitions[centroid])).mean())
                new_centroids.append(weighted_average)
            return new_centroids
        
        #function used to tie everything together
        def iteration(centroids, count):
            old_centroids = centroids
            partitions = calc_partitions(self.training_set, centroids)
            new_centroids = calc_centroids(partitions, centroids)
            while count < 7:
                count = count + 1
                return iteration(new_centroids, count)
            self.centroids = new_centroids
            self.centroid_map = {index:centroid_val for index, centroid_val in enumerate(self.centroids)}

            return self.centroids

        #return calc_partitions(self.training_set, centroids)
        return iteration(centroids, 0)
            

    def quantize(self, value):
        mse_array = []
        for index, centroid in enumerate(self.centroids):
            mse_array.append(self.__calc_expected_distortion(self.centroids, index, value))
        mse_array = np.array(mse_array)
        return np.argmin(mse_array)
        # quantize a given input value


