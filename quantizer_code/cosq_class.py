import numpy as np
from ctypes import *
from sys import platform
if platform == "linux" or platform == "linux2":
    my_functions = CDLL('./cosq_funct.so')
else:
    my_functions = CDLL('cosq_funct.so')


class CoSQ:
    def __init__(self, epsilon, bits):
        self.epsilon = epsilon
        self.bits = bits
        self.centroids = []
        self.centroid_map = {}
        self.quantizer_map = np.zeros(shape=(2000+2000)*10, dtype=int)

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
            my_functions.expected_distortion.argtypes = (POINTER(c_float), c_int, c_int, c_float, c_float, c_int)
            my_functions.expected_distortion.restype = c_float

            c_centroid_array = (c_float * len(self.centroids))(*self.centroids)
            c_centroid_len = len(self.centroids)
            int_index = int(index)

            distortion = my_functions.expected_distortion(c_centroid_array, c_centroid_len, int_index, point, self.epsilon, self.bits)
            #using function made in C instead of this
            #for i in range(len(centroids)):
                #trans_prob = self.__calc_transition_prob(index, i)
                #distortion = distortion + trans_prob*((centroids[i] - point)**2)
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
                mse_array = np.array([])
                for input_index in range(len(centroids)):
                    #calculating expected distortion for a given quantization, input_index
                    mse_array = np.append(mse_array, calc_expected_distortion(centroids, input_index, point))
                    #mse_array.append(calc_expected_distortion(centroids, input_index, point))
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
            while count < 15:
                count = count + 1
                return iteration(new_centroids, count)
            self.centroids = new_centroids
            self.centroid_map = {index:centroid_val for index, centroid_val in enumerate(self.centroids)}

            return self.centroids

        #return calc_partitions(self.training_set, centroids)
        return iteration(centroids, 0)
            

    def quantize(self, value):
        mse_array = []
        #mse_array = np.array([])
        for index, centroid in enumerate(self.centroids):
            #mse_array = np.append(mse_array, self.__calc_expected_distortion(self.centroids, index, value))
            mse_array.append(self.__calc_expected_distortion(self.centroids, index, value))
        mse_array = np.array(mse_array)
        #return self.centroid_map[np.argmin(mse_array)]
        return np.argmin(mse_array)
        # quantize a given input value

    def compute_quantizer_map(self):
        for i, val in enumerate(np.arange(-2000, 2000, 0.1)):
            #if i % 1000 == 0:
            #    print("Precomputing index: {}, val: {}".format(i, val))
            self.quantizer_map[i] = self.quantize(val)
        return self.quantizer_map

    def quantize_optimized(self, value):
        #round to nearest tenth
        rounded_val = round(value,1)
        index = int(10*(rounded_val + 2000))
        try:
            return self.quantizer_map[index]
        except:
            return self.quantize(value)

    # Set centroids from previously saved model
    def set_centroids(self, centroids):
        self.centroids = centroids
        self.centroid_map = {index:centroid_val for index, centroid_val in enumerate(self.centroids)}


    def c_fit(self):
        num_centroids = 2**self.bits
        max_val = np.array(self.training_set).max()
        self.centroids = [np.random.randint(0, max_val) for i in range(num_centroids)]
        self.centroids.sort()
        len_training_set = len(self.training_set)
        print(len(self.centroids), len(self.training_set))
        my_functions.iteration.argtypes = (POINTER(c_float), c_int, POINTER(c_float), c_int, c_int, c_float, c_int, c_int)
        my_functions.iteration.restype = POINTER(c_float)

        #grab 10000 random training samples and loop mini batch gradient descent
        #for i in range(5 * int(len_training_set/5000)):
        #print('iteration ', i)

        subset = self.training_set

        c_centroid_array = (c_float * len(self.centroids))(*self.centroids)
        c_centroid_len = len(self.centroids)
        c_training_set = (c_float * len(subset))(*subset)
        c_training_len = len(subset)
        c_print_distortion = 1 

        return_iter = my_functions.iteration(c_centroid_array, c_centroid_len, c_training_set, c_training_len, 0, self.epsilon, self.bits, c_print_distortion)
        self.centroids = np.fromiter(return_iter, c_float, c_centroid_len)
        self.centroid_map = {index:centroid_val for index, centroid_val in enumerate(self.centroids)}
        print(self.centroids)
