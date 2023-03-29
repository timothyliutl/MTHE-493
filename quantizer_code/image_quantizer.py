# creating a class to take in an image training set and compress given images
import numpy as np
import random
import os
import math
import cv2
from cosq_class import CoSQ

class ImageQuantizer:
    def __init__(self, bit_al_mat, epsilon):
        self.bit_allocation_matrix = bit_al_mat
        self.training_set = []
        self.epsilon = epsilon
        self.trained = False
        self.gaussian = False
        self.mean_mat = np.matrix(np.zeros(shape=(8,8)))
        self.std_mat = np.matrix(np.zeros(shape=(8,8)))

        self.quantizer_array = []
        self.quantizer_precomp_arr = np.zeros(shape=(8,8,(2000+2000)*10), dtype=int)

        for i in range(self.bit_allocation_matrix.shape[0]):
            for j in range(self.bit_allocation_matrix.shape[1]):
                if self.bit_allocation_matrix[i,j]!=0:
                    self.quantizer_array.append([CoSQ(epsilon=self.epsilon, bits=bit_al_mat[i,j]), (i,j)])

    #hidden helper functions
        

    def __import_image(self, file_path, image_file_path):
        print(image_file_path + file_path)
        im = cv2.imread(image_file_path + file_path)
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    def __c_funct(self,i):
        if i==0:
            return 1/(2**(1/2))
        else:
            return 1

    def __dct_matrix(self):
        result_matrix = np.zeros(shape=(8,8))
        sum = 0
        for i in range(8):
            for j in range(8):
                if i ==0:
                    result_matrix[i,j] = np.sqrt(1/8)
                else:
                    result_matrix[i,j] = np.sqrt(2/8)*math.cos(((2*j+1)*i*math.pi)/16)
        return result_matrix
    
    def __dct_funct(self,block):
        return np.matrix(self.__dct_matrix()) * np.matrix(block) * np.matrix(self.__dct_matrix().transpose())

    def __idct_funct(self, block):
        return np.matrix(self.__dct_matrix().transpose()) * np.matrix(block) * np.matrix(self.__dct_matrix())

    # randomly sample num_samples blocks from an image
    def __sample_blocks(self, image, num_samples):
        height_rounded = int(image.shape[0]/8)
        width_rounded = int(image.shape[1]/8)
        used_blocks = np.zeros(shape=(height_rounded, width_rounded))
        blocks = []
        x = 0
        y = 0
        for i in range(num_samples):
            while True:
                x = random.randint(0, height_rounded-1) 
                y = random.randint(0, width_rounded-1)
                # Check if already used
                if used_blocks[x, y] == 0:
                    used_blocks[x, y] = 1
                    break
            blocks.append(self.__dct_funct(image[x*8:(x+1)*8, y*8:(y+1)*8]))
        return blocks

    def __blockify(self, image):
        len_rounded = int(image.shape[0]/8)
        width_rounded = int(image.shape[1]/8)
        return_array = np.zeros(shape=(len_rounded*8, width_rounded*8))

        for i in range(len_rounded):
            for j in range(width_rounded):
                image_block = np.array(image[i*8:(i+1)*8, j*8:(j+1)*8], dtype=float) #decide whether to subtract 128
                return_array[i*8:(i+1)*8, j*8:(j+1)*8] = self.__dct_funct(image_block)
        return return_array
    
    def __generate_training(self,dct_block):
        return_arr = []
        len_rounded = int(dct_block.shape[0]/8)
        width_rounded = int(dct_block.shape[1]/8)
        for i in range(len_rounded):
            for j in range(width_rounded):
                return_arr.append(dct_block[i*8:(i+1)*8, j*8:(j+1)*8])
        return return_arr
    
    def __unquantize_block(self, block):
        for element in self.quantizer_array:
            location = element[1]
            block[location] = element[0].centroid_map[int(block[location])]
        return block
    

    def reconstruct_image(self, quantized_block):
        len_rounded = int(quantized_block.shape[0]/8)
        width_rounded = int(quantized_block.shape[1]/8)
        recon_image = np.zeros(shape=(len_rounded*8, width_rounded*8))
        for i in range(len_rounded):
            for j in range(width_rounded):
                centroid_block = self.__unquantize_block(quantized_block[8*i:8*(i+1), 8*j:8*(j+1)])
                recon_image[8*i:8*(i+1), 8*j:8*(j+1)] = self.__idct_funct(centroid_block)
        return recon_image

    # functions used for compressing an image set
    
    def import_training_set(self, image_path, random = False):
        self.training_set = []
        files = [f for f in os.listdir(image_path)]
        #files = files[:20]
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if not random:
            count = 0
            for image_name in files:
                #print(count)
                count = count + 1
                blocks = self.__blockify(self.__import_image(image_name, image_path))
                self.training_set = self.training_set + self.__generate_training(blocks)
        else:
            for image_name in files:
                self.training_set = self.training_set + self.__sample_blocks(self.__import_image(image_name, image_path), 100)
        print(len(self.training_set))

    def train(self):
        self.trained = True
        count = 0
        for element in self.quantizer_array:
            print(count)
            count = count + 1
            bit_location = element[1]
            element[0].training_set(np.array(self.training_set)[:,bit_location[0], bit_location[1]])
            element[0].c_fit()
    
    def gaussian_train(self):

        for i in range(8):
            for j in range(8):
                self.mean_mat[i,j] = np.mean(np.array(self.training_set)[:,i,j])
                self.std_mat[i,j] = np.std(np.array(self.training_set)[:,i,j])

        self.trained = True
        count = 0
        for element in self.quantizer_array:
            print(count)
            count = count + 1
            bit_location = element[1]
            
            element[0].training_set(np.array(np.random.normal(self.mean_mat[bit_location], self.std_mat[bit_location],5000)))
            element[0].c_fit()


    def save_model(self, file_path, comments):
        # Save model (bit allocation matrix, centroids etc.) to flat .txt file
        # comments is a set of strings starting with '#' and ending in '\n' (newline) 
        #if not self.trained:
        #    raise Exception('Model not trained')

        f = open(file_path, "w")

        # Add comments to file
        if comments != None:
            for comment in comments:
                f.write("{}\n".format(comment))
        
        # Write bit allocation matrix to file
        # Dimensions
        f.write("({},{})\n".format(self.bit_allocation_matrix.shape[0], self.bit_allocation_matrix.shape[1]))
        for i in range(self.bit_allocation_matrix.shape[0]):
            for j in range(self.bit_allocation_matrix.shape[1]):
                f.write("{} ".format(self.bit_allocation_matrix[i,j]))
            f.write("\n")
            
        # Write centroid positions
        for element in self.quantizer_array:
            f.write("{}\n".format(element[1]))
            f.write("[")
            for centroid in element[0].centroids:
                f.write("{} ".format(centroid))
            f.write("]\n")

        f.close()
        

    # Load model from file
    def load_model(self, file_path):
        f = open(file_path, "r")
        line = f.readline()
        # skip through comments
        while line[0] == '#':
            line = f.readline()

        # Get bit allocation matrix 
        dimension = [int(x) for x in line.strip('[()]\n').split(',')]
        bit_al_mat = np.zeros([dimension[0], dimension[1]], dtype=int) 
        for i in range(dimension[0]):
            row = [int(x) for x in f.readline().strip('[()]\n ').split(' ')]
            bit_al_mat[i] = np.array(row)
        print(bit_al_mat)
        self.bit_allocation_matrix = bit_al_mat

        # Get centroids
        while True:
            line = f.readline()
            # EOF
            if len(line) == 0:
                break
            location = tuple(int(x) for x in line.strip('[()]\n ').split(','))
            centroids = [float(x) for x in f.readline().strip('[()]\n ').split()] 
            for element in self.quantizer_array:
                if element[1] == location and len(centroids)!=0:
                    element[0].set_centroids(centroids) 
        self.trained = True

    def compute_encoder_mapping(self, file_path):
        # pre-compute partitions for faster compression
        for element in self.quantizer_array:
            print("Computing encoder map for quantizer: {}".format(element[1]))
            self.quantizer_precomp_arr[element[1]] = element[0].compute_quantizer_map()
        np.save(file_path, self.quantizer_precomp_arr)

    def load_encoder_mapping(self, file_path):
        self.quantizer_precomp_arr = np.load(file_path)
        for element in self.quantizer_array:
            element[0].quantizer_map = self.quantizer_precomp_arr[element[1]]
        print("Precomputed encoder map loaded successfully!")


    def compress_image(self, image):
        if not self.trained:
            raise Exception('uwu i made a fucky: shit aint trained')
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_dct_blocks = self.__blockify(image)
        
        return_block_array = []
        length = int(image.shape[0]/8)
        width = int(image.shape[1]/8)
        quantized_output = np.zeros(shape=(length*8, width*8), dtype=int)
        count = 0

        max_val = -5000
        min_val = 0
        for i in range(length):
            for j in range(width):
                count = count + 1
                #print('compressing block ',count, ' out of ', length*width )
                block = image_dct_blocks[i*8:(i+1)*8, j*8:(j+1)*8]
                quantized_block = np.zeros(shape=(8,8))
                for element in self.quantizer_array:
                    location = element[1]
                    centroid_locations = element[0].centroids
                    pixel_val = block[location]
                    quantized_val = element[0].quantize(pixel_val)
                    #quantized_val = element[0].quantize_optimized(pixel_val)
                    quantized_block[location] = quantized_val
                quantized_output[i*8: (i+1)*8, j*8: (j+1)*8] = quantized_block
        #return self.reconstruct_image(quantized_output)
        print("max val: {}, min val = {}".format(max_val, min_val))
        return quantized_output

    # tim
    # finish compress [done]
    # save centroid positions
    # create c functions to optimize fit function on cosq_class [done]

    def __path_to_matrix(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        height_rounded = int(image.shape[0]/8)
        width_rounded = int(image.shape[1]/8)
        trimmed_image = image[0:height_rounded*8, 0:width_rounded*8]
        return trimmed_image
        
    def calc_distortion(self, sent_image_path, received_image_path):
        sent_image = self.__path_to_matrix(sent_image_path) 
        received_image = self.__path_to_matrix(received_image_path) 
        if sent_image.shape[0] != received_image.shape[0] or sent_image.shape[1] != received_image.shape[1]:
            raise Exception('Images have mismatched size')
        distortion = np.float32(0)
        for i in range(sent_image.shape[0]):
            for j in range(sent_image.shape[1]):
                distortion = distortion + (float(sent_image[i][j]) - float(received_image[i][j]))**2
        return distortion/(sent_image.shape[0] * sent_image.shape[1])

    # mitch
    # method to calculate distortion between compressed and original image
    # trim all the images

    # look into gaussian and laplacian distribution + latex
    # if we have time: port over the quantizer training in C
    # hardware + optimization 
    # video encoding
    
