# creating a class to take in an image training set and compress given images
import numpy as np
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

        self.quantizer_array = []

        for i in range(self.bit_allocation_matrix.shape[0]):
            for j in range(self.bit_allocation_matrix.shape[1]):
                if self.bit_allocation_matrix[i,j]!=0:
                    self.quantizer_array.append([CoSQ(epsilon=self.epsilon, bits=bit_al_mat[i,j]), (i,j)])

    #hidden helper functions
    def __import_image(self, file_path, image_file_path):
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
    
    def __reconstruct(self, quantized_block):
        len_rounded = int(quantized_block.shape[0]/8)
        width_rounded = int(quantized_block.shape[1]/8)
        recon_image = np.zeros(shape=(len_rounded*8, width_rounded*8))
        for i in range(len_rounded):
            for j in range(width_rounded):
                recon_image[8*i:8*(i+1), 8*j:8*(j+1)] = self.__idct_funct(quantized_block[8*i:8*(i+1), 8*j:8*(j+1)])
        return recon_image

    # functions used for compressing an image set
    
    def import_training_set(self, image_path):
        self.training_set = []
        files = [f for f in os.listdir(image_path)]
        files = files[:5]
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        count = 0
        for image_name in files:
            #print(count)
            count = count + 1
            blocks = self.__blockify(self.__import_image(image_name, image_path))
            self.training_set = self.training_set + self.__generate_training(blocks)
        print(len(self.training_set))

    def train(self):
        self.trained = True
        count = 0
        for element in self.quantizer_array:
            print(count)
            count = count + 1
            bit_location = element[1]
            element[0].training_set(np.array(self.training_set)[:,bit_location[0], bit_location[1]])
            element[0].fit()
            
    def compress_image(self, image):
        if not self.trained:
            raise Exception('uwu i made a fucky: shit aint trained')
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_dct_blocks = self.__blockify(image)
        
        return_block_array = []
        length = int(image.shape[0]/8)
        width = int(image.shape[1]/8)
        quantized_output = np.zeros(shape=(length*8, width*8))

        for i in range(length):
            for j in range(width):
                block = image_dct_blocks[i*8:(i+1)*8, j*8:(j+1)*8]
                quantized_block = np.zeros(shape=(8,8))
                for element in self.quantizer_array:
                    location = element[1]
                    centroid_locations = element[0].centroids
                    pixel_val = block[location]
                    quantized_val = element[0].quantize(pixel_val)
                    quantized_block[location] = quantized_val
                quantized_output[i*8: (i+1)*8, j*8: (j+1)*8] = quantized_block
        return self.__reconstruct(quantized_output)

    # tim
    # finish compress [done]
    # save centroid positions
    # create c functions to optimize fit function on cosq_class

    # mitch
    # method to calculate distortion between compressed and original image
    # trim all the images

    # look into gaussian and laplacian distribution + latex
    # if we have time: port over the quantizer training in C
    # hardware + optimization 
    # video encoding
    