from image_quantizer import ImageQuantizer
import math
from channel_class import Channel
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

epsilon = 0.05
bit_al_mat = np.matrix([[8,8,0,0,0,0,0,0],
                        [8,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        ])

img_quant = ImageQuantizer(bit_al_mat=bit_al_mat, epsilon=epsilon)
img_quant.import_training_set('../data/org2/')
img_quant.train()
img_quant.save_model('./model_files/e005_b24', '')

#img_quant.load_model('./model_files/e005_b24')
img_quant.compute_encoder_mapping("./model_files/e005_b24")
img_quant.load_encoder_mapping("./model_files/e005_b24.npy")

image = cv2.imread('./original_fox.png')
compressed_img = img_quant.compress_image(image)
channel = Channel(epsilon, bit_al_mat)
received_img = channel.send_image(compressed_img)
uncompressed_img = img_quant.reconstruct_image(received_img)
plt.imsave('./images/received_fox_24bit_e005.png', uncompressed_img, cmap='gray')

psnr = 10*math.log(255**2 / img_quant.calc_distortion('./images/received_fox_24bit_e005.png', './original_fox.png'), 10)
print("24 bit PSNR for epsilon {}:  {}".format(epsilon, psnr))

