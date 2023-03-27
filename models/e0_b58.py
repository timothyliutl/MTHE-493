from image_quantizer import ImageQuantizer
from channel_class import Channel
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

epsilon = 0
bit_al_mat = np.matrix([[7,5,4,3,2,1,0,0],
                        [4,4,3,3,2,0,0,0],
                        [3,3,3,2,1,0,0,0],
                        [2,2,2,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        ])

img_quant = ImageQuantizer(bit_al_mat=bit_al_mat, epsilon=epsilon)
img_quant.import_training_set('../data/org2/')
img_quant.train()
img_quant.save_model('./model_files/e0_b58', '')