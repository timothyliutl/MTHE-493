from cosq_class import CoSQ
from image_quantizer import ImageQuantizer
from channel_class import Channel
import numpy as np
import cv2
import matplotlib.pyplot as plt


bit_al_mat_76 = np.array([
    [8,8,5,3,3,2,0,0],
    [6,5,3,3,2,1,0,0],
    [3,3,3,2,2,1,0,0],
    [2,2,2,2,1,0,0,0],
    [1,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
])




img_quant = ImageQuantizer(bit_al_mat=bit_al_mat_76, epsilon=0.005)
img_quant.load_model('./model_76_new')
print(img_quant.calc_distortion('../data/hires_photos/mak-IqOCrPo2zf4-unsplash.jpg', './images/city_76bit_e005.png'))