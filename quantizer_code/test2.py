from cosq_class import CoSQ
from image_quantizer import ImageQuantizer
from channel_class import Channel
import numpy as np
import cv2
import matplotlib.pyplot as plt

bit_al_mat = np.matrix([[6,5,3,1,0,0,0,0],
                        [3,2,2,0,0,0,0,0],
                        [1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        ])

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

img_quant = ImageQuantizer(bit_al_mat=bit_al_mat, epsilon=0.005)
#img_quant.import_training_set('../data/training_photos/')
#img_quant.train()
#img_quant.save_model('./model_24_new', '')
img_quant.load_model('./quantizer_code/model_76_hi_res')

epsilon = 0.005
#obj2 = ImageQuantizer(bit_al_mat, epsilon)
#obj2.import_training_set('../data/org2/')
image = cv2.imread('../data/hires_photos/mak-IqOCrPo2zf4-unsplash.jpg')
#obj2.train()
#obj2.save_model('./model_01', ["#Epsilon = {}".format(epsilon)])
#obj2.load_model('./model_01')
#obj2.quantizer_array[0][0].set_centroids([1,2])


#print(obj2.quantizer_array)
compressed_img = img_quant.compress_image(image)
print(compressed_img)
channel = Channel(0.005, bit_al_mat)
received_img = channel.send_image(compressed_img)
uncompressed_img = img_quant.reconstruct_image(received_img)
plot = plt.imsave('image_city_24b_e005.png',uncompressed_img, cmap='gray')
#plot = plt.imsave('image_city_24.png',cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')
print(img_quant.calc_distortion('../data/hires_photos/mak-IqOCrPo2zf4-unsplash.jpg', 'image_city_24b_e005.png'))