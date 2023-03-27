from cosq_class import CoSQ
from image_quantizer import ImageQuantizer
from channel_class import Channel
import numpy as np
import cv2
import matplotlib.pyplot as plt


#arr = [np.random.rand()*1000 for i in range(1000)]
#
#
#
#obj = CoSQ(0.05, 1)
#obj.training_set([1,2,3,4,5,6,7,8,9,10])
#obj.fit()


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

img_quant = ImageQuantizer(bit_al_mat=bit_al_mat_76, epsilon=0.005)
img_quant.import_training_set('../data/training_photos/')
img_quant.train()
img_quant.save_model('./model_76_hi_res', '')



epsilon = 0.005
image = cv2.imread('../data/hires_photos/mak-IqOCrPo2zf4-unsplash.jpg')
#obj2.train()
#obj2.save_model('./model_01', ["#Epsilon = {}".format(epsilon)])
#obj2.load_model('./model_01')
#obj2.quantizer_array[0][0].set_centroids([1,2])


#print(obj2.quantizer_array)
compressed_img = img_quant.compress_image(image)
print(compressed_img)
channel = Channel(0.01, bit_al_mat)
received_img = channel.send_image(compressed_img)
uncompressed_img = img_quant.reconstruct_image(received_img)
plot = plt.imsave('city_76bit_e005.png',uncompressed_img, cmap='gray')

#keinar photo 24 bits with PSNR of about 13.35

