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




epsilon = 0.01
obj2 = ImageQuantizer(bit_al_mat, epsilon)
obj2.import_training_set('../data/org2/')
image = cv2.imread('../data/org2/16.pgm')
obj2.train()
obj2.save_model('./model_01', ["#Epsilon = {}".format(epsilon)])
#obj2.load_model('./model_01')
#obj2.quantizer_array[0][0].set_centroids([1,2])


print(obj2.quantizer_array)
compressed_img = obj2.compress_image(image)
print(compressed_img)
channel = Channel(0.01, bit_al_mat)
received_img = channel.send_image(compressed_img)
uncompressed_img = obj2.reconstruct_image(received_img)
plot = plt.imsave('uncompressed_image_01.png',uncompressed_img, cmap='gray')

