from cosq_class import CoSQ
from image_quantizer import ImageQuantizer
import numpy as np
import cv2
import matplotlib.pyplot as plt


obj = CoSQ(0.05, 1)
obj.training_set([1,2,3,4,5,6,7,8,9,10])
obj.fit()


bit_al_mat = np.matrix([[6,4,3,1,0,0,0,0],
                        [3,2,2,0,0,0,0,0],
                        [1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        ])

obj2 = ImageQuantizer(bit_al_mat, 0)
obj2.import_training_set('../data/org/')
image = cv2.imread('../data/org/1.pgm')
obj2.train()
compressed_img = obj2.compress_image(image)
plot = plt.imsave('compressed_image.png',compressed_img, cmap='gray')

