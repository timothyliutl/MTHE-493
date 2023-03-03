from cosq_class import CoSQ
from image_quantizer import ImageQuantizer
import numpy as np
import cv2

obj = CoSQ(0.05, 1)
obj.training_set([np.random.randint(0,255) for i in range(100)])
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

obj2 = ImageQuantizer(bit_al_mat, 0.05)
obj2.import_training_set('../data/org/')
obj2.train()
image = cv2.imread('data/org/1.pgm')
obj2.compress_image()


