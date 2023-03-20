import matplotlib.pyplot as plt
import os
import cv2

# Small script to trim images so that they are always a multiple of 8x8 blocks
def trim_pgm(image_path):
    for file in os.listdir(image_path):
        im = cv2.imread(image_path + file)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        height_rounded = int(img.shape[0]/8)
        width_rounded = int(img.shape[1]/8)

        # slice image
        trimmed_img = img[0:height_rounded*8, 0:width_rounded*8]
        new_file = file.split('.')[0] + '_trimmed.pgm'
        cv2.imwrite(image_path + new_file, trimmed_img)

        # test
        im = cv2.imread(image_path + new_file)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
            print("Error: incorrect trim on " + file)


trim_pgm("./data/org/")
