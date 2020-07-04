from skimage import feature, color, transform, io
from skimage import exposure
import numpy as np
import logging
from edgelets import compute_edgelets, edgelet_lines
from vanishing_point import compute_votes, ransac_vanishing_point, ransac_3_line
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage import img_as_float, img_as_ubyte


from retification_image import *
if __name__ == '__main__':

    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    image_name= "https://i.pinimg.com/originals/4e/02/91/4e02918ad60924448e6b7811b450ed7f.jpg"
    #image_name = "jogo_mp4_0.jpg"
    image = io.imread(image_name)
    plt.imshow(image)
    plt.show()
    # plt.plot(100,100,'r*')
    #print("Rectifying {}".format(image_name))
    save_name = 'result' + '_warped.jpg'
    output_img, hom = rectify_image(image_name, 4, algorithm='independent')
    #print(save_name)

    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')


    plt.imshow(img_as_ubyte(output_img))
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    ax1.imshow(image)
    ax2.imshow(output_img)
    # plt.plot(0,0,'r*')
    plt.show()
    # img_as_ubyte(output_img)