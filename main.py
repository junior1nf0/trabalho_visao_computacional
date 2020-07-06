import numpy as np
from skimage import feature, color, transform, io
from skimage import exposure
import logging
from edgelets import identificarBordas, linhasDeBorda
from vanishing_point import calcularVotacao, identificarPontoFuga, identificarPontoFugaOrtogonal
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage import img_as_float, img_as_ubyte


from retificacaoImagem import *
if __name__ == '__main__':

    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    nomeImagem= "https://i.pinimg.com/originals/4e/02/91/4e02918ad60924448e6b7811b450ed7f.jpg"
    #image_name = "jogo_mp4_0.jpg"
    imagem = io.imread(nomeImagem)
    plt.imshow(imagem)
    plt.show()
    # plt.plot(100,100,'r*')
    #print("Rectifying {}".format(image_name))
    nomeImagemResultado = 'resultado' + '_retificada.jpg'
    imagemResultado, hom = retificarImagem(nomeImagem, 4, algorithm='independente')
    #print(save_name)

    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')


    plt.imshow(img_as_ubyte(imagemResultado))
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    ax1.imshow(imagem)
    ax2.imshow(imagemResultado)
    # plt.plot(0,0,'r*')
    plt.show()
    # img_as_ubyte(output_img)

    io.imsave(nomeImagemResultado, imagemResultado)