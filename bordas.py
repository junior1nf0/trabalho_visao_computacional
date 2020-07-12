import numpy as np
from skimage import exposure
from skimage import feature, color, transform


def identificarBordas(imagem, sigma=3):
    """
    Identifica as bordas na imagem
    """

    tempMonoImg = color.rgb2gray(imagem)
    imagemEqualizada = exposure.equalize_hist(tempMonoImg)

    #identifica as bordas com canny
    bordas = feature.canny(imagemEqualizada, sigma)

    #identifica as linhas com hough
    linhas = transform.probabilistic_hough_line(bordas, line_length=3,line_gap=2)

    locais = []
    direcoes = []
    intensidades = []

    #encontra os pontos medios das linhas, suas direcoes e a normal de cada uma
    for p0, p1 in linhas:
        p0, p1 = np.array(p0), np.array(p1)
        locais.append((p0 + p1) / 2)
        direcoes.append(p1 - p0)
        intensidades.append(np.linalg.norm(p1 - p0))

    # converte para vetores
    locais = np.array(locais)
    direcoes = np.array(direcoes)
    intensidades = np.array(intensidades)

    #print('L= {} D= {} S={}'.format(locations, directions,strengths))

    direcoes = np.array(direcoes) / np.linalg.norm(direcoes, axis=1)[:, np.newaxis]

    #print('L= {} D= {} S={}'.format(locations, directions,strengths))
    return (locais, direcoes, intensidades)


def linhasDeBorda(bordas):
    """
    Calcula as linhas para as bordas
    """

    locais, direcoes, _ = bordas
    # print('D {} shape{}'.format(directions,directions.shape))
    normais = np.zeros_like(direcoes)
    normais[:, 0] = direcoes[:, 1]
    normais[:, 1] = -direcoes[:, 0]

    p = -np.sum(locais * normais, axis=1)
    linhas = np.concatenate((normais, p[:, np.newaxis]), axis=1)
    print(linhas.shape)
    return linhas

