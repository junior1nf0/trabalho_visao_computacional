import matplotlib.pyplot as plt
import numpy as np
from skimage import transform, io

from bordas import identificarBordas, linhasDeBorda
from pontoFuga import calcularVotacao, identificarPontoFuga


def reestimarModelo(modelo, bordas, reestimativaLimite=5):
    """Reestime o ponto de fuga usando inliers e mínimos quadrados.
     Todas as edgelets que estão dentro de um limite são usadas para reestimar o modelo
    ----------
    model: ponto de fuga
    """
    locais, direcoes, intensidades = bordas

    inliers = calcularVotacao(bordas, modelo, reestimativaLimite) > 0
    locais = locais[inliers]
    direcoes = direcoes[inliers]
    intensidades = intensidades[inliers]

    linhas = linhasDeBorda((locais, direcoes, intensidades))

    a = linhas[:, :2]
    b = -linhas[:, 2]
    modeloEstimado = np.linalg.lstsq(a, b)[0]
    return np.concatenate((modeloEstimado, [1.]))


def removerInliers(modelo, bordas):

    inliers = calcularVotacao(bordas, modelo, 10) > 0
    locais, direcoes, intensidades = bordas
    locais = locais[~inliers]
    direcoes = direcoes[~inliers]
    intensidades = intensidades[~inliers]
    bordas = (locais, direcoes, intensidades)
    return bordas


def calcularMatrizHomografica(imagem, vp1, vp2, cortar=True, fatorCorte=3):
    """
        Calcula a matriz homografica atraves dos pontos de fuga.
    """

    linhaFuga = np.cross(vp1, vp2)

    plt.suptitle('Pontos de fuga')
    plt.imshow(imagem)
    vp1n = vp1 / vp1[2]
    plt.plot(vp1n[0], vp1n[1], 'bo')
    vp2n = vp2 / vp2[2]
    plt.plot(vp2n[0], vp2n[1], 'bo')
    plt.show()

    H = np.eye(3)

    H[2] = linhaFuga / linhaFuga[2]
    H = H / H[2, 2]
    by_only_vanishingLine=False
    if not by_only_vanishingLine:
        # Encontra a direção correspondente para o ponto de fuga

        pontoProd1 = np.dot(H, vp1)
        pontoProd2 = np.dot(H, vp2)
        pontoProd1 = pontoProd1 / np.sqrt(pontoProd1[0] ** 2 + pontoProd1[1] ** 2)
        pontoProd2 = pontoProd2 / np.sqrt(pontoProd2[0] ** 2 + pontoProd2[1] ** 2)

        # direcao do ponto de fuga
        direcoes = np.array([[pontoProd1[0], -pontoProd1[0], pontoProd2[0], -pontoProd2[0]],
                               [pontoProd1[1], -pontoProd1[1], pontoProd2[1], -pontoProd2[1]]])

        thetas = np.arctan2(direcoes[0], direcoes[1])

        # Encontra a direção mais proxima do eixo horizontal
        hInd = np.argmin(np.abs(thetas))

        # Encontra o angulo positivo quando o resto for zero
        if hInd // 2 == 0:
            v_ind = 2 + np.argmax([thetas[2], thetas[3]])
        else:
            v_ind = np.argmax([thetas[2], thetas[3]])

        A1 = np.array([[direcoes[0, v_ind], direcoes[0, hInd], 0],
                       [direcoes[1, v_ind], direcoes[1, hInd], 0],
                       [0, 0, 1]])

        if np.linalg.det(A1) < 0:
            A1[:, 0] = -A1[:, 0]

        A = np.linalg.inv(A1)

        # Aplica translação para que toda imagem apareça
        interMatriz = np.dot(A, H)

        cords = np.dot(interMatriz, [[0, 0, imagem.shape[1], imagem.shape[1]],
                                      [0, imagem.shape[0], 0, imagem.shape[0]],
                                      [1, 1, 1, 1]])
        cords = cords[:2] / cords[2]

        tx = min(0, cords[0].min())
        ty = min(0, cords[1].min())

        max_x = cords[0].max() - tx
        max_y = cords[1].max() - ty

        if cortar:
            # Imagem pode ser grande , aplica um corte como no artigo
            maxOffset = max(imagem.shape) * fatorCorte / 2
            tx = max(tx, -maxOffset)
            ty = max(ty, -maxOffset)

            max_x = min(max_x, -tx + maxOffset)
            max_y = min(max_y, -ty + maxOffset)

        max_x = int(max_x)
        max_y = int(max_y)

        T = np.array([[1, 0, -tx],
                      [0, 1, -ty],
                      [0, 0, 1]])
        matrizHomografica = np.dot(T, interMatriz)

        imagemResultante = transform.warp(imagem, np.linalg.inv(matrizHomografica))
        # ,output_shape=(max_y, max_x))
        return imagemResultante, matrizHomografica
    else:
        imagemResultante = transform.warp(imagem, np.linalg.inv(H))
        # ,output_shape=(max_y, max_x))
        return imagemResultante, H


def exibirBordas(imagem, bordas, exibir=True):

    plt.figure(figsize=(10, 10))
    plt.imshow(imagem)
    locais, direcoes, intensidades = bordas
    for i in range(locais.shape[0]):
        xax = [locais[i, 0] - direcoes[i, 0] * intensidades[i] / 2,
               locais[i, 0] + direcoes[i, 0] * intensidades[i] / 2]
        yax = [locais[i, 1] - direcoes[i, 1] * intensidades[i] / 2,
               locais[i, 1] + direcoes[i, 1] * intensidades[i] / 2]

        plt.plot(xax, yax, 'r-')

    if exibir:
        plt.show()


def exibirModeloCalculado(imagem, modelo, exibir=True):
    import matplotlib.pyplot as plt
    bordas = identificarBordas(imagem)
    locais, direcoes, intensidades = bordas
    inliers = calcularVotacao(bordas, modelo, 10) > 0

    bordas = (locais[inliers], direcoes[inliers], intensidades[inliers])
    locais, direcoes, strengths = bordas
    exibirBordas(imagem, bordas, False)
    vp = modelo / modelo[2]
    plt.plot(vp[0], vp[1], 'bo')
    for i in range(locais.shape[0]):
        xax = [locais[i, 0], vp[0]]
        yax = [locais[i, 1], vp[1]]
        plt.plot(xax, yax, 'b-.')

    if exibir:
        plt.show()


def retificarImagem(imagem, fatorCorte=6, reestimar=False):
    """
    Retifica a imagem usando os pontos de fugas encontrados com o Ransac
    """
    if type(imagem) is not np.ndarray:
        imagem = io.imread(imagem)

    # calculando todas as bordas
    bordas1 = identificarBordas(imagem,sigma=4)

    #exibirBordas(imagem, bordas1) #mostrar arestas

    # Encontra o primeiro ponto de fuga
    vp1 = identificarPontoFuga(bordas1, 5000, limiteInlier=5)
    if reestimar:
        vp1 = reestimarModelo(vp1, bordas1, 5)

    #exibirModeloCalculado(imagem, vp1)

    # Remove as linhas do ponto de fuga
    bordas2 = removerInliers(vp1, bordas1)

    # Encontra o segundo ponto de fuga
    vp2 = identificarPontoFuga(bordas2, 5000, limiteInlier=5)
    if reestimar:
        vp2 = reestimarModelo(vp2, bordas2, 5)

    #exibirModeloCalculado(imagem, vp2)



    imagemResultante, matrizHomografica = calcularMatrizHomografica(imagem, vp1, vp2, cortar=True,
                                                               fatorCorte=fatorCorte)

    return imagemResultante, matrizHomografica



def propagarPonto(mat_hom,ponto):
    """ identifica um ponto de uma imagem em outra a partir da matriz homografica"""

    if type(ponto) is not np.ndarray:
        ponto=np.array(ponto)
    novoPonto=mat_hom@ponto
    novoPonto=novoPonto/novoPonto[2]
    return novoPonto


def recupera_base_jogador(box):
    '''
       TODO calcula centroide do bounding box
    :param box:
    :return:
    '''
    top, left, bottom, right=box
    base_jogador=((left+right)/2,bottom,1)
    return base_jogador


def calcularDistancia(ponto1,ponto2,matriz):
    '''
    TODO calcula distancia entre dois pontos (euclidiana)
    :param ponto1: em  dimensao n
    :param ponto2: em  dimensao n
    :return: retorna a distancia euclinada e por eixo
    '''
    ret_pon1=propagarPonto(matriz,ponto1)
    ret_pon2 = propagarPonto(matriz, ponto2)
    subt=ret_pon1-ret_pon2
    quadrado=subt*subt
    dist=np.sqrt(np.sum(quadrado))
    dist_por_eixo = np.sqrt(quadrado)
    return dist,dist_por_eixo



if __name__ == '__main__':
    matriz=np.eye(3)
    pos1=np.ones(3)
    pos2 = np.ones(3)+[4,3,0]
    dist=calcularDistancia(pos1,pos2,matriz)
    for i in range(1, len(pos1)):
        print(i)
