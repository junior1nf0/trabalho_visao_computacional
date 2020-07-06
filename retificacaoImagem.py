import matplotlib.pyplot as plt
import numpy as np
from skimage import transform, io

from bordas import identificarBordas, linhasDeBorda
from pontoFuga import calcularVotacao, identificarPontoFuga, identificarPontoFugaOrtogonal


def reestimarModelo(modelo, bordas, reestimativaLimite=5):
    """Reestimate vanishing point using inliers and least squares.
    All the edgelets which are within a threshold are used to reestimate model
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        All edgelets from which inliers will be computed.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    restimated_model: ndarry of shape (3,)
        Reestimated model for vanishing point in homogenous coordinates.
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
    """Compute homography from vanishing points and warp the image.
    It is assumed that vp1 and vp2 correspond to horizontal and vertical
    directions, although the order is not assumed.
    Firstly, projective transform is computed to make the vanishing points go
    to infinty so that we have a fronto parellel view. Then,Computes affine
    transfom  to make axes corresponding to vanishing points orthogonal.
    Finally, Image is translated so that the image is not missed. Note that
    this image can be very large. `clip` is provided to deal with this.
    Parameters
    ----------
    image: ndarray
        Image which has to be wrapped.
    vp1: ndarray of shape (3, )
        First vanishing point in homogenous coordinate system.
    vp2: ndarray of shape (3, )
        Second vanishing point in homogenous coordinate system.
    clip: bool, optional
        If True, image is clipped to clip_factor.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    Returns
    -------
    warped_img: ndarray
        Image warped using homography as described above.
    """
    # Find Projective Transform
    linhaFuga = np.cross(vp1, vp2)

    H = np.eye(3)

    H[2] = linhaFuga / linhaFuga[2]
    H = H / H[2, 2]

    # Find directions corresponding to vanishing points

    pontoProd1 = np.dot(H, vp1)
    pontoProd2 = np.dot(H, vp2)
    pontoProd1 = pontoProd1 / np.sqrt(pontoProd1[0] ** 2 + pontoProd1[1] ** 2)
    pontoProd2 = pontoProd2 / np.sqrt(pontoProd2[0] ** 2 + pontoProd2[1] ** 2)

    directions = np.array([[pontoProd1[0], -pontoProd1[0], pontoProd2[0], -pontoProd2[0]],
                           [pontoProd1[1], -pontoProd1[1], pontoProd2[1], -pontoProd2[1]]])

    thetas = np.arctan2(directions[0], directions[1])

    # Find direction closest to horizontal axis
    hInd = np.argmin(np.abs(thetas))

    # Find positve angle among the rest for the vertical axis
    if hInd // 2 == 0:
        v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
        v_ind = np.argmax([thetas[2], thetas[3]])

    A1 = np.array([[directions[0, v_ind], directions[0, hInd], 0],
                   [directions[1, v_ind], directions[1, hInd], 0],
                   [0, 0, 1]])
    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(A1) < 0:
        A1[:, 0] = -A1[:, 0]

    A = np.linalg.inv(A1)

    # Translate so that whole of the image is covered
    interMatriz = np.dot(A, H)
    #print(A)
    #print(inter_matrix)
    cords = np.dot(interMatriz, [[0, 0, imagem.shape[1], imagem.shape[1]],
                                  [0, imagem.shape[0], 0, imagem.shape[0]],
                                  [1, 1, 1, 1]])
    cords = cords[:2] / cords[2]

    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())

    max_x = cords[0].max() - tx
    max_y = cords[1].max() - ty

    if cortar:
        # These might be too large. Clip them.
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
    #print(T)
    matrizHomografica = np.dot(T, interMatriz)

    imagemResultante = transform.warp(imagem, np.linalg.inv(matrizHomografica))
    # ,output_shape=(max_y, max_x))
    return imagemResultante, matrizHomografica


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
    """Helper function to visualize computed model."""
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


def retificarImagem(imagem, fatorCorte=6, algoritmo='independente', reestimar=False):
    """Rectified image with vanishing point computed using ransac.
    Parameters
    ----------
    image: ndarray
        Image which has to be rectified.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    algorithm: one of {'3-line', 'independent'}
        independent ransac algorithm finds the orthogonal vanishing points by
        applying ransac twice.
        3-line algorithm finds the orthogonal vanishing points together, but
        assumes knowledge of focal length.
    reestimate: bool
        If ransac results are to be reestimated using least squares with
        inlers. Turn this off if getting bad results.
    Returns
    -------
    warped_img: ndarray
        Rectified image.
    """
    if type(imagem) is not np.ndarray:
        imagem = io.imread(imagem)

    # Compute all edgelets.
    bordas1 = identificarBordas(imagem,sigma=4)

    exibirBordas(imagem, bordas1) #mostrar arestas

    if algoritmo == 'independente':
        # Find first vanishing point
        vp1 = identificarPontoFuga(bordas1, 5000, limiteInlier=5)
        if reestimar:
            vp1 = reestimarModelo(vp1, bordas1, 5)

        exibirModeloCalculado(imagem, vp1)  # Visualize the vanishing point model

        # Remove inlier to remove dominating direction.
        bordas2 = removerInliers(vp1, bordas1)

        # Find second vanishing point
        vp2 = identificarPontoFuga(bordas2, 5000, limiteInlier=5)
        if reestimar:
            vp2 = reestimarModelo(vp2, bordas2, 5)
            exibirModeloCalculado(imagem, vp2)  # Visualize the vanishing point model

    elif algoritmo == '3-linha':
       comprimentoFocal = None
       vp1, vp2 = identificarPontoFugaOrtogonal(bordas1, comprimentoFocal, qtdIteracoesRansac=3000, limiteInlier=5)

    else:
        raise KeyError(
            "O parametro algoritmo deve ser {'3-linha', 'independente'}")

    # Compute the homography and warp
    imagemResultante, matrizHomografica = calcularMatrizHomografica(imagem, vp1, vp2, cortar=False,
                                                               fatorCorte=fatorCorte)

    return imagemResultante, matrizHomografica



def propagarPonto(mat_hom,ponto):
    """ identifica um ponto de uma imagem em outra a partir da matriz homografica"""

    if type(ponto) is not np.ndarray:
        ponto=np.array(ponto)

    novoPonto=mat_hom@ponto
    novoPonto=novoPonto/novoPonto[2]
    return novoPonto


def calcularCentroide(ponto1,ponto2):
    '''
       TODO calcula centroide do bounding box
    :param ponto1:
    :param ponto2:
    :return:
    '''
    return (0,0)


def calcularDistancia(ponto1,ponto2):
    '''
    TODO calcula distancia entre dois pontos (euclidiana)
    :param ponto1:
    :param ponto2:
    :return:
    '''

