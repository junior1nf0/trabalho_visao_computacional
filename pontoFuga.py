import logging as log

import numpy as np

from bordas import identificarBordas
from bordas import linhasDeBorda


def calcularVotacao(bordas, modelo, limiteInlier=5):
    """
    dado um modelo de ponto de fuga, calcula por meio de votacao se as bordas sao as mesmas dentro de um limite
    de angulo da linha que passa entre a borda identificada com o ponto de fuga

    """
    vp = modelo[:2] / modelo[2]

    locais, direcoes, intensidades = bordas

    direcoesVP = locais - vp
    somaPontos = np.sum(direcoesVP * direcoes, axis=1)

    prodAbs = np.linalg.norm(direcoes, axis=1) * np.linalg.norm(direcoesVP, axis=1)
    prodAbs[prodAbs == 0] = 1e-5

    thetaCos = somaPontos / prodAbs
    theta = np.arccos(np.abs(thetaCos))

    thetaLim = limiteInlier * np.pi / 180
    return (theta < thetaLim) * intensidades

def identificarPontoFuga(bordas, qtdIteracoesRansac=2000, limiteInlier=5):
    """
    identifica ponto de fuga utilizando RANSAC

    """
    locais, direcoes, intensidades = bordas
    linhas = linhasDeBorda(bordas)

    qtdPontos = intensidades.size

    indicesOrdenados = np.argsort(-intensidades)
    i1 = indicesOrdenados[:qtdPontos // 5]
    i2 = indicesOrdenados[:qtdPontos // 2]

    melhorModelo = None
    melhoresVotos = np.zeros(qtdPontos)

    for iteracaoRansac in range(qtdIteracoesRansac):
        ind1 = np.random.choice(i1)
        ind2 = np.random.choice(i2)

        l1 = linhas[ind1]
        l2 = linhas[ind2]

        modeloAtual = np.cross(l1, l2)

        if np.sum(modeloAtual**2) < 1 or modeloAtual[2] == 0:
            # reject degenerate candidates
            continue

        votosAtuais = calcularVotacao(
            bordas, modeloAtual, limiteInlier)

        if votosAtuais.sum() > melhoresVotos.sum():
            melhorModelo = modeloAtual
            melhoresVotos = votosAtuais
            log.info("O melhor modelo tem {} votos na iteração {}".format(
                votosAtuais.sum(), iteracaoRansac))

    return melhorModelo


def identificarPontoFugaOrtogonal(bordas, comprimentoFocal, qtdIteracoesRansac=2000, limiteInlier=5):
    """ Estima ponto de fuga usando a terceira linha Ransac.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    focal_length: float
        Focal length of the camera used.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    vp1: ndarry of shape (3,)
        Estimated model for first vanishing point.
    vp2: ndarry of shape (3,)
        Estimated model for second vanishing point, which is orthogonal to
        first vanishing point.
    Reference
    ---------
    Bazin, Jean-Charles, and Marc Pollefeys. "3-line RANSAC for orthogonal
    vanishing point detection." 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems. IEEE, 2012.
    """
    locais, direcoes, intensidades = bordas
    linhas = identificarBordas(bordas)

    qtdPontos = intensidades.size

    indicesOrdenados = np.argsort(-intensidades)
    i1 = indicesOrdenados[:qtdPontos // 5]
    i2 = indicesOrdenados[:qtdPontos // 5]
    i3 = indicesOrdenados[:qtdPontos // 2]

    melhorModelo = (None, None)
    melhoresVotos = 0

    for ransacIteracoes in range(qtdIteracoesRansac):
        ind1 = np.random.choice(i1)
        ind2 = np.random.choice(i2)
        ind3 = np.random.choice(i3)

        l1 = linhas[ind1]
        l2 = linhas[ind2]
        l3 = linhas[ind3]

        vp1 = np.cross(l1, l2)
        # linha de fuga do ponto de fuga 1
        h = np.dot(vp1, [1 / comprimentoFocal**2, 1 / comprimentoFocal**2, 1])
        vp2 = np.cross(h, l3)

        if np.sum(vp1**2) < 1 or vp1[2] == 0:
            # rejeita candidatos invalidos
            continue

        if np.sum(vp2**2) < 1 or vp2[2] == 0:
            # rejeita candidatos invalidos
            continue

        vp1_votes = calcularVotacao(bordas, vp1, limiteInlier)
        vp2_votes = calcularVotacao(bordas, vp2, limiteInlier)
        votosAtuais = (vp1_votes > 0).sum() + (vp2_votes > 0).sum()

        if votosAtuais > melhoresVotos:
            melhorModelo = (vp1, vp2)
            melhoresVotos = votosAtuais
            log.info("O melhor modelo tem {} votos na iteração{}".format(
                votosAtuais, ransacIteracoes))

    return melhorModelo
