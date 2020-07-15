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
            continue

        votosAtuais = calcularVotacao(
            bordas, modeloAtual, limiteInlier)

        if votosAtuais.sum() > melhoresVotos.sum():
            melhorModelo = modeloAtual
            melhoresVotos = votosAtuais
            log.info("O melhor modelo tem {} votos na iteração {}".format(
                votosAtuais.sum(), iteracaoRansac))

    return melhorModelo