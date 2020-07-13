from matplotlib.pyplot import figure
from skimage import img_as_float, img_as_ubyte
from PIL import Image, ImageFont, ImageDraw
from deteccao_classe import Yolo4
from retificacaoImagem import *

def calc_dist_rel(list_pont,mat_hom):
    list_metric = []
    for i in range(1, len(pont_recup)):
        print('pontos {0}->{1}'.format(0, i), list_pont[0], list_pont[i])
        dist, dist_eixo = calcularDistancia(list_pont[0], list_pont[i], mat_hom)
        print('distancia {0}->{1}'.format(0, i), dist, dist_eixo)
        if (len(list_metric) == 0):
            list_metric.append([list_pont[0], list_pont[i], dist, 1])
        else:
            dist_p0p1 = list_metric[0][2]
            dist_relat = dist / dist_p0p1
            list_metric.append([list_pont[0], list_pont[i], dist, dist_relat])
    return list_metric


if __name__ == '__main__':

    model_path = 'trained_weights_final.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/times_classes.txt'

    score = 0.5
    iou = 0.5

    model_image_size = (608, 608)

    yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            result,out_boxes, out_scores, out_classes = yolo4_model.detect_image(image.copy(), model_image_size=model_image_size)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Detecção dos jogadores')
            ax1.imshow(image)
            ax2.imshow(result)
            plt.show()

            imagemResultado, hom = retificarImagem(img_as_ubyte(image.copy()), 2)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Retificacao da imagem')
            ax1.imshow(image)
            ax2.imshow(imagemResultado)
            plt.show()

            jog_fla=[out_boxes[i] for (i, t) in enumerate(out_classes) if t == 0]
            pont_recup=[recupera_base_jogador(jf) for jf in jog_fla]

            pont_retif = [propagarPonto(hom,ponto) for ponto in pont_recup]

            plt.imshow(image)
            for i,ponto in enumerate(pont_recup):
                plt.annotate("P "+str(i), xy=(ponto[0], ponto[1]),xytext=(ponto[0], ponto[1]+25),
                             arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))


            #calcular a distancia relativa
            list_met=calc_dist_rel(pont_recup,mat_hom=hom)
            for met in list_met:
                x = [met[0][0],met[1][0]]
                y = [met[0][1],met[1][1]]
                plt.plot(x,y)
                plt.annotate("d {:01.2f} ".format(met[3]), xy=(np.mean(x),np.mean(y)), xytext=(np.mean(x)+25,np.mean(y)+5),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            fig.suptitle('Distância relativa entre os jogadores')
            plt.show()

            '''
            plt.imshow(imagemResultado);
            for i,novo_ponto in enumerate(pont_retif):
                #plt.plot(novo_ponto[0], novo_ponto[1], 'b*')
                print('Novo P {0}->{1}'.format(0, i), pont_retif[0], pont_retif[i])
                plt.annotate("P "+str(i), xy=(novo_ponto[0], novo_ponto[1]),xytext=(novo_ponto[0], novo_ponto[1]+25),
                             arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
            plt.show()
            '''
    yolo4_model.close_session()

