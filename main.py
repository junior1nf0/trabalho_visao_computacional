from matplotlib.pyplot import figure
from skimage import img_as_float, img_as_ubyte
from PIL import Image, ImageFont, ImageDraw
from deteccao_classe import Yolo4
from retificacaoImagem import *


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

            imagemResultado, hom = retificarImagem(img_as_ubyte(image.copy()), 4)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Retificacao da imagem')
            ax1.imshow(image)
            ax2.imshow(imagemResultado)
            plt.show()

            jog_fla=[out_boxes[i] for (i, t) in enumerate(out_classes) if t == 0]
            pont_recup=[recupera_base_jogador(jf) for jf in jog_fla]

            pont_retif = [propagarPonto(hom,ponto) for ponto in pont_recup]

            plt.imshow(image)
            for ponto in pont_recup:
                plt.plot(ponto[0], ponto[1], 'r*')
            plt.show()


            plt.imshow(imagemResultado);
            for novo_ponto in pont_retif:
                plt.plot(novo_ponto[0], novo_ponto[1], 'b*')
            plt.show()
    yolo4_model.close_session()

