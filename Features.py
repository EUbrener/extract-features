import argparse
import os   
import cv2
import numpy as np

def process_image(path, threshold=160, blur_ksize=(7, 7), save_output=False):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Não foi possível encontrar a imagem: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")
    
    # Se já for grayscale, mantém; caso contrário, converte
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    suave = cv2.GaussianBlur(gray, blur_ksize, 0)

    _, bin = cv2.threshold(suave, threshold, 255, cv2.THRESH_BINARY)   
    _, binI = cv2.threshold(suave, threshold, 255, cv2.THRESH_BINARY_INV)

    # Garante que a imagem mascarada também fique em escala de cinza
    masked = cv2.bitwise_and(gray, gray, mask=binI)

    # Empilha imagens para visualização
    top = np.hstack([suave, bin])
    bottom = np.hstack([binI, masked])
    resultado = np.vstack([top, bottom])

    if save_output:
        output_path = "resultado.png"
        cv2.imwrite(output_path, resultado)
        print(f"Resultado salvo em: {output_path}")
    else:
        cv2.imshow("Binarização da imagem", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Processa imagem e mostra binarização.")
    parser.add_argument("image", help="Caminho para a imagem")
    parser.add_argument("--threshold", "-t", type=int, default=160, help="Limiar de binarização")
    parser.add_argument("--save", "-s", action="store_true", help="Salvar o resultado da imagem")
    args = parser.parse_args()

    try:
        process_image(args.image, args.threshold, save_output=args.save)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
    
