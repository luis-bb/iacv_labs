import cv2 as cv
import numpy as np
import sys

# 1. Tente carregar uma imagem que REALMENTE exista na sua pasta assignment1
# Mude 'o_nome_da_sua_imagem.jpg' para o ficheiro real (ex: 'lena.png', 'scene.jpg')
nome_ficheiro = 'o_nome_da_sua_imagem.jpg' 
img = cv.imread(nome_ficheiro, cv.IMREAD_GRAYSCALE)

# PROTEÇÃO: Se a imagem não for encontrada, o script para aqui em vez de estourar no BRISK
if img is None:
    print(f"❌ ERRO: Não consegui abrir '{nome_ficheiro}'.")
    print("Verifique se o ficheiro está na pasta ~/Desktop/iacv_labs/assignment1 e se o nome está correto.")
    sys.exit()

# 2. Configurar o BRISK (Já está a funcionar!)
brisk = cv.BRISK_create(thresh=30, octaves=3, patternScale=1.0)

# 3. Detetar e computar
keypoints, descriptors = brisk.detectAndCompute(img, None)

print(f"✅ SUCESSO! Pontos-chave detetados: {len(keypoints)}")

