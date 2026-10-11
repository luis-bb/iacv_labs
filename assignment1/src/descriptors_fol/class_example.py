import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


IMG = "./data/images.jpeg"

imagem = cv2.imread(IMG)


def SIFT_matching():
    smatch = cv2.SIFT_CREATE()
    kp1, des1 = smatch.detectAndCompute()
    
    
#se for descritor binario e usar distanicia de hamilton


def extract_kp_fast():
    fast = cv2.FastFeatureDetector_create(threshold = 150)
    keypoints = fast.detect(imagem, None)
    imagem_resultado = cv2.drawKeypoints(imagem, keypoints, None, color=(0, 255, 0))
    cv2.imshow("Fast Keypoints", imagem_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Found Keypoints: ", {len(keypoints)})
    
extract_kp_fast()
