"""
config.py - Protocolo de Avaliação para Feature Matching (HPatches)
"""

import cv2

# ==========================================
# A1. Configurações Gerais do Protocolo
# ==========================================

# Dataset e Categorias
DATASET_NAME = "HPatches"
CATEGORIES = {
    "illumination": "i_*",  # 57 sequências de iluminação
    "viewpoint": "v_*"     # 59 sequências de viewpoint
}

# Estrutura de pares a avaliar
# Cada sequência tem img 1.ppm e img k.ppm (com k de 2 a 6)
REFERENCE_IMAGE_INDEX = 1
TARGET_IMAGE_INDICES = [2, 3, 4, 5, 6]  # Eixo de dificuldade (k)

# Pré-processamento
PREPROCESS_GRAYSCALE = True
MAX_RESOLUTION_SIDE = 1024  # Lado maior ≤ 1024 px

# Orçamento de Keypoints
# Orçamento fixo para comparação justa vs. parâmetros default
USE_KEYPOINT_BUDGET = True
MAX_KEYPOINTS = 2000

# Parâmetros de Matching Clássico
KNN_K = 2
RATIO_TEST_THRESHOLD = 0.8
ENFORCE_ONE_TO_ONE = True  # Cross-check mútuo ou remoção de duplicados

# Distâncias por tipo de descritor
DESCRIPTOR_METRICS = {
    "float": cv2.NORM_L2,     # Para SIFT, KAZE, etc.
    "binary": cv2.NORM_HAMMING # Para descritores binários (ORB, BRISK, etc.)
}

# Critério de Correção de um Match (Reprojection Error)
CORRECT_MATCH_THRESHOLD_PX = 3.0

# Parâmetros RANSAC (cv2.findHomography)
RANSAC_METHOD = cv2.RANSAC
RANSAC_REPROJ_THRESHOLD = 3.0  # Limiar em pixels
RANSAC_MAX_iters = 2000
RANSAC_CONFIDENCE = 0.995
RANDOM_SEED = 42  # Seed fixa para reprodutibilidade
