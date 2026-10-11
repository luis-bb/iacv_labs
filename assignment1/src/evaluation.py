"""
evaluation.py - Implementação das métricas de avaliação para o pipeline de Feature Matching
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def get_descriptor_bytes(descriptor_type):
    """Retorna o tamanho em bytes por descritor conforme especificado em ASE A2 (9)."""
    descriptor_sizes = {
        "SIFT": 128 * 4,       # 512 B
        "KAZE": 64 * 4,        # 256 B
        "ORB": 32,             # 32 B
        "BRISK": 64,           # 64 B
        "FREAK": 64,           # 64 B
        "SuperPoint": 256 * 4  # 1024 B
    }
    return descriptor_sizes.get(descriptor_type.upper(), 64)  # Default para 64 B se desconhecido


def compute_memory_footprint(num_keypoints, descriptor_type):
    """Calcula a memória total ocupada pelos descritores em bytes."""
    bytes_per_desc = get_descriptor_bytes(descriptor_type)
    return num_keypoints * bytes_per_desc


def get_covisible_keypoints(kps1, kps2, H, shape2, shape1, threshold_px=3.0):
    """
    ASE A2 (1): Projeta os kps da imagem 1 com H para a imagem 2 e mantém 
    só os que caem dentro dos limites (e faz o inverso com H⁻¹).
    Retorna os índices ou coordenadas dos keypoints co-visíveis.
    """
    h2, w2 = shape2[:2]
    h1, w1 = shape1[:2]

    # Converter keypoints para matriz (N, 2)
    pts1 = np.array([kp.pt for kp in kps1]) if len(kps1) > 0 else np.zeros((0, 2))
    pts2 = np.array([kp.pt for kp in kps2]) if len(kps2) > 0 else np.zeros((0, 2))

    if len(pts1) == 0 or len(pts2) == 0:
        return [], []

    # Homografia inversa
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return [], []

    # Projetar pts1 para img2
    ones1 = np.ones((pts1.shape[0], 1))
    pts1_h = np.hstack([pts1, ones1])
    proj1 = (H @ pts1_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:]

    # Filtrar válidos dentro dos limites de img2
    mask1 = (proj1[:, 0] >= 0) & (proj1[:, 0] < w2) & (proj1[:, 1] >= 0) & (proj1[:, 1] < h2)
    
    # Projetar pts2 para img1
    ones2 = np.ones((pts2.shape[0], 1))
    pts2_h = np.hstack([pts2, ones2])
    proj2 = (H_inv @ pts2_h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:]

    # Filtrar válidos dentro dos limites de img1
    mask2 = (proj2[:, 0] >= 0) & (proj2[:, 0] < w1) & (proj2[:, 1] >= 0) & (proj2[:, 1] < h1)

    return np.where(mask1)[0], np.where(mask2)[0]


def compute_repeatability(kps1, kps2, H, shape2, shape1, threshold_px=3.0):
    """
    ASE A2 (2): Repetibilidade com atribuição one-to-one (via linear_sum_assignment).
    """
    vis1_idx, vis2_idx = get_covisible_keypoints(kps1, kps2, H, shape2, shape1)
    n1_vis, n2_vis = len(vis1_idx), len(vis2_idx)
    
    if n1_vis == 0 or n2_vis == 0:
        return 0.0, 0, n1_vis, n2_vis

    pts1 = np.array([kps1[i].pt for i in vis1_idx])
    pts2 = np.array([kps2[i].pt for i in vis2_idx])

    # Projetar pts1 para a imagem 2
    ones = np.ones((pts1.shape[0], 1))
    proj1 = (H @ np.hstack([pts1, ones]).T).T
    proj1 = proj1[:, :2] / proj1[:, 2:]

    # Matriz de distâncias euclidianas
    dist_matrix = np.linalg.norm(proj1[:, np.newaxis, :] - pts2[np.newaxis, :, :], axis=2)

    # Atribuição one-to-one (Hungarian Algorithm / linear_sum_assignment)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Contar matches válidos (< threshold_px)
    valid_matches = sum(1 for r, c in zip(row_ind, col_ind) if dist_matrix[r, c] < threshold_px)
    
    denominator = min(n1_vis, n2_vis)
    repeatability = valid_matches / denominator if denominator > 0 else 0.0
    
    return repeatability, valid_matches, n1_vis, n2_vis


def evaluate_matching_performance(putative_matches, matches_gt, n_features, n_correspondences):
    """
    ASE A2 (5 & 6): Calcula PMR, Precision, Matching Score, Recall e executa o sanity check.
    """
    num_putatives = len(putative_matches)
    num_correct = len([m for m in putative_matches if m in matches_gt])  # ou avaliado por limiar de erro

    pmr = num_putatives / n_features if n_features > 0 else 0.0
    precision = num_correct / num_putatives if num_putatives > 0 else 0.0
    matching_score = num_correct / n_features if n_features > 0 else 0.0
    recall = num_correct / n_correspondences if n_correspondences > 0 else 0.0

    # Sanity check obrigatório (ASE A2 - 6)
    ms_pmr_precision = pmr * precision
    assert abs(matching_score - ms_pmr_precision) < 1e-9, f"Sanity check falhou: MS ({matching_score}) != PMR*Precision ({ms_pmr_precision})"
    assert recall <= 1.0 + 1e-9, f"Recall inválido: {recall} > 1.0"

    return {
        "PMR": pmr,
        "Precision": precision,
        "Matching_Score": matching_score,
        "Recall": recall
    }


def evaluate_homography_accuracy(H_estimated, H_gt, img_shape):
    """
    ASE A2 (7 & 8): Erro médio de canto (px) e percentagem de pares com erro ≤ 1, 3 e 5 px.
    Se RANSAC falhar (< 4 inliers / H vazia), retorna erro infinito.
    """
    if H_estimated is None:
        return {"mean_corner_err": float('inf'), "acc_1px": 0.0, "acc_3px": 0.0, "acc_5px": 0.0}

    h, w = img_shape[:2]
    # Cantos da imagem 1
    corners = np.array([
        [0, 0, 1],
        [w - 1, 0, 1],
        [w - 1, h - 1, 1],
        [0, h - 1, 1]
    ]).T

    proj_gt = (H_gt @ corners)
    proj_gt = proj_gt[:2] / proj_gt[2:]

    proj_est = (H_estimated @ corners)
    proj_est = proj_est[:2] / proj_est[2:]

    corner_errors = np.linalg.norm(proj_gt - proj_est, axis=0)
    mean_corner_err = np.mean(corner_errors)

    return {
        "mean_corner_err": mean_corner_err,
        "acc_1px": float(np.mean(corner_errors <= 1.0) * 100),
        "acc_3px": float(np.mean(corner_errors <= 3.0) * 100),
        "acc_5px": float(np.mean(corner_errors <= 5.0) * 100)
    }
