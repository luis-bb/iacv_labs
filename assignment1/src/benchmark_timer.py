"""
benchmark_timer.py - Medição rigorosa de tempo (Warm-up, Mediana, CPU/GPU)
"""

import time
import statistics
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def synchronize_device(is_gpu=False):
    """Sincroniza o dispositivo CUDA se PyTorch estiver disponível e em uso."""
    if is_gpu and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_function(func, *args, is_gpu=False, num_repeats=5, **kwargs):
    """
    Executa uma função com 1 warm-up e mede a mediana de pelo menos `num_repeats` repetições.
    Separa a medição e lida com sincronização de GPU se aplicável.
    
    Args:
        func: Função a ser medida (ex: detecção, descrição ou matching).
        is_gpu: Booleano indicando se corre em GPU (ex: SuperPoint, D2-Net).
        num_repeats: Número de repetições para o cálculo da mediana (default: 5).
        
    Returns:
        tuple: (resultado da função na última execução, tempo mediano em segundos)
    """
    # 1. Execução de Warm-up (essencial para carregar caches, kernels CUDA, etc.)
    synchronize_device(is_gpu)
    _ = func(*args, **kwargs)
    synchronize_device(is_gpu)

    timings = []
    result = None

    # 2. ≥ 5 repetições
    for _ in range(num_repeats):
        synchronize_device(is_gpu)
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        synchronize_device(is_gpu)
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)

    # 3. Cálculo da mediana
    median_time = statistics.median(timings)
    
    return result, median_time


def measure_pipeline_steps(detector_describer_func, matcher_func, img, img_ref=None, is_gpu=False, num_repeats=5):
    """
    Exemplo prático de como separar a medição de:
    - Detecção & Descrição (combinadas ou separadas consoante a arquitetura)
    - Matching
    """
    # Se a detecção e descrição forem juntas (ex: SIFT, ORB, SuperPoint)
    def run_detect_describe():
        return detector_describer_func(img)

    # Medir Detecção/Descrição
    (kps, descs), time_det_desc = benchmark_function(
        run_detect_describe, is_gpu=is_gpu, num_repeats=num_repeats
    )

    time_matching = 0.0
    matches = None

    if img_ref is not None:
        # Assumindo que temos os descritores da imagem de referência pré-calculados (descs_ref)
        # Passamos para a função de matching
        def run_matching():
            return matcher_func(descs_ref, descs)

        # Se houver imagem de referência, medimos o matching em separado
        # (Nota: ajusta os argumentos conforme a tua função real de matching)
        pass 

    return {
        "keypoints": kps,
        "descriptors": descs,
        "time_detection_description_s": time_det_desc,
        "time_matching_s": time_matching
    }
