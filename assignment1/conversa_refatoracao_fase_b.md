# Registo da Conversa: Análise de Redundâncias, Arranjo da Arquitetura e Preparação da Fase B

**Data:** 14 de Outubro de 2026  
**Projeto:** Assignment #1 – Keypoint Detectors, Feature Descriptors and Applications (IACV)

---

## 1. Pergunta Inicial do Utilizador

> *Procura nesta codebase especialmente em `src/` excluindo `src/Experiments/` e diz-me: Há código redundante e qual a melhor forma de o corrigir? O que preciso para começar a Fase B, parte 1 do `plan.md`?*

---

## 2. Diagnóstico de Redundâncias Identificadas em `src/`

1. **Extração de Features Fragmentada e Triplicada**:
   - `src/detectors.py`: Wrappers isolados (`sift_detector`, `fast_detector`, `kaze_detector`, `orb_detector`) e rotina própria de desenho.
   - `src/descriptors.py`: Criação repetida dos mesmos detetores/descritores (`sift_descriptor`, `kaze_descriptor`, `orb_descriptor`, `brief_descriptor`, etc.), além de uma função de matching simplificada (`match`) que colide com o protocolo oficial.
   - `src/feature_extraction.py`: Factory functions duplicadas (`_make_sift`, `_make_kaze`, etc.), gestão de orçamento (`_apply_budget`), controlo de tamanho do FAST e funções unificadas `extract()`, `detect_only()`, `describe_only()`.

2. **Inconsistência e Falha de Integração no `Own BRIEF`**:
   - `src/descriptors.py` possuía uma implementação (`my_brief_descriptor`).
   - `src/own_brief.py` tinha uma implementação parcial com assinaturas não sincronizadas (`generate_sampling_pairs`, `compute_single_descriptor`, `compute_descriptors`, `mybrief_run`).
   - `src/feature_extraction.py` tentava importar `detect_and_compute_own_brief`, `generate_pairs` e `compute_own_brief`, gerando erro de importação porque nenhuma destas funções existia com esses nomes em `own_brief.py`.

3. **Ficheiros Exploratórios / Demonstrações de Aula**:
   - `src/descriptors_fol/class_example.py`: Rascunho com caminhos hardcoded (`./data/images.jpeg`), código incompleto e `cv2.imshow`.
   - `src/descriptors_fol/superglue.py`: Teste isolado de inferência do SuperPoint via HuggingFace com imagens de teste fixas.

4. **Funções Utilitárias Repetidas**:
   - `to_gray()`: Duplicada em `detectors.py`, `descriptors.py` e inline em `stitching.py`.
   - `show_points()`: Repetida em `detectors.py` e `descriptors.py`, competindo com o módulo `visualization.py`.

---

## 3. O que é Necessário para Iniciar a Fase B, Parte 1 (`plan.md`)

1. **Dataset HPatches**:
   - Pasta `hpatches-sequences-release/` presente e acessível com sequências `i_*` e `v_*` (`1.ppm` a `6.ppm` e ficheiros `H_1_2` a `H_1_6`).
2. **Ambiente com `opencv-contrib-python`**:
   - Obrigatório para suportar os módulos `cv2.xfeatures2d` (usados por BRIEF, FREAK, BRISK, KAZE).
   - Dependências: `numpy`, `scipy` (para o algoritmo Húngaro `linear_sum_assignment`), `matplotlib`, `pandas`.
3. **Consolidação do `own_brief.py`**:
   - Como a matriz de B1/B2 inclui `FAST+OwnBRIEF_GI`, `GII`, `GIII`, a interface do `own_brief` tem de estar funcional antes da execução.
4. **Execução do Teste de Identidade (Passo B1.5)**:
   - Correr o par `(1, 1)` com $H = I$ para validar que a repetibilidade é `1.0` e a precisão $\approx 1.0$ em todos os métodos.
5. **Calibração de Orçamento no Tuning Set (Passo B1.2)**:
   - Ajustar limiares nas 6 sequências (`i_dc`, `i_fruits`, `i_fog`, `v_adam`, `v_bark`, `v_there`) para obter $\approx 2000$ keypoints numa imagem de 1024 px.

---

## 4. Reestruturação Solicitada pelo Utilizador

> *Em vez de `src/feature_extraction.py`, quero ficar apenas com `descriptors` e `detectors`. A versão do meu brief deve ser a do `own_brief`. Como arrumar as coisas, sabendo que o `config.py` está confuso? Ter configurações de descritores no `descriptors.py` e de detetores no `detectors.py`.*

---

## 5. Arquitetura Modular Proposta

### Estrutura dos Módulos
```text
src/
├── detectors.py      # Criação dos detetores + parâmetros locais + função detect()
├── descriptors.py    # Criação dos descritores + metadados (norma, bytes) + describe()
├── own_brief.py      # Implementação fiel do BRIEF (G I, G II, G III) chamada por descriptors.py
├── matching.py       # kNN, ratio test 0.8, one-to-one, inliers
├── evaluation.py     # Repetibilidade, PMR, Precision, Matching Score, Homografia
├── data_loader.py    # Leitura de imagens e homografias HPatches
└── config.py         # Apenas protocolo global (caminhos de dados e limiares do benchmark)
```

---

### Módulo: `src/detectors.py`
Centraliza a criação, parâmetros e orçamento de pontos de interesse:
```python
import cv2
import numpy as np

DETECTOR_PARAMS = {
    "SIFT":  {"contrastThreshold": 0.02, "edgeThreshold": 10, "sigma": 1.6},
    "KAZE":  {"threshold": 0.0008, "nOctaves": 4, "nOctaveLayers": 4},
    "ORB":   {"nfeatures": 2000, "scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "patchSize": 31},
    "FAST":  {"threshold": 20, "nonmaxSuppression": True, "type": cv2.FastFeatureDetector_TYPE_9_16},
}

DETECTOR_DEFAULTS = {
    "SIFT":  {"contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
    "KAZE":  {"threshold": 0.001, "nOctaves": 4, "nOctaveLayers": 4},
    "ORB":   {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "patchSize": 31},
    "FAST":  {"threshold": 10, "nonmaxSuppression": True, "type": cv2.FastFeatureDetector_TYPE_9_16},
}

FAST_KP_SIZE = 31.0

def get_detector(name, use_defaults=False):
    params = (DETECTOR_DEFAULTS if use_defaults else DETECTOR_PARAMS).get(name, {})
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=0, **params)
    elif name == "KAZE":
        return cv2.KAZE_create(**params)
    elif name == "ORB":
        return cv2.ORB_create(**params)
    elif name == "FAST":
        return cv2.FastFeatureDetector_create(**params)
    raise ValueError(f"Detetor desconhecido: {name}")

def detect(detector_name, gray, max_keypoints=2000, use_defaults=False):
    """Devolve lista de cv2.KeyPoint ordenados pelo top-N response."""
    det = get_detector(detector_name, use_defaults=use_defaults)
    kps = det.detect(gray, None)
    
    if detector_name == "FAST":
        for kp in kps:
            kp.size = FAST_KP_SIZE
            
    if max_keypoints is not None and len(kps) > max_keypoints:
        indices = np.argsort([-kp.response for kp in kps])[:max_keypoints]
        kps = [kps[i] for i in indices]
        
    return kps
```

---

### Módulo: `src/descriptors.py`
Centraliza a extração dos descritores e os metadados necessários ao matching e tabelas de memória:
```python
import cv2
import numpy as np
from own_brief import compute_own_brief

DESCRIPTORS_INFO = {
    "SIFT":         {"norm": cv2.NORM_L2,      "bytes": 512, "dim": 128, "is_binary": False},
    "KAZE":         {"norm": cv2.NORM_L2,      "bytes": 256, "dim": 64,  "is_binary": False},
    "ORB":          {"norm": cv2.NORM_HAMMING, "bytes": 32,  "dim": 256, "is_binary": True},
    "BRIEF":        {"norm": cv2.NORM_HAMMING, "bytes": 32,  "dim": 256, "is_binary": True},
    "BRISK":        {"norm": cv2.NORM_HAMMING, "bytes": 64,  "dim": 512, "is_binary": True},
    "FREAK":        {"norm": cv2.NORM_HAMMING, "bytes": 64,  "dim": 512, "is_binary": True},
    "OwnBRIEF_GI":  {"norm": cv2.NORM_HAMMING, "bytes": 32,  "dim": 256, "is_binary": True},
    "OwnBRIEF_GII": {"norm": cv2.NORM_HAMMING, "bytes": 32,  "dim": 256, "is_binary": True},
    "OwnBRIEF_GIII":{"norm": cv2.NORM_HAMMING, "bytes": 32,  "dim": 256, "is_binary": True},
}

def get_descriptor_extractor(name):
    if name == "SIFT":
        return cv2.SIFT_create()
    elif name == "KAZE":
        return cv2.KAZE_create()
    elif name == "ORB":
        return cv2.ORB_create()
    elif name == "BRIEF":
        return cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)
    elif name == "BRISK":
        return cv2.BRISK_create()
    elif name == "FREAK":
        return cv2.xfeatures2d.FREAK_create()
    return None

def describe(descriptor_name, gray, keypoints):
    """Devolve (kps_validos, descriptors)."""
    if not keypoints:
        return [], None
        
    if descriptor_name.startswith("OwnBRIEF"):
        strategy = descriptor_name.split("_")[1]
        return compute_own_brief(gray, keypoints, strategy=strategy, n_bits=256)
        
    extractor = get_descriptor_extractor(descriptor_name)
    kps, descs = extractor.compute(gray, keypoints)
    return kps, descs
```

---

### Módulo: `src/own_brief.py`
Implementa o algoritmo de Calonder et al. (B4 do plano) com pré-suavização, amostragem fixa e empacotamento em bytes:
```python
import cv2
import numpy as np

def generate_pairs(strategy="GII", n_bits=256, patch_size=48, seed=42):
    rng = np.random.default_rng(seed)
    r = patch_size // 2
    if strategy == "GI":
        p1 = rng.uniform(-r, r, size=(n_bits, 2))
        p2 = rng.uniform(-r, r, size=(n_bits, 2))
    elif strategy == "GII":
        sigma = patch_size / 5.0
        p1 = rng.normal(0, sigma, size=(n_bits, 2))
        p2 = rng.normal(0, sigma, size=(n_bits, 2))
    elif strategy == "GIII":
        sigma1 = patch_size / 5.0
        sigma2 = patch_size / 10.0
        p1 = rng.normal(0, sigma1, size=(n_bits, 2))
        p2 = p1 + rng.normal(0, sigma2, size=(n_bits, 2))
    p1 = np.clip(np.round(p1), -r, r).astype(int)
    p2 = np.clip(np.round(p2), -r, r).astype(int)
    return p1, p2

def compute_own_brief(image, keypoints, strategy="GII", n_bits=256, patch_size=48, seed=42):
    blurred = cv2.GaussianBlur(image, (9, 9), 2.0)
    p1, p2 = generate_pairs(strategy, n_bits, patch_size, seed)
    
    h, w = blurred.shape[:2]
    r = patch_size // 2
    valid_kps, bits_list = [], []
    
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            continue
        vals1 = blurred[y + p1[:, 1], x + p1[:, 0]]
        vals2 = blurred[y + p2[:, 1], x + p2[:, 0]]
        bits_list.append(vals1 < vals2)
        valid_kps.append(kp)
        
    if not bits_list:
        return [], None
    descs = np.packbits(np.array(bits_list, dtype=np.uint8), axis=1)
    return valid_kps, descs
```

---

### Módulo: `src/config.py` (Versão Simplificada)
Mantém apenas parâmetros globais e regras do protocolo de avaliação:
```python
HPATCHES_DIR = "hpatches-sequences-release"
MAX_RESOLUTION_SIDE = 1024
MAX_KEYPOINTS = 2000

KNN_K = 2
RATIO_TEST_THRESHOLD = 0.8
CORRECT_MATCH_THRESHOLD_PX = 3.0

RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995
RANDOM_SEED = 42

TUNING_SEQUENCES = {
    "i_dc", "i_fruits", "i_fog",
    "v_adam", "v_bark", "v_there",
}
```

---

### 6. Como Fica a Execução nos Experimentos da Fase B

```python
import detectors
import descriptors
import matching

# Exemplo de par (1, k)
kps1 = detectors.detect("FAST", img1)
kps1, descs1 = descriptors.describe("BRISK", img1, kps1)

kps2 = detectors.detect("FAST", img2)
kps2, descs2 = descriptors.describe("BRISK", img2, kps2)

norm = descriptors.DESCRIPTORS_INFO["BRISK"]["norm"]
matches = matching.match_pipeline(descs1, descs2, norm)
```
Esta separação facilita a medição isolada de tempos de deteção vs descrição requerida pelo passo A3/B1 do plano de trabalho.
