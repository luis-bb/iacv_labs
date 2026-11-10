"""
Retrieval helpers for Part 5 — Open-Ended Image Retrieval.

Responsibilities:
- Load and validate query/corpus/relevance manifests.
- Align feature matrices from different sources to a canonical filename order.
- Compute cosine similarity rankings.
- Compute AP, Precision@K, Recall@K, and mAP.
- Save ranked results and produce visualizations.
"""

from time import perf_counter
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RETRIEVAL_DIR = DATA_DIR / "retrieval"
RESULTS_DIR = Path("results") / "part5"
FIGURES_DIR = RESULTS_DIR / "figures"

K_VALUES = [5, 10, 20]  # for Precision@K and Recall@K



def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_queries(path: Union[str, Path] = None) -> pd.DataFrame:
    """Load image_queries.csv (query_id, filename, optional notes)."""
    path = RETRIEVAL_DIR / "image_queries.csv" if path is None else Path(path)
    df = pd.read_csv(path, dtype={"query_id": str, "filename": str})
    _require_columns(df, ("query_id", "filename"), "image_queries.csv")
    if df.empty or df[["query_id", "filename"]].isna().any().any():
        raise ValueError("Queries must contain nonblank IDs and filenames")
    df["query_id"] = df["query_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    if df["query_id"].str.strip().eq("").any() or df["filename"].str.strip().eq("").any():
        raise ValueError("Query IDs and filenames cannot be empty strings")
    if not df["query_id"].is_unique:
        raise ValueError("Duplicate query IDs found")
    if not df["filename"].is_unique:
        raise ValueError("Duplicate query filenames found")
    return df


def load_corpus(path: Union[str, Path] = None) -> pd.DataFrame:
    """Load corpus.csv (filename, category)."""
    path = RETRIEVAL_DIR / "corpus.csv" if path is None else Path(path)
    df = pd.read_csv(path, dtype={"filename": str, "category": str})
    _require_columns(df, ("filename", "category"), "corpus.csv")
    if df.empty or df[["filename", "category"]].isna().any().any():
        raise ValueError("Corpus must contain nonblank filenames and categories")
    df["filename"] = df["filename"].astype(str)
    df["category"] = df["category"].astype(str)
    if df["filename"].str.strip().eq("").any() or df["category"].str.strip().eq("").any():
        raise ValueError("Corpus filenames and categories cannot be empty strings")
    if not df["filename"].is_unique:
        raise ValueError("Duplicate corpus filenames found")
    return df


def load_relevance(path: Union[str, Path] = None) -> pd.DataFrame:
    """Load sparse or exhaustive binary relevance judgments.

    Omitted query/filename pairs are interpreted as irrelevant.
    """
    path = RETRIEVAL_DIR / "relevance.csv" if path is None else Path(path)
    df = pd.read_csv(path, dtype={"query_id": str, "filename": str})
    _require_columns(df, ("query_id", "filename", "relevant"), "relevance.csv")
    if df[["query_id", "filename", "relevant"]].isna().any().any():
        raise ValueError("Relevance IDs, filenames, and labels cannot be blank")
    df["query_id"] = df["query_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    if df["query_id"].str.strip().eq("").any() or df["filename"].str.strip().eq("").any():
        raise ValueError("Relevance IDs and filenames cannot be empty strings")
    df["relevant"] = pd.to_numeric(df["relevant"], errors="raise")
    if not df["relevant"].isin([0, 1]).all():
        raise ValueError("Relevance labels must be binary 0/1")
    df["relevant"] = df["relevant"].astype(int)
    return df


def load_clip_prompts(path: Union[str, Path] = None) -> pd.DataFrame:
    """Load CLIP prompts (prompt_id, prompt_text, optional detail metadata)."""
    path = RETRIEVAL_DIR / "clip_prompts.csv" if path is None else Path(path)
    df = pd.read_csv(path, dtype={"prompt_id": str, "prompt_text": str})
    _require_columns(df, ("prompt_id", "prompt_text"), "clip_prompts.csv")
    if df.empty or df[["prompt_id", "prompt_text"]].isna().any().any():
        raise ValueError("CLIP prompts must contain nonblank IDs and text")
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["prompt_text"] = df["prompt_text"].astype(str)
    if df["prompt_id"].str.strip().eq("").any() or df["prompt_text"].str.strip().eq("").any():
        raise ValueError("CLIP prompt IDs and text cannot be empty strings")
    if not df["prompt_id"].is_unique:
        raise ValueError("Duplicate CLIP prompt IDs found")
    return df


def validate_manifests(queries_df: pd.DataFrame,
                       corpus_df: pd.DataFrame,
                       relevance_df: pd.DataFrame,
                       query_id_col: str = "query_id") -> None:
    """Validate query/corpus/relevance consistency.

    Relevance rows may be sparse: omitted pairs are irrelevant. Each query
    must have at least one positive corpus item.
    """
    _require_columns(queries_df, (query_id_col,), "queries/prompts manifest")
    _require_columns(corpus_df, ("filename", "category"), "corpus manifest")
    _require_columns(relevance_df, ("query_id", "filename", "relevant"),
                     "relevance manifest")
    if queries_df.empty or corpus_df.empty:
        raise ValueError("Query and corpus manifests must not be empty")
    if queries_df[query_id_col].isna().any():
        raise ValueError(f"{query_id_col} values cannot be blank")
    if queries_df[query_id_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"{query_id_col} values cannot be empty strings")
    query_ids = set(queries_df[query_id_col].astype(str))
    if len(query_ids) != len(queries_df):
        raise ValueError(f"Duplicate values in {query_id_col}")
    corpus_filenames = set(corpus_df["filename"].astype(str))
    if corpus_df["filename"].isna().any() or corpus_df["filename"].astype(str).str.strip().eq("").any():
        raise ValueError("Corpus filenames cannot be blank")
    if len(corpus_filenames) != len(corpus_df):
        raise ValueError("Duplicate corpus filenames found")

    rel = relevance_df.copy()
    if rel[["query_id", "filename", "relevant"]].isna().any().any():
        raise ValueError("Relevance IDs, filenames, and labels cannot be blank")
    rel["query_id"] = rel["query_id"].astype(str)
    rel["filename"] = rel["filename"].astype(str)
    rel["relevant"] = pd.to_numeric(rel["relevant"], errors="raise")
    if rel.duplicated(["query_id", "filename"]).any():
        raise ValueError("Duplicate query/corpus pairs in relevance manifest")
    query_filenames = (
        set(queries_df["filename"].astype(str))
        if "filename" in queries_df.columns else set()
    )
    overlap = query_filenames & corpus_filenames
    if overlap:
        raise ValueError(f"Query/corpus overlap: {sorted(overlap)[:10]}")

    rel_ids = set(rel["query_id"])
    rel_filenames = set(rel["filename"])
    missing_ids = rel_ids - query_ids
    missing_files = rel_filenames - corpus_filenames
    if missing_ids:
        raise ValueError(f"Unknown relevance query IDs: {sorted(missing_ids)[:10]}")
    if missing_files:
        raise ValueError(f"Relevance filenames not in corpus: {sorted(missing_files)[:10]}")
    if not rel["relevant"].isin([0, 1]).all():
        raise ValueError("Relevance labels must be binary 0/1")

    positive_counts = rel.groupby("query_id")["relevant"].sum()
    no_positive = [
        qid for qid in query_ids if int(positive_counts.get(qid, 0)) == 0
    ]
    if no_positive:
        raise ValueError(f"Queries without relevant corpus items: {sorted(no_positive)}")


def validate_image_files(filenames: Sequence[str],
                         img_dir: Union[str, Path] = DATA_DIR / "images") -> None:
    """Raise a clear error if any manifest image is absent from the image directory."""
    img_dir = Path(img_dir)
    missing = [str(name) for name in filenames if not _image_path(img_dir, str(name)).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing from {img_dir}: {missing[:10]}"
        )



def _align_feature_rows(feature_names: Sequence[str],
                        features: np.ndarray,
                        filenames: Sequence[str],
                        source_name: str) -> np.ndarray:
    """Validate a feature archive and return rows in requested filename order."""
    feature_names = [str(name) for name in feature_names]
    filenames = [str(name) for name in filenames]
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"{source_name} features must be 2-D, got {features.shape}")
    if len(feature_names) != features.shape[0]:
        raise ValueError(
            f"{source_name}: {len(feature_names)} names for {features.shape[0]} rows"
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"{source_name} contains duplicate filenames")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Requested filename list contains duplicates")
    if not np.isfinite(features).all():
        raise ValueError(f"{source_name} contains NaN or infinite feature values")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in filenames if name not in name_to_idx]
    if missing:
        raise ValueError(f"{source_name} is missing requested filenames: {missing[:10]}")
    aligned = features[[name_to_idx[name] for name in filenames]]
    return aligned


def _deduplicate_identical_rows(feature_names: Sequence[str],
                                features: np.ndarray,
                                source_name: str
                                ) -> Tuple[List[str], np.ndarray]:
    """Collapse repeated filenames only when their stored feature rows agree."""
    names = [str(name) for name in feature_names]
    features = np.asarray(features)
    if features.ndim != 2 or len(names) != features.shape[0]:
        raise ValueError(f"{source_name}: feature rows and filenames do not align")
    indices = {}
    for i, name in enumerate(names):
        if name in indices and not np.array_equal(features[indices[name]], features[i]):
            raise ValueError(f"{source_name} has conflicting cached rows for {name}")
        indices.setdefault(name, i)
    unique_names = list(indices)
    unique_features = features[[indices[name] for name in unique_names]]
    return unique_names, unique_features


def load_handcrafted_features(encoding_name: str,
                              filenames: Sequence[str],
                              data_dir: Union[str, Path] = DATA_DIR,
                              filename_categories: Optional[Dict[str, str]] = None
                              ) -> np.ndarray:
    """Load val BoVW/VLAD/FV archives and align rows by filename.

    Some validation images occur in multiple class split files. Identical
    feature copies are deduplicated. If copies differ, ``filename_categories``
    selects the row from the archive for that image's manifest category.
    """
    encoding_dir = Path(data_dir) / "encodings" / encoding_name
    archive_paths = sorted(encoding_dir.glob("*_val.npz"))
    if not archive_paths:
        raise FileNotFoundError(f"No validation feature archives in {encoding_dir}")

    rows_by_name = {}
    for archive_path in archive_paths:
        archive_category = archive_path.stem.removesuffix("_val")
        with np.load(archive_path, allow_pickle=True) as archive:
            names = np.asarray(archive["file_names"]).astype(str).tolist()
            features = np.asarray(archive["histograms"])
            for name, row in zip(names, features):
                rows_by_name.setdefault(name, []).append((archive_category, row))

    selected_names, selected_features = [], []
    for name, candidates in rows_by_name.items():
        chosen = candidates[0][1]
        if any(not np.array_equal(chosen, row) for _, row in candidates[1:]):
            preferred_category = (
                filename_categories.get(name)
                if filename_categories is not None else None
            )
            matching = [row for category, row in candidates
                        if category == preferred_category]
            if len(matching) != 1:
                sources = [category for category, _ in candidates]
                raise ValueError(
                    f"Conflicting {encoding_name} features for {name} in "
                    f"{sources}; pass its manifest category to select the correct row"
                )
            chosen = matching[0]
        selected_names.append(name)
        selected_features.append(chosen)

    return _align_feature_rows(
        selected_names, np.vstack(selected_features), filenames, encoding_name
    )


def load_deep_features_aligned(model_name: str,
                               filenames: Sequence[str],
                               data_dir: Union[str, Path] = DATA_DIR) -> np.ndarray:
    """Load cached ResNet-18, ViT-B/32, or DINOv2 val features by filename."""
    file_map = {
        "resnet18": "resnet18_features.npz",
        "dinov2_vits14": "dinov2_vits14_features.npz",
        "vit_b_32": "vit_b_32_features.npz",
    }
    if model_name not in file_map:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {list(file_map)}")

    feature_path = Path(data_dir) / "deep_features" / file_map[model_name]
    with np.load(feature_path, allow_pickle=True) as archive:
        names, features = _deduplicate_identical_rows(
            archive["val_names"], archive["x_val"], model_name
        )
        return _align_feature_rows(
            names, features, filenames, model_name
        )


def load_clip_image_features_aligned(
    filenames: Sequence[str],
    data_dir: Union[str, Path] = DATA_DIR,
) -> np.ndarray:
    """Load cached CLIP validation image embeddings by filename."""
    feature_path = Path(data_dir) / "deep_features" / "clip_vit_b_32_val_images.npz"
    with np.load(feature_path, allow_pickle=True) as archive:
        names, features = _deduplicate_identical_rows(
            archive["val_names"], archive["image_features"], "CLIP image"
        )
        return _align_feature_rows(
            names, features, filenames, "CLIP image"
        )


def encode_clip_text_queries(prompts_df: pd.DataFrame,
                             clip_model,
                             device: str = "cpu",
                             batch_size: int = 32) -> Tuple[List[str], np.ndarray]:
    """Encode prompt_text rows as normalized CLIP text-query vectors.

    Returns prompt IDs in row order and a (n_prompts, embedding_dim) matrix.
    The model should be loaded with the project helper in src.deep_models.
    """
    _require_columns(prompts_df, ("prompt_id", "prompt_text"), "CLIP prompts")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    import clip
    import torch

    prompt_ids = prompts_df["prompt_id"].astype(str).tolist()
    prompt_texts = prompts_df["prompt_text"].astype(str).tolist()
    vectors = []
    clip_model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompt_texts), batch_size):
            tokens = clip.tokenize(prompt_texts[start:start + batch_size]).to(device)
            text_features = clip_model.encode_text(tokens).float()
            vectors.append(text_features.cpu().numpy())

    if not vectors:
        raise ValueError("No CLIP prompts provided")
    return prompt_ids, l2_normalize(np.vstack(vectors))


def benchmark_image_encoder(encoder,
                            image_paths: Sequence[Union[str, Path]],
                            cache_dir: Optional[Union[str, Path]] = None
                            ) -> Dict[str, float]:
    """Measure end-to-end per-image encode time for an ImageEncoder-like object.

    Model/config loading is excluded. Set cache_dir=None to include dense local
    descriptor extraction for every image.
    """
    if len(image_paths) == 0:
        raise ValueError("At least one image path is required")
    start = perf_counter()
    for image_path in image_paths:
        encoder.encode(Path(image_path), cache_dir=cache_dir)
    elapsed = perf_counter() - start
    return {
        "n_images": len(image_paths),
        "total_encoding_s": elapsed,
        "mean_encoding_ms_per_image": 1000 * elapsed / len(image_paths),
    }


def measure_storage(paths: Sequence[Union[str, Path]]) -> Dict[str, float]:
    """Measure bytes occupied by files and/or recursively by directories."""
    files = set()
    for item in paths:
        path = Path(item)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError(path)
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "file_count": len(files),
        "storage_bytes": total_bytes,
        "storage_MiB": total_bytes / (1024 ** 2),
    }


def save_retrieval_outputs(metrics_df: pd.DataFrame,
                           rankings_df: pd.DataFrame,
                           method_name: str,
                           output_dir: Union[str, Path] = RESULTS_DIR
                           ) -> Tuple[Path, Path]:
    """Save one method's metric and ranking tables as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in method_name)
    metrics_path = output_dir / f"{safe_name}_metrics.csv"
    rankings_path = output_dir / f"{safe_name}_rankings.csv"
    metrics_df.to_csv(metrics_path, index=False)
    rankings_df.to_csv(rankings_path, index=False)
    return metrics_path, rankings_path


# ---------------------------------------------------------------------------
# Normalization & similarity
# ---------------------------------------------------------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows of X. Avoids division by zero."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Feature matrix contains zero-length rows")
    return X / norms


def cosine_similarity_matrix(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query rows and corpus rows.
    
    Args:
        Q: (n_queries, dim) — already L2-normalized
        C: (n_corpus, dim) — already L2-normalized
    
    Returns:
        sim: (n_queries, n_corpus) similarity matrix
    """
    Q = np.asarray(Q)
    C = np.asarray(C)
    if Q.ndim != 2 or C.ndim != 2:
        raise ValueError("Query and corpus features must be 2-D matrices")
    if Q.shape[1] != C.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: query={Q.shape[1]}, corpus={C.shape[1]}"
        )
    if not np.isfinite(Q).all() or not np.isfinite(C).all():
        raise ValueError("Query or corpus features contain NaN or infinite values")
    return Q @ C.T


def rank_by_similarity(sim_scores: np.ndarray) -> np.ndarray:
    """Return indices that sort each row in descending order of similarity.
    
    Args:
        sim_scores: (n_queries, n_corpus)
    
    Returns:
        ranked_indices: (n_queries, n_corpus) — column indices sorted desc
    """
    sim_scores = np.asarray(sim_scores)
    if sim_scores.ndim != 2 or not np.isfinite(sim_scores).all():
        raise ValueError("Similarity scores must be a finite 2-D matrix")
    # Stable sorting uses canonical corpus order to break score ties.
    return np.argsort(-sim_scores, axis=1, kind="stable")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def average_precision(relevant: np.ndarray) -> float:
    """Compute Average Precision from a binary relevance vector 
    sorted by descending score.
    
    Args:
        relevant: 1D bool/int array where relevant[i] = 1 if the i-th 
                  ranked item is relevant.
    
    Returns:
        AP value (0 to 1).
    """
    relevant = np.asarray(relevant)
    if relevant.ndim != 1 or not np.isin(relevant, (0, 1, False, True)).all():
        raise ValueError("Relevance must be a 1-D binary vector")
    relevant = relevant.astype(float)
    n_relevant = relevant.sum()
    if n_relevant == 0:
        return 0.0
    
    cumsum = np.cumsum(relevant)
    precision_at_k = cumsum / np.arange(1, len(relevant) + 1)
    ap = (precision_at_k * relevant).sum() / n_relevant
    return float(ap)


def precision_at_k(relevant: np.ndarray, k: int) -> float:
    """Precision@K from a sorted binary relevance vector."""
    relevant = np.asarray(relevant)
    if relevant.ndim != 1 or not np.isin(relevant, (0, 1, False, True)).all():
        raise ValueError("Relevance must be a 1-D binary vector")
    if k <= 0:
        raise ValueError("k must be positive")
    denominator = min(k, len(relevant))
    if denominator == 0:
        return 0.0
    return float(relevant[:k].sum() / denominator)


def recall_at_k(relevant: np.ndarray, k: int, n_total_relevant: int) -> float:
    """Recall@K from a sorted binary relevance vector."""
    relevant = np.asarray(relevant)
    if relevant.ndim != 1 or not np.isin(relevant, (0, 1, False, True)).all():
        raise ValueError("Relevance must be a 1-D binary vector")
    if k <= 0:
        raise ValueError("k must be positive")
    if n_total_relevant < 0:
        raise ValueError("n_total_relevant cannot be negative")
    if n_total_relevant == 0:
        return 0.0
    return float(relevant[:k].sum() / n_total_relevant)


def compute_retrieval_metrics(ranked_relevance: np.ndarray,
                               n_total_relevant: int,
                               ks: List[int] = None
                               ) -> Dict[str, float]:
    """Compute all retrieval metrics for a single query.
    
    Args:
        ranked_relevance: 1D binary array, sorted by descending score.
        n_total_relevant: total number of relevant items in the corpus.
        ks: list of K values for P@K and R@K.
    
    Returns:
        Dictionary with AP, P@K, R@K for each K.
    """
    if ks is None:
        ks = K_VALUES
    if n_total_relevant != int(np.asarray(ranked_relevance).sum()):
        raise ValueError("n_total_relevant must match positives in ranked_relevance")

    metrics = {"AP": average_precision(ranked_relevance)}
    for k in ks:
        metrics[f"P@{k}"] = precision_at_k(ranked_relevance, k)
        metrics[f"R@{k}"] = recall_at_k(ranked_relevance, k, n_total_relevant)
    
    return metrics


# ---------------------------------------------------------------------------
# Full retrieval pipeline
# ---------------------------------------------------------------------------


def run_retrieval(query_features: np.ndarray,
                  corpus_features: np.ndarray,
                  corpus_filenames: Sequence[str],
                  relevance_per_query: Dict[str, np.ndarray],
                  query_ids: Sequence[str],
                  method_name: str,
                  normalize: bool = True
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the corpus for each query and return metrics and rankings.

    Query features may be image embeddings or CLIP text embeddings. If
    normalize=False, both matrices must already be L2-normalized.
    """
    query_ids = [str(qid) for qid in query_ids]
    corpus_filenames = [str(filename) for filename in corpus_filenames]
    query_features = np.asarray(query_features)
    corpus_features = np.asarray(corpus_features)

    if query_features.ndim != 2 or corpus_features.ndim != 2:
        raise ValueError("Query and corpus features must be 2-D matrices")
    if query_features.shape[0] != len(query_ids):
        raise ValueError("Query feature rows must match query_ids")
    if corpus_features.shape[0] != len(corpus_filenames):
        raise ValueError("Corpus feature rows must match corpus_filenames")
    if query_features.shape[1] != corpus_features.shape[1]:
        raise ValueError("Query and corpus feature dimensions do not match")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique")
    if len(set(corpus_filenames)) != len(corpus_filenames):
        raise ValueError("corpus_filenames must be unique")
    if not query_ids or not corpus_filenames:
        raise ValueError("At least one query and one corpus image are required")

    relevance_per_query = {str(qid): rel for qid, rel in relevance_per_query.items()}
    missing_relevance = set(query_ids) - set(relevance_per_query)
    if missing_relevance:
        raise ValueError(f"Missing relevance vectors for queries: {sorted(missing_relevance)}")
    extra_relevance = set(relevance_per_query) - set(query_ids)
    if extra_relevance:
        raise ValueError(f"Relevance provided for unknown queries: {sorted(extra_relevance)}")

    validated_relevance = {}
    for qid in query_ids:
        rel = np.asarray(relevance_per_query[qid])
        if rel.shape != (len(corpus_filenames),):
            raise ValueError(
                f"Relevance vector for {qid} has shape {rel.shape}; "
                f"expected ({len(corpus_filenames)},)"
            )
        if not np.isin(rel, (0, 1, False, True)).all():
            raise ValueError(f"Relevance vector for {qid} must contain only 0/1")
        if not np.any(rel):
            raise ValueError(f"Query {qid} has no relevant corpus images")
        validated_relevance[qid] = rel.astype(int)

    if normalize:
        query_features = l2_normalize(query_features)
        corpus_features = l2_normalize(corpus_features)
    else:
        query_norms = np.linalg.norm(query_features, axis=1)
        corpus_norms = np.linalg.norm(corpus_features, axis=1)
        if not np.allclose(query_norms, 1.0, atol=1e-4) or not np.allclose(
            corpus_norms, 1.0, atol=1e-4
        ):
            raise ValueError(
                "normalize=False requires pre-normalized query and corpus features"
            )

    t_start = perf_counter()
    sim = cosine_similarity_matrix(query_features, corpus_features)
    ranked_idx = rank_by_similarity(sim)
    query_total_s = perf_counter() - t_start
    query_per_item_s = query_total_s / len(query_ids)

    all_metrics, all_rankings = [], []
    corpus_filenames = np.asarray(corpus_filenames)
    for i, qid in enumerate(query_ids):
        rel_vector = validated_relevance[qid]
        n_relevant = int(rel_vector.sum())
        ranked_rel = rel_vector[ranked_idx[i]]
        ranked_fnames = corpus_filenames[ranked_idx[i]]
        ranked_scores = sim[i, ranked_idx[i]]

        metrics = compute_retrieval_metrics(ranked_rel, n_relevant)
        metrics.update({
            "query_id": qid,
            "method": method_name,
            "n_relevant": n_relevant,
            "query_time_s": query_per_item_s,
            "feature_dim": int(query_features.shape[1]),
        })
        all_metrics.append(metrics)

        all_rankings.extend(
            {
                "query_id": qid,
                "method": method_name,
                "rank": rank,
                "filename": filename,
                "score": float(score),
                "relevant": int(relevant),
            }
            for rank, (filename, score, relevant) in enumerate(
                zip(ranked_fnames, ranked_scores, ranked_rel), start=1
            )
        )

    metrics_df = pd.DataFrame(all_metrics)
    summary = {
        "query_id": "mAP",
        "method": method_name,
        "AP": float(metrics_df["AP"].mean()),
        "n_relevant": "",
        "query_time_s": query_per_item_s,
        "query_time_total_s": query_total_s,
        "feature_dim": int(query_features.shape[1]),
    }
    for k in K_VALUES:
        summary[f"P@{k}"] = float(metrics_df[f"P@{k}"].mean())
        summary[f"R@{k}"] = float(metrics_df[f"R@{k}"].mean())
    metrics_df["query_time_total_s"] = query_total_s
    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], ignore_index=True)
    rankings_df = pd.DataFrame(all_rankings)
    return metrics_df, rankings_df


# ---------------------------------------------------------------------------
# Build relevance vectors from manifests
# ---------------------------------------------------------------------------


def build_relevance_vectors(query_df: pd.DataFrame,
                            corpus_df: pd.DataFrame,
                            relevance_df: pd.DataFrame,
                            query_id_col: str = "query_id"
                            ) -> Dict[str, np.ndarray]:
    """Build relevance vectors for image queries or CLIP prompt queries.

    Use query_id_col="prompt_id" for the CLIP prompt manifest. Sparse
    relevance manifests are supported: omitted pairs are treated as negatives.
    """
    _require_columns(query_df, (query_id_col,), "query manifest")
    _require_columns(corpus_df, ("filename",), "corpus manifest")
    _require_columns(relevance_df, ("query_id", "filename", "relevant"),
                     "relevance manifest")

    corpus_filenames = corpus_df["filename"].astype(str).tolist()
    if len(set(corpus_filenames)) != len(corpus_filenames):
        raise ValueError("Corpus filenames must be unique")
    fname_to_idx = {filename: idx for idx, filename in enumerate(corpus_filenames)}
    query_ids = query_df[query_id_col].astype(str).tolist()
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"{query_id_col} values must be unique")

    rel = relevance_df.copy()
    rel["query_id"] = rel["query_id"].astype(str)
    rel["filename"] = rel["filename"].astype(str)
    rel["relevant"] = pd.to_numeric(rel["relevant"], errors="raise")
    if rel.duplicated(["query_id", "filename"]).any():
        raise ValueError("Duplicate query/corpus pairs in relevance manifest")
    if not rel["relevant"].isin([0, 1]).all():
        raise ValueError("Relevance labels must be binary 0/1")

    unknown_queries = set(rel["query_id"]) - set(query_ids)
    unknown_files = set(rel["filename"]) - set(corpus_filenames)
    if unknown_queries or unknown_files:
        raise ValueError(
            f"Unknown relevance references: query_ids={sorted(unknown_queries)[:10]}, "
            f"filenames={sorted(unknown_files)[:10]}"
        )

    result = {}
    for qid in query_ids:
        rel_vector = np.zeros(len(corpus_filenames), dtype=np.int8)
        rows = rel[rel["query_id"] == qid]
        for row in rows.itertuples(index=False):
            rel_vector[fname_to_idx[row.filename]] = int(row.relevant)
        if not np.any(rel_vector):
            raise ValueError(f"Query {qid} has no relevant corpus images")
        result[qid] = rel_vector
    return result



def _image_path(img_dir: Path, filename: str) -> Path:
    path = img_dir / filename
    return path if path.suffix else path.with_suffix(".jpg")


def plot_top_results(query_id: str,
                     query_filename: Optional[str],
                     rankings_df: pd.DataFrame,
                     method_name: str,
                     top_k: int = 10,
                     img_dir: Union[str, Path] = DATA_DIR / "images",
                     save_path: Optional[Union[str, Path]] = None,
                     query_text: Optional[str] = None):
    """Plot an image query or text query beside its top-K retrieved results."""
    import matplotlib.pyplot as plt
    from PIL import Image

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {"query_id", "method", "rank", "filename", "score", "relevant"}
    _require_columns(rankings_df, tuple(required), "rankings table")
    mask = (rankings_df["query_id"].astype(str) == str(query_id)) & \
           (rankings_df["method"] == method_name)
    subset = rankings_df.loc[mask].sort_values("rank").head(top_k)
    if subset.empty:
        raise ValueError(f"No rankings for query={query_id}, method={method_name}")

    img_dir = Path(img_dir)
    n_cols = len(subset) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3), squeeze=False)
    axes = axes[0]

    if query_filename:
        qpath = _image_path(img_dir, str(query_filename))
        if qpath.is_file():
            with Image.open(qpath) as query_image:
                axes[0].imshow(query_image.convert("RGB"))
        axes[0].set_title(f"Query\n{query_filename}", fontsize=8, fontweight="bold")
    else:
        axes[0].text(
            0.5, 0.5, query_text or str(query_id),
            ha="center", va="center", wrap=True, fontsize=10,
        )
        axes[0].set_title(f"Text query\n{query_id}", fontsize=8, fontweight="bold")
    axes[0].axis("off")

    for j, row in enumerate(subset.itertuples(index=False), start=1):
        ax = axes[j]
        image_path = _image_path(img_dir, str(row.filename))
        if image_path.is_file():
            with Image.open(image_path) as result_image:
                ax.imshow(result_image.convert("RGB"))
        color = "green" if int(row.relevant) else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(
            f"#{row.rank} ({row.score:.3f})\n{row.filename}",
            fontsize=7, color=color,
        )
        ax.axis("off")

    fig.suptitle(f"{method_name} — Query: {query_id}", fontsize=10, y=1.02)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_comparison(all_metrics: pd.DataFrame,
                            save_path: Optional[Union[str, Path]] = None):
    """Bar chart comparing mAP across methods."""
    import matplotlib.pyplot as plt

    _require_columns(all_metrics, ("query_id", "method", "AP"), "metrics table")
    map_rows = all_metrics[all_metrics["query_id"].astype(str) == "mAP"].copy()
    if map_rows.empty:
        raise ValueError("No mAP summary rows found in metrics table")

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = map_rows["method"].values
    maps = map_rows["AP"].astype(float).values
    bars = ax.bar(range(len(methods)), maps, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mAP")
    ax.set_title("Mean Average Precision — All Methods")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
