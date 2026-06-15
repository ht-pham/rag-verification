#!/usr/bin/env python3
"""Inspect a local FAISS vector store without using Hugging Face embeddings.

Usage:
    python analysis/inspect_faiss_store.py
    python analysis/inspect_faiss_store.py --path data/pubmed_faiss_index
"""

import argparse
import math
import os
import pickle
from pathlib import Path

import faiss
import numpy as np


def load_store(path: str):
    base = Path(path).resolve()
    index_path = base / "index.faiss"
    pickle_path = base / "index.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not pickle_path.exists():
        raise FileNotFoundError(f"Missing FAISS metadata: {pickle_path}")

    index = faiss.read_index(str(index_path))
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)

    return base, index, payload


def describe_index(index: faiss.Index) -> dict:
    metric_name = {
        faiss.METRIC_L2: "L2",
        faiss.METRIC_INNER_PRODUCT: "IP",
    }.get(index.metric_type, str(index.metric_type))

    stats = {
        "class": index.__class__.__name__,
        "ntotal": int(index.ntotal),
        "dimension": int(index.d),
        "is_trained": bool(index.is_trained),
        "metric_type": metric_name,
        "nprobe": getattr(index, "nprobe", None),
    }

    try:
        sample_count = min(100, stats["ntotal"])
        vectors = index.reconstruct_n(0, sample_count)
        vectors = np.asarray(vectors, dtype=np.float32)
        stats['vector_id_0'] = vectors[0]
        #stats["sample_vectors"] = sample_count
        stats["sample_mean_norm"] = float(np.linalg.norm(vectors, axis=1).mean())
        stats["sample_mean"] = float(vectors.mean())
        stats["sample_std"] = float(vectors.std())
        stats["sample_min"] = float(vectors.min())
        stats["sample_max"] = float(vectors.max())
    except Exception as exc:
        stats["sample_vectors_error"] = str(exc)

    return stats


def describe_docstore(payload) -> dict:
    docstore = payload[0]
    id_map = payload[1] if len(payload) > 1 else {}

    doc_items = list(getattr(docstore, "_dict", {}).items())
    metadata_keys = set()
    sample_docs = []

    for doc_id, doc in doc_items[:5]:
        metadata_keys.update(doc.metadata.keys())
        sample_docs.append({
            "doc_id": doc_id,
            "title": doc.metadata.get("title", "<no title>"),
            "preview": doc.page_content[:180].replace("\n", " "),
            "metadata_keys": list(doc.metadata.keys()),
        })

    return {
        "docstore_type": type(docstore).__name__,
        "docstore_size": len(doc_items),
        "id_map_size": len(id_map),
        "metadata_keys": sorted(metadata_keys),
        "sample_docs": sample_docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local FAISS vector store")
    parser.add_argument("--path", default="data/pubmed_faiss_index", help="Path to the FAISS store directory")
    args = parser.parse_args()

    base, index, payload = load_store(args.path)
    index_info = describe_index(index)
    doc_info = describe_docstore(payload)

    print(f"FAISS store path: {base}")
    print("=" * 72)
    print("Index summary")
    print("=" * 72)
    for key, value in index_info.items():
        print(f"{key:18s}: {value}")

    print("\n" + "=" * 72)
    print("Docstore summary")
    print("=" * 72)
    for key, value in doc_info.items():
        if key == "sample_docs":
            print("sample_docs:")
            for doc in value:
                print("  - doc_id=", doc["doc_id"])
                print("    title=", doc["title"])
                print("    preview=", doc["preview"])
                print("    metadata_keys=", ", ".join(doc["metadata_keys"]))
        else:
            print(f"{key:18s}: {value}")

    print("\n" + "=" * 72)
    print("Quick checks")
    print("=" * 72)
    print("vectors_match_docstore:", index_info["ntotal"] == doc_info["docstore_size"])
    print("vector_dimension_matches_doc_dimension:", index_info["dimension"] > 0)
    print("index_files_present:", (base / "index.faiss").exists(), (base / "index.pkl").exists())


if __name__ == "__main__":
    main()
