"""
Phase 1 Experiments: Chunking strategy comparison.

Runs 3 chunking configs (fixed, recursive, semantic) × 150 QA pairs
with constant embedding (all-MiniLM-L6-v2) and Gemma 3 12B on remote GPU.

Usage:
    python run_phase1.py
"""
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_clinico.loader import load_all_guidelines
from rag_clinico.chunking import fixed_size_chunker, recursive_chunker, semantic_chunker
from rag_clinico.embedding.models import get_embedding_model
from rag_clinico.pipeline import RAGPipeline
from rag_clinico.evaluation.metrics import compute_rouge, compute_bertscore

GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"
QA_FILE = PROJECT_ROOT / "data" / "qa_dataset" / "qa_pairs.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_NAME = "all-MiniLM-L6-v2"

CONFIGS = {
    "C1_fixed": {
        "name": "Tamaño fijo",
        "chunker": lambda docs: fixed_size_chunker(docs, chunk_size=512, chunk_overlap=50),
    },
    "C2_recursive": {
        "name": "Recursivo",
        "chunker": lambda docs: recursive_chunker(docs, chunk_size=512, chunk_overlap=50),
    },
    "C3_semantic": {
        "name": "Semántico",
        "chunker": lambda docs: semantic_chunker(docs, embedding_model_name=EMBEDDING_NAME),
    },
}


def load_qa_dataset():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_config(config_id, config, documents, embedding_model, qa_pairs):
    print(f"\n{'='*60}")
    print(f"  Config {config_id}: {config['name']}")
    print(f"{'='*60}")

    # Step 1: Chunk
    t0 = time.time()
    print(f"  [1/5] Chunking...")
    chunks = config["chunker"](documents)
    chunk_time = time.time() - t0
    print(f"        {len(chunks)} chunks in {chunk_time:.1f}s")

    # Step 2: Build pipeline
    t0 = time.time()
    print(f"  [2/5] Building vector store + pipeline...")
    pipeline = RAGPipeline(
        chunks=chunks,
        embedding_model=embedding_model,
        k=4,
        collection_name=f"phase1_{config_id}",
    )
    index_time = time.time() - t0
    print(f"        Indexed in {index_time:.1f}s")

    # Step 3: Run QA pairs
    print(f"  [3/5] Running {len(qa_pairs)} queries...")
    results = []
    errors = 0
    for i, qa in enumerate(qa_pairs):
        try:
            t0 = time.time()
            response = pipeline.query_with_sources(qa["question"])
            query_time = time.time() - t0

            rouge = compute_rouge(response["answer"], qa["answer"])

            results.append({
                "id": qa["id"],
                "type": qa.get("type", "unknown"),
                "guideline": qa.get("guideline", "unknown"),
                "question": qa["question"],
                "reference": qa["answer"],
                "generated": response["answer"],
                "sources": [doc.page_content[:200] for doc in response["source_documents"]],
                "rouge1_f1": rouge["rouge1_f1"],
                "rougeL_f1": rouge["rougeL_f1"],
                "query_time": query_time,
            })

            if (i + 1) % 10 == 0:
                avg_r1 = sum(r["rouge1_f1"] for r in results) / len(results)
                print(f"        {i+1}/{len(qa_pairs)} done (avg ROUGE-1: {avg_r1:.3f}, last {query_time:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"        ERROR on Q{i+1} ({qa['id']}): {e}")
            results.append({
                "id": qa["id"],
                "type": qa.get("type", "unknown"),
                "question": qa["question"],
                "reference": qa["answer"],
                "generated": f"ERROR: {str(e)}",
                "rouge1_f1": 0.0,
                "rougeL_f1": 0.0,
                "query_time": 0.0,
                "error": str(e),
            })

    # Step 4: BERTScore (batch)
    print(f"  [4/5] Computing BERTScore (batch)...")
    predictions = [r["generated"] for r in results if "error" not in r]
    references = [r["reference"] for r in results if "error" not in r]
    if predictions:
        bert_scores = compute_bertscore(predictions, references)
        j = 0
        for r in results:
            if "error" not in r:
                r["bertscore_f1"] = bert_scores[j]
                j += 1
            else:
                r["bertscore_f1"] = 0.0

    # Step 5: Aggregate
    print(f"  [5/5] Aggregating metrics...")
    valid = [r for r in results if "error" not in r]
    summary = {
        "config_id": config_id,
        "config_name": config["name"],
        "n_chunks": len(chunks),
        "n_questions": len(qa_pairs),
        "n_valid": len(valid),
        "n_errors": errors,
        "chunk_time_s": chunk_time,
        "index_time_s": index_time,
        "avg_query_time_s": sum(r["query_time"] for r in valid) / len(valid) if valid else 0,
        "metrics": {
            "rouge1_f1": sum(r["rouge1_f1"] for r in valid) / len(valid) if valid else 0,
            "rougeL_f1": sum(r["rougeL_f1"] for r in valid) / len(valid) if valid else 0,
            "bertscore_f1": sum(r["bertscore_f1"] for r in valid) / len(valid) if valid else 0,
        },
    }

    # Breakdown by question type
    for qtype in ["factual", "explanatory", "comparative"]:
        typed = [r for r in valid if r.get("type") == qtype]
        if typed:
            summary[f"metrics_{qtype}"] = {
                "n": len(typed),
                "rouge1_f1": sum(r["rouge1_f1"] for r in typed) / len(typed),
                "rougeL_f1": sum(r["rougeL_f1"] for r in typed) / len(typed),
                "bertscore_f1": sum(r["bertscore_f1"] for r in typed) / len(typed),
            }

    print(f"\n  Results: ROUGE-1={summary['metrics']['rouge1_f1']:.4f}  "
          f"ROUGE-L={summary['metrics']['rougeL_f1']:.4f}  "
          f"BERTScore={summary['metrics']['bertscore_f1']:.4f}")

    return results, summary


def main():
    print(f"Phase 1 Experiments - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"LLM: gemma3:12b on 192.168.1.22 (RTX 5060)")
    print(f"Embedding: {EMBEDDING_NAME} (constant)")

    # Load documents
    print("\n[SETUP] Loading ESC guidelines...")
    documents = load_all_guidelines(GUIDELINES_DIR)
    print(f"  Loaded {len(documents)} documents")

    # Load QA dataset
    qa_pairs = load_qa_dataset()
    print(f"  QA dataset: {len(qa_pairs)} pairs")

    # Load embedding model (once, reuse for all configs)
    print(f"  Loading embedding model: {EMBEDDING_NAME}...")
    embedding_model = get_embedding_model(EMBEDDING_NAME)
    print(f"  Ready.")

    # Run each config
    all_summaries = []
    all_results = {}
    total_start = time.time()

    for config_id, config in CONFIGS.items():
        results, summary = run_config(config_id, config, documents, embedding_model, qa_pairs)
        all_results[config_id] = results
        all_summaries.append(summary)

        # Save per-config results
        out_file = RESULTS_DIR / f"phase1_{config_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file}")

    total_time = time.time() - total_start

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  PHASE 1 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Chunks':>7} {'ROUGE-1':>9} {'ROUGE-L':>9} {'BERTScore':>10} {'Time':>8}")
    print(f"{'-'*15} {'-'*7} {'-'*9} {'-'*9} {'-'*10} {'-'*8}")
    for s in all_summaries:
        m = s["metrics"]
        t = s["chunk_time_s"] + s["index_time_s"] + s["avg_query_time_s"] * s["n_valid"]
        print(f"{s['config_name']:<15} {s['n_chunks']:>7} {m['rouge1_f1']:>9.4f} {m['rougeL_f1']:>9.4f} {m['bertscore_f1']:>10.4f} {t:>7.0f}s")
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")

    # Save combined summary
    summary_file = RESULTS_DIR / "phase1_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Phase 1 - Chunking comparison",
            "date": datetime.now().isoformat(),
            "embedding": EMBEDDING_NAME,
            "llm": "gemma3:12b",
            "llm_server": "192.168.1.22:11434",
            "total_time_minutes": total_time / 60,
            "configs": all_summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
