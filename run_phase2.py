"""
Phase 2 Experiments: Embedding model comparison.

Runs 4 embedding configs (MiniLM, BGE, PubMedBERT, BioLORD) × 150 QA pairs
with constant chunking (C2 recursive - C_best from Phase 1) and Gemma 3 12B.

Usage:
    python run_phase2.py
"""
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_clinico.loader import load_all_guidelines
from rag_clinico.chunking import recursive_chunker
from rag_clinico.embedding.models import get_embedding_model
from rag_clinico.pipeline import RAGPipeline
from rag_clinico.evaluation.metrics import compute_rouge, compute_bertscore

GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"
QA_FILE = PROJECT_ROOT / "data" / "qa_dataset" / "qa_pairs.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKING_STRATEGY = "C2_recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

CONFIGS = {
    "E1_MiniLM": {
        "name": "all-MiniLM-L6-v2",
        "type": "Generalista",
        "model_key": "all-MiniLM-L6-v2",
    },
    "E2_BGE": {
        "name": "bge-base-en-v1.5",
        "type": "Generalista",
        "model_key": "bge-base-en-v1.5",
    },
    "E3_PubMedBERT": {
        "name": "PubMedBERT",
        "type": "Biomédico",
        "model_key": "PubMedBERT",
    },
    "E4_BioLORD": {
        "name": "BioLORD-2023",
        "type": "Biomédico",
        "model_key": "BioLORD-2023",
    },
}


def load_qa_dataset():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_config(config_id, config, chunks, qa_pairs):
    print(f"\n{'='*60}", flush=True)
    print(f"  Config {config_id}: {config['name']} ({config['type']})", flush=True)
    print(f"{'='*60}", flush=True)

    # Step 1: Load embedding model
    t0 = time.time()
    print(f"  [1/5] Loading embedding model: {config['model_key']}...", flush=True)
    embedding_model = get_embedding_model(config["model_key"])
    load_time = time.time() - t0
    print(f"        Loaded in {load_time:.1f}s", flush=True)

    # Step 2: Build pipeline
    t0 = time.time()
    print(f"  [2/5] Building vector store + pipeline...", flush=True)
    pipeline = RAGPipeline(
        chunks=chunks,
        embedding_model=embedding_model,
        k=4,
        collection_name=f"phase2_{config_id}",
    )
    index_time = time.time() - t0
    print(f"        Indexed {len(chunks)} chunks in {index_time:.1f}s", flush=True)

    # Step 3: Run QA pairs
    print(f"  [3/5] Running {len(qa_pairs)} queries...", flush=True)
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
                print(f"        {i+1}/{len(qa_pairs)} done (avg ROUGE-1: {avg_r1:.3f}, last {query_time:.1f}s)", flush=True)

        except Exception as e:
            errors += 1
            print(f"        ERROR on Q{i+1} ({qa['id']}): {e}", flush=True)
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
    print(f"  [4/5] Computing BERTScore (batch)...", flush=True)
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
    print(f"  [5/5] Aggregating metrics...", flush=True)
    valid = [r for r in results if "error" not in r]
    summary = {
        "config_id": config_id,
        "config_name": config["name"],
        "config_type": config["type"],
        "n_chunks": len(chunks),
        "n_questions": len(qa_pairs),
        "n_valid": len(valid),
        "n_errors": errors,
        "model_load_time_s": load_time,
        "index_time_s": index_time,
        "avg_query_time_s": sum(r["query_time"] for r in valid) / len(valid) if valid else 0,
        "metrics": {
            "rouge1_f1": sum(r["rouge1_f1"] for r in valid) / len(valid) if valid else 0,
            "rougeL_f1": sum(r["rougeL_f1"] for r in valid) / len(valid) if valid else 0,
            "bertscore_f1": sum(r["bertscore_f1"] for r in valid) / len(valid) if valid else 0,
        },
    }

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
          f"BERTScore={summary['metrics']['bertscore_f1']:.4f}", flush=True)

    return results, summary


def main():
    print(f"Phase 2 Experiments - {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print(f"LLM: gemma3:12b on 192.168.1.22 (RTX 5060)", flush=True)
    print(f"Chunking: {CHUNKING_STRATEGY} (C_best from Phase 1)", flush=True)

    # Load documents
    print("\n[SETUP] Loading ESC guidelines...", flush=True)
    documents = load_all_guidelines(GUIDELINES_DIR)
    print(f"  Loaded {len(documents)} documents", flush=True)

    # Chunk once (same for all configs - C2 recursive)
    print(f"  Chunking with recursive strategy (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...", flush=True)
    chunks = recursive_chunker(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks", flush=True)

    # Load QA dataset
    qa_pairs = load_qa_dataset()
    print(f"  QA dataset: {len(qa_pairs)} pairs", flush=True)

    # Run each config
    all_summaries = []
    total_start = time.time()

    for config_id, config in CONFIGS.items():
        results, summary = run_config(config_id, config, chunks, qa_pairs)
        all_summaries.append(summary)

        out_file = RESULTS_DIR / f"phase2_{config_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file}", flush=True)

    total_time = time.time() - total_start

    # Final comparison table
    print(f"\n{'='*75}", flush=True)
    print(f"  PHASE 2 RESULTS SUMMARY", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"{'Config':<20} {'Type':<12} {'ROUGE-1':>9} {'ROUGE-L':>9} {'BERTScore':>10} {'Time':>8}", flush=True)
    print(f"{'-'*20} {'-'*12} {'-'*9} {'-'*9} {'-'*10} {'-'*8}", flush=True)
    for s in all_summaries:
        m = s["metrics"]
        t = s["model_load_time_s"] + s["index_time_s"] + s["avg_query_time_s"] * s["n_valid"]
        print(f"{s['config_name']:<20} {s['config_type']:<12} {m['rouge1_f1']:>9.4f} {m['rougeL_f1']:>9.4f} {m['bertscore_f1']:>10.4f} {t:>7.0f}s", flush=True)

    # Identify E_best
    best = max(all_summaries, key=lambda s: s["metrics"]["bertscore_f1"])
    print(f"\nE_best = {best['config_name']} (BERTScore: {best['metrics']['bertscore_f1']:.4f})", flush=True)
    print(f"Optimal config: C_best (Recursivo) + E_best ({best['config_name']})", flush=True)
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes", flush=True)

    # Save combined summary
    summary_file = RESULTS_DIR / "phase2_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Phase 2 - Embedding comparison",
            "date": datetime.now().isoformat(),
            "chunking": CHUNKING_STRATEGY,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "llm": "gemma3:12b",
            "llm_server": "192.168.1.22:11434",
            "total_time_minutes": total_time / 60,
            "e_best": best["config_name"],
            "configs": all_summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
