"""
Wilcoxon signed-rank tests for TFM experiment results.
Compares per-question ROUGE-1 scores between configurations.
Applies Bonferroni correction for multiple comparisons.
"""
import json
import itertools
from pathlib import Path
from scipy import stats
import numpy as np

RESULTS_DIR = Path("data/results")

def load_per_question_scores(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    for r in data["results"]:
        scores[r["id"]] = {
            "rouge1_f1": r["rouge1_f1"],
            "rougeL_f1": r["rougeL_f1"],
            "bertscore_f1": r["bertscore_f1"],
            "type": r["type"],
        }
    return scores

def wilcoxon_test(scores_a, scores_b, metric="rouge1_f1"):
    ids = sorted(scores_a.keys())
    a = np.array([scores_a[qid][metric] for qid in ids])
    b = np.array([scores_b[qid][metric] for qid in ids])
    diff = a - b
    nonzero = np.sum(diff != 0)
    if nonzero < 10:
        return None, None, nonzero
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return stat, p, nonzero

def main():
    # Phase 1
    print("=" * 70)
    print("PHASE 1: Wilcoxon Signed-Rank Tests (Chunking Strategies)")
    print("=" * 70)

    p1_files = {
        "C1": RESULTS_DIR / "phase1_C1_fixed.json",
        "C2": RESULTS_DIR / "phase1_C2_recursive.json",
        "C3": RESULTS_DIR / "phase1_C3_semantic.json",
    }
    p1_scores = {k: load_per_question_scores(v) for k, v in p1_files.items()}

    pairs_p1 = list(itertools.combinations(["C1", "C2", "C3"], 2))
    n_tests_p1 = len(pairs_p1)
    bonferroni_p1 = n_tests_p1

    print(f"\nNumber of pairwise comparisons: {n_tests_p1}")
    print(f"Bonferroni correction factor: {bonferroni_p1}")
    print(f"Adjusted alpha (0.05 / {bonferroni_p1}): {0.05 / bonferroni_p1:.4f}")

    for metric in ["rouge1_f1", "rougeL_f1", "bertscore_f1"]:
        print(f"\n--- Metric: {metric} ---")
        for a, b in pairs_p1:
            stat, p, nz = wilcoxon_test(p1_scores[a], p1_scores[b], metric)
            if p is not None:
                p_adj = min(p * bonferroni_p1, 1.0)
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                print(f"  {a} vs {b}: W={stat:.1f}, p={p:.6f}, p_adj={p_adj:.6f} ({sig}), non-zero diffs={nz}")
            else:
                print(f"  {a} vs {b}: insufficient non-zero differences ({nz})")

    # Phase 2
    print("\n" + "=" * 70)
    print("PHASE 2: Wilcoxon Signed-Rank Tests (Embedding Models)")
    print("=" * 70)

    p2_files = {
        "E1_MiniLM": RESULTS_DIR / "phase2_E1_MiniLM.json",
        "E2_BGE": RESULTS_DIR / "phase2_E2_BGE.json",
        "E3_PubMedBERT": RESULTS_DIR / "phase2_E3_PubMedBERT.json",
        "E4_BioLORD": RESULTS_DIR / "phase2_E4_BioLORD.json",
    }
    p2_scores = {k: load_per_question_scores(v) for k, v in p2_files.items()}

    pairs_p2 = list(itertools.combinations(["E1_MiniLM", "E2_BGE", "E3_PubMedBERT", "E4_BioLORD"], 2))
    n_tests_p2 = len(pairs_p2)
    bonferroni_p2 = n_tests_p2

    print(f"\nNumber of pairwise comparisons: {n_tests_p2}")
    print(f"Bonferroni correction factor: {bonferroni_p2}")
    print(f"Adjusted alpha (0.05 / {bonferroni_p2}): {0.05 / bonferroni_p2:.4f}")

    for metric in ["rouge1_f1", "rougeL_f1", "bertscore_f1"]:
        print(f"\n--- Metric: {metric} ---")
        for a, b in pairs_p2:
            stat, p, nz = wilcoxon_test(p2_scores[a], p2_scores[b], metric)
            if p is not None:
                p_adj = min(p * bonferroni_p2, 1.0)
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                print(f"  {a} vs {b}: W={stat:.1f}, p={p:.6f}, p_adj={p_adj:.6f} ({sig}), non-zero diffs={nz}")
            else:
                print(f"  {a} vs {b}: insufficient non-zero differences ({nz})")

    # Effect sizes (rank-biserial correlation r = 1 - 2W / (n*(n+1)/2))
    print("\n" + "=" * 70)
    print("EFFECT SIZES (rank-biserial correlation)")
    print("=" * 70)

    print("\nPhase 1 (ROUGE-1):")
    for a, b in pairs_p1:
        ids = sorted(p1_scores[a].keys())
        va = np.array([p1_scores[a][qid]["rouge1_f1"] for qid in ids])
        vb = np.array([p1_scores[b][qid]["rouge1_f1"] for qid in ids])
        diff = va - vb
        nz_mask = diff != 0
        n = nz_mask.sum()
        if n >= 10:
            stat, p = stats.wilcoxon(va, vb, alternative="two-sided")
            r = 1 - (2 * stat) / (n * (n + 1) / 2)
            print(f"  {a} vs {b}: r={r:.4f} (n={n})")

    print("\nPhase 2 (ROUGE-1):")
    for a, b in pairs_p2:
        ids = sorted(p2_scores[a].keys())
        va = np.array([p2_scores[a][qid]["rouge1_f1"] for qid in ids])
        vb = np.array([p2_scores[b][qid]["rouge1_f1"] for qid in ids])
        diff = va - vb
        nz_mask = diff != 0
        n = nz_mask.sum()
        if n >= 10:
            stat, p = stats.wilcoxon(va, vb, alternative="two-sided")
            r = 1 - (2 * stat) / (n * (n + 1) / 2)
            print(f"  {a} vs {b}: r={r:.4f} (n={n})")

    # Generalista vs Biomédico aggregate
    print("\n" + "=" * 70)
    print("GENERALISTA vs BIOMEDICO (aggregated)")
    print("=" * 70)

    ids = sorted(p2_scores["E1_MiniLM"].keys())
    for metric in ["rouge1_f1", "rougeL_f1", "bertscore_f1"]:
        gen = np.array([(p2_scores["E1_MiniLM"][qid][metric] + p2_scores["E2_BGE"][qid][metric]) / 2 for qid in ids])
        bio = np.array([(p2_scores["E3_PubMedBERT"][qid][metric] + p2_scores["E4_BioLORD"][qid][metric]) / 2 for qid in ids])
        stat, p = stats.wilcoxon(gen, bio, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {metric}: W={stat:.1f}, p={p:.6f} ({sig})")

    # Save results as JSON for reference
    results = {"phase1": {}, "phase2": {}, "gen_vs_bio": {}}
    for metric in ["rouge1_f1", "rougeL_f1", "bertscore_f1"]:
        results["phase1"][metric] = {}
        for a, b in pairs_p1:
            stat, p, nz = wilcoxon_test(p1_scores[a], p1_scores[b], metric)
            if p is not None:
                results["phase1"][metric][f"{a}_vs_{b}"] = {
                    "W": float(stat), "p": float(p),
                    "p_adjusted": float(min(p * bonferroni_p1, 1.0)),
                    "n_nonzero": int(nz),
                }

        results["phase2"][metric] = {}
        for a, b in pairs_p2:
            stat, p, nz = wilcoxon_test(p2_scores[a], p2_scores[b], metric)
            if p is not None:
                results["phase2"][metric][f"{a}_vs_{b}"] = {
                    "W": float(stat), "p": float(p),
                    "p_adjusted": float(min(p * bonferroni_p2, 1.0)),
                    "n_nonzero": int(nz),
                }

    with open(RESULTS_DIR / "wilcoxon_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'wilcoxon_results.json'}")

if __name__ == "__main__":
    main()
