"""
Evaluate SBERT on the new ULO (Unit Learning Outcome) dataset and measure label quality.

Inputs (all in ulo_dataset/):
    ulo_dataset_labelled.csv    500 pairs with the final LLM judge's score (0-5) and the SBERT cosine
                                (made by analyse_labels.py from llm_labels.csv)
    ulos_clean.csv              205 cleaned outcomes (corpus used to fit TF-IDF)
    llm_labels_claude.csv       second LLM judge, all 500 pairs (judge-vs-judge agreement)
    llm_labels_round1.csv       first-round labels for pairs 1-150, October 2026 (consistency check)
    claude_seen_round1_ids.txt  pairs whose first-round labels the second judge had seen
    my_ratings.csv              20 pairs rated by the author (first round)
    my_ratings_blind.csv        50 pairs rated blind by the author (made by make_blind_sample.py)

Run from the repository root:
    python ulo_dataset/evaluate_ulo.py            # statistics only: any CPU, about 2 minutes
    python ulo_dataset/evaluate_ulo.py --models   # also re-encodes the pairs with the paper's
                                                  # baselines (downloads about 2 GB of models)

Outputs:
    results/ulo_evaluation.txt   human-readable report (all numbers used in the final report)
    results/ulo_scores.csv       per-pair scores of every method
    results/ulo_baselines.csv    only with --models: Table-1-style comparison on the ULO data
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent          # .../ulo_dataset
ROOT = HERE.parent                              # repository root
RESULTS = ROOT / "results"
SEED = 42
N_BOOT = 10_000

lines = []


def say(text=""):
    print(text)
    lines.append(text)


# ---------------------------------------------------------------- helpers
def spearman100(x, y):
    if np.ptp(np.asarray(x, float)) == 0 or np.ptp(np.asarray(y, float)) == 0:
        return float("nan")                          # undefined for a constant input
    return spearmanr(x, y).correlation * 100


def bootstrap_ci(x, y, stat=spearman100, n_boot=N_BOOT, seed=SEED):
    """95% percentile bootstrap interval, resampling pairs with replacement."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:     # constant resample: undefined
            continue
        vals.append(stat(x[i], y[i]))
    return np.percentile(vals, [2.5, 97.5])


def fmt_ci(x, y):
    lo, hi = bootstrap_ci(x, y)
    return f"{spearman100(x, y):6.2f}  [95% CI {lo:5.1f}, {hi:5.1f}]"


def paired_diff_ci(x, y1, y2, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap interval for spearman(x, y1) - spearman(x, y2), both on the same resampled pairs."""
    x, y1, y2 = (np.asarray(v, float) for v in (x, y1, y2))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if min(np.ptp(x[i]), np.ptp(y1[i]), np.ptp(y2[i])) == 0:
            continue
        vals.append(spearman100(x[i], y1[i]) - spearman100(x[i], y2[i]))
    return np.percentile(vals, [2.5, 97.5])


def lead_verb(text):
    """First content word of an outcome, skipping adverbs such as 'Critically'."""
    words = re.findall(r"[a-z]+", str(text).lower())
    for w in words[:3]:
        if not w.endswith("ly"):
            return w
    return words[0] if words else ""


def content_tokens(text):
    return {w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in ENGLISH_STOP_WORDS}


def krippendorff_ordinal(a, b):
    try:
        import krippendorff
    except ImportError:
        return float("nan")
    return krippendorff.alpha(reliability_data=np.vstack([a, b]).astype(float),
                              level_of_measurement="ordinal")


def agreement_block(name, human, llm, a_name="human", b_name="LLM"):
    human, llm = np.asarray(human, int), np.asarray(llm, int)
    lo, hi = bootstrap_ci(human, llm)
    say(f"  {name}: n = {len(human)} pairs")
    say(f"    Spearman rho            = {spearman100(human, llm) / 100:.2f}  "
        f"[95% CI {lo / 100:.2f}, {hi / 100:.2f}]")
    say(f"    Pearson r               = {pearsonr(human, llm)[0]:.2f}")
    say(f"    Quadratic-weighted kappa= {cohen_kappa_score(human, llm, weights='quadratic', labels=list(range(6))):.2f}")
    say(f"    Krippendorff alpha (ord)= {krippendorff_ordinal(human, llm):.2f}")
    say(f"    Exact agreement         = {np.mean(human == llm) * 100:.0f}%")
    say(f"    Within +/-1 point       = {np.mean(np.abs(human - llm) <= 1) * 100:.0f}%")
    say(f"    Mean |difference|       = {np.mean(np.abs(human - llm)):.2f}")
    say(f"    Mean ({a_name} - {b_name}){' ' * max(0, 15 - len(a_name) - len(b_name))}= {np.mean(human - llm):+.2f}   "
        f"({a_name} mean {human.mean():.2f}, {b_name} mean {llm.mean():.2f})")


# ---------------------------------------------------------------- optional model baselines
def encode_with_models(texts):
    """Re-encode texts with the paper's Table 1 systems. Returns {name: embeddings}."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    def st(name):
        m = SentenceTransformer(name, device=device)
        return m.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

    @torch.no_grad()
    def raw_bert(pooling):
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        embs = []
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(device)
            h = bert(**b).last_hidden_state
            if pooling == "cls":
                e = h[:, 0]
            else:                                   # mean over real tokens only
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.vstack(embs)

    runs = [
        ("Avg. GloVe embeddings (6B)", lambda: st("sentence-transformers/average_word_embeddings_glove.6B.300d")),
        ("Avg. BERT embeddings", lambda: raw_bert("mean")),
        ("BERT CLS-vector", lambda: raw_bert("cls")),
        ("SBERT-NLI-base", lambda: st("sentence-transformers/bert-base-nli-mean-tokens")),
        ("SBERT-NLI-large", lambda: st("sentence-transformers/bert-large-nli-mean-tokens")),
    ]
    for name, fn in runs:
        print(f"  encoding with {name} ...")
        try:
            out[name] = fn()
        except Exception as e:                      # record the failure, keep going
            print(f"    FAILED: {type(e).__name__}: {e}")
    return out


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", action="store_true",
                        help="also re-encode with GloVe / BERT / SBERT baselines (needs internet)")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    d = pd.read_csv(HERE / "ulo_dataset_labelled.csv")
    if "sbert_cosine" not in d.columns or d["sbert_cosine"].isna().all():
        raise SystemExit("ulo_dataset_labelled.csv has no SBERT cosines yet. Run this first:\n"
                         "    cd ulo_dataset && python analyse_labels.py && cd ..\n"
                         "(it needs sentence-transformers installed: pip install -e .)")
    corpus = pd.read_csv(HERE / "ulos_clean.csv")["ulo_text"].tolist()
    d = d.dropna(subset=["llm_score", "sbert_cosine"]).reset_index(drop=True)
    d["llm_score"] = d["llm_score"].astype(int)

    # ---------- 1. the labels themselves
    say("=" * 72)
    say("1. LLM LABELS")
    say(f"  pairs labelled: {len(d)}")
    dist = d["llm_score"].value_counts().reindex(range(6), fill_value=0)
    say("  score distribution: " + ", ".join(f"{k}: {v}" for k, v in dist.items()))
    if "sample" in d.columns:
        say("  by sampling stratum (n, mean LLM score):")
        for t, g in d.groupby("sample"):
            say(f"    {t:<22} n={len(g):3d}  LLM={g['llm_score'].mean():.2f}")
    say("  by pair type (n, mean LLM score, mean SBERT cosine):")
    for t in ["same_unit", "same_discipline", "cross_discipline"]:
        g = d[d["pair_type"] == t]
        say(f"    {t:<17} n={len(g):3d}  LLM={g['llm_score'].mean():.2f}  cosine={g['sbert_cosine'].mean():.3f}")

    # other LLM labels for the same pairs, matched by pair_id (used in sections 2 and 2b)
    def load_labels(fname, col):
        """Read a pasted label file (tolerates repeated headers); return pair_id and one score column."""
        path = HERE / fname
        if not path.exists():
            return None
        lab = pd.read_csv(path, dtype=str, on_bad_lines="skip")
        lab = lab[lab["pair_id"].str.strip() != "pair_id"]           # pasted batches repeat the header
        lab["pair_id"] = pd.to_numeric(lab["pair_id"], errors="coerce")
        lab[col] = pd.to_numeric(lab["llm_score"], errors="coerce")
        lab = lab.dropna(subset=["pair_id", col]).drop_duplicates("pair_id", keep="last")
        return lab[["pair_id", col]].astype({"pair_id": int})

    second = load_labels("llm_labels_claude.csv", "second_judge")     # a different LLM, all pairs
    october = load_labels("llm_labels_round1.csv", "october")         # first round, pairs 1-150
    seen_path = HERE / "claude_seen_round1_ids.txt"
    seen = ({int(line) for line in seen_path.read_text().splitlines() if line.strip().isdigit()}
            if seen_path.exists() else set())

    # ---------- 2. human vs LLM agreement
    say("\n" + "=" * 72)
    say("2. HUMAN vs LLM AGREEMENT (measure of label quality; LLM = final judge)")
    for fname, label in [("my_ratings.csv", "Round 1 (my_ratings.csv)"),
                         ("my_ratings_blind.csv", "Round 2 blind (my_ratings_blind.csv)")]:
        path = HERE / fname
        if not path.exists():
            continue
        h = pd.read_csv(path)
        h["my_score"] = pd.to_numeric(h["my_score"], errors="coerce")
        # match on the two texts, not pair_id: pair ids change whenever pairs.csv is regenerated
        h = h.dropna(subset=["my_score"]).drop(columns="pair_id").merge(
            d[["pair_id", "text_a", "text_b", "llm_score", "sbert_cosine", "pair_type"]], on=["text_a", "text_b"])
        if len(h) < 5:
            say(f"  {label}: fewer than 5 ratings, skipped")
            continue
        agreement_block(label, h["my_score"], h["llm_score"])
        say(f"    pair types rated        = {h['pair_type'].value_counts().to_dict()}")
        say(f"    SBERT vs human (same pairs): Spearman = {fmt_ci(h['sbert_cosine'], h['my_score'])}")
        say(f"    SBERT vs LLM   (same pairs): Spearman = {fmt_ci(h['sbert_cosine'], h['llm_score'])}")
        gap = spearman100(h["sbert_cosine"], h["llm_score"]) - spearman100(h["sbert_cosine"], h["my_score"])
        lo, hi = paired_diff_ci(h["sbert_cosine"], h["llm_score"], h["my_score"])
        say(f"    difference (LLM - human)   : {gap:+6.2f}  [95% CI {lo:5.1f}, {hi:5.1f}]  (paired bootstrap)")
        for other, col, name in [(second, "second_judge", "second judge"), (october, "october", "October labels")]:
            if other is None:
                continue
            m = h.merge(other, on="pair_id")
            if len(m) >= 5:
                say(f"    vs {name} ({len(m)} of these pairs): human Spearman = "
                    f"{spearman100(m['my_score'], m[col]) / 100:.2f};  SBERT Spearman = "
                    f"{spearman100(m['sbert_cosine'], m[col]):.2f}")

    # ---------- 2b. agreement between LLM judges on the shared pairs
    say("\n" + "=" * 72)
    say("2b. AGREEMENT BETWEEN LLM JUDGES (pairs labelled by both)")
    if second is not None:
        both = d[["pair_id", "llm_score", "sbert_cosine"]].merge(second, on="pair_id")
        if len(both) >= 5:
            agreement_block("Final judge vs second judge (llm_labels_claude.csv)",
                            both["llm_score"], both["second_judge"], "final", "second")
            say(f"    SBERT vs second judge (same pairs): Spearman = {fmt_ci(both['sbert_cosine'], both['second_judge'])}")
            say(f"    SBERT vs final judge  (same pairs): Spearman = {fmt_ci(both['sbert_cosine'], both['llm_score'])}")
            if seen:
                unseen = both[~both["pair_id"].isin(seen)]
                agreement_block(f"Same, excluding the {len(seen)} pairs whose first-round labels the second judge had seen",
                                unseen["llm_score"], unseen["second_judge"], "final", "second")
    if october is not None:
        rnd = d[["pair_id", "llm_score", "sbert_cosine"]].merge(october, on="pair_id")
        if len(rnd) >= 5:
            agreement_block("Final judge vs first-round labels (llm_labels_round1.csv, October)",
                            rnd["llm_score"], rnd["october"], "final", "October")
            say(f"    SBERT vs first-round labels (same pairs): Spearman = {fmt_ci(rnd['sbert_cosine'], rnd['october'])}")
            say(f"    SBERT vs final judge        (same pairs): Spearman = {fmt_ci(rnd['sbert_cosine'], rnd['llm_score'])}")
            if second is not None:
                rnd = rnd.merge(second, on="pair_id")
                agreement_block("Second judge vs first-round labels", rnd["second_judge"], rnd["october"], "second", "October")
                if seen:
                    unseen = rnd[~rnd["pair_id"].isin(seen)]
                    agreement_block(f"Same, excluding the {len(seen)} pairs the second judge had seen",
                                    unseen["second_judge"], unseen["october"], "second", "October")

    # ---------- 3. SBERT and lexical baselines vs LLM labels
    say("\n" + "=" * 72)
    say(f"3. SIMILARITY METHODS vs LLM LABELS  (Spearman x 100, {len(d)} pairs)")
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english").fit(corpus)
    A, B = tfidf.transform(d["text_a"]), tfidf.transform(d["text_b"])
    d["tfidf_cosine"] = np.asarray(A.multiply(B).sum(1)).ravel()
    d["jaccard"] = [len(content_tokens(a) & content_tokens(b)) / max(1, len(content_tokens(a) | content_tokens(b)))
                    for a, b in zip(d["text_a"], d["text_b"])]
    say(f"  SBERT-NLI-base (cosine)       {fmt_ci(d['sbert_cosine'], d['llm_score'])}")
    say(f"  TF-IDF cosine (lexical)       {fmt_ci(d['tfidf_cosine'], d['llm_score'])}")
    say(f"  Jaccard word overlap          {fmt_ci(d['jaccard'], d['llm_score'])}")
    say(f"  SBERT Pearson r x 100         {pearsonr(d['sbert_cosine'], d['llm_score'])[0] * 100:6.2f}")
    say("  Within each pair type (SBERT vs LLM, Spearman x 100):")
    for t in ["same_unit", "same_discipline", "cross_discipline"]:
        g = d[d["pair_type"] == t]
        say(f"    {t:<17} SBERT {spearman100(g['sbert_cosine'], g['llm_score']):6.2f}   "
            f"TF-IDF {spearman100(g['tfidf_cosine'], g['llm_score']):6.2f}")
    if "sample" in d.columns:
        say("  Within each sampling stratum (SBERT / TF-IDF vs LLM, Spearman x 100):")
        for t, g in d.groupby("sample"):
            if g["llm_score"].nunique() > 1:
                say(f"    {t:<22} n={len(g):3d}  SBERT {spearman100(g['sbert_cosine'], g['llm_score']):6.2f}   "
                    f"TF-IDF {spearman100(g['tfidf_cosine'], g['llm_score']):6.2f}")
        v1 = d[d["sample"] == "v1_random_by_type"]
        if len(v1) > 10:
            say(f"  Random v1 pairs only (no overlap-based selection): SBERT {fmt_ci(v1['sbert_cosine'], v1['llm_score'])}")
            say(f"                                                     TF-IDF {fmt_ci(v1['tfidf_cosine'], v1['llm_score'])}")
    type_rank = d["pair_type"].map({"cross_discipline": 0, "same_discipline": 1, "same_unit": 2})
    say(f"  Pair type alone (0/1/2) vs LLM labels: {spearman100(type_rank, d['llm_score']):.2f}")

    # ---------- 4. does a shared leading verb inflate SBERT similarity?
    say("\n" + "=" * 72)
    say("4. TEMPLATE EFFECT: shared leading verb (e.g. 'Apply ...' / 'Apply ...')")
    d["verb_a"], d["verb_b"] = d["text_a"].map(lead_verb), d["text_b"].map(lead_verb)
    d["same_verb"] = d["verb_a"] == d["verb_b"]
    # residual = cosine minus the mean cosine of pairs with the same LLM score
    d["resid"] = d["sbert_cosine"] - d.groupby("llm_score")["sbert_cosine"].transform("mean")
    obs = d.loc[d["same_verb"], "resid"].mean() - d.loc[~d["same_verb"], "resid"].mean()
    rng = np.random.default_rng(SEED)
    perm = []
    for _ in range(N_BOOT):                          # shuffle the flag within each score level
        flag = d.groupby("llm_score")["same_verb"].transform(lambda s: rng.permutation(s.values))
        perm.append(d.loc[flag, "resid"].mean() - d.loc[~flag, "resid"].mean())
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1)
    say(f"  pairs sharing the leading verb: {int(d['same_verb'].sum())} of {len(d)}")
    say(f"  extra SBERT cosine for a shared verb, at equal LLM score: {obs:+.3f}  "
        f"(stratified permutation p = {p:.3f})")
    llm_gap = (d.loc[d["same_verb"], "llm_score"].mean() - d.loc[~d["same_verb"], "llm_score"].mean())
    say(f"  (for reference: mean LLM score, shared minus not shared = {llm_gap:+.2f})")

    # ---------- 4b. boilerplate capability tags, e.g. "(Scientist and Scholar)"
    tag = re.compile(r"\((?:Capability \d+: )?(?:Scientist|Professional|Practitioner|Citizen|Exercise Science|Engaged)[^)]*\)\s*$")
    d["tag_a"] = d["text_a"].str.contains(tag)
    d["tag_b"] = d["text_b"].str.contains(tag)
    both_tagged = d["tag_a"] & d["tag_b"]
    if both_tagged.any() and (~both_tagged).any():
        obs = d.loc[both_tagged, "resid"].mean() - d.loc[~both_tagged, "resid"].mean()
        perm = []
        for _ in range(N_BOOT):
            flag = d.groupby("llm_score")["tag_a"].transform(lambda s: rng.permutation((d.loc[s.index, "tag_a"] & d.loc[s.index, "tag_b"]).values))
            perm.append(d.loc[flag, "resid"].mean() - d.loc[~flag, "resid"].mean())
        p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1)
        n_tagged = len(set(d.loc[d["tag_a"], "text_a"]) | set(d.loc[d["tag_b"], "text_b"]))
        say("\n  Capability tags such as '(Scientist and Scholar)' at the end of an outcome:")
        say(f"  distinct tagged outcomes: {n_tagged}; pairs where both outcomes are tagged: {int(both_tagged.sum())}")
        say(f"  extra SBERT cosine when both are tagged, at equal LLM score: {obs:+.3f}  (stratified permutation p = {p:.3f})")

    # ---------- 5. largest disagreements (for error analysis)
    say("\n" + "=" * 72)
    say("5. LARGEST SBERT-LLM DISAGREEMENTS (percentile ranks)")
    d["rank_gap"] = rankdata(d["sbert_cosine"]) / len(d) - rankdata(d["llm_score"]) / len(d)
    for title, sub in [("SBERT much higher than LLM", d.nlargest(4, "rank_gap")),
                       ("LLM much higher than SBERT", d.nsmallest(4, "rank_gap"))]:
        say(f"  {title}:")
        for r in sub.itertuples():
            say(f"    [pair {r.pair_id}] LLM={r.llm_score} cos={r.sbert_cosine:.2f} ({r.pair_type})")
            say(f"      A: {r.text_a}")
            say(f"      B: {r.text_b}")
            say(f"      LLM reason: {r.llm_reason}")

    # ---------- 6. optional: the paper's Table 1 systems on the ULO data
    if args.models:
        say("\n" + "=" * 72)
        say("6. PAPER BASELINES ON ULO DATA (Spearman x 100 vs LLM labels)")
        texts = list(dict.fromkeys(d["text_a"].tolist() + d["text_b"].tolist()))
        pos = {t: i for i, t in enumerate(texts)}
        ia, ib = d["text_a"].map(pos).values, d["text_b"].map(pos).values
        rows = []
        for name, emb in encode_with_models(texts).items():
            cos = (emb[ia] * emb[ib]).sum(1)
            d[f"cos_{name}"] = cos
            lo, hi = bootstrap_ci(cos, d["llm_score"])
            rows.append([name, round(spearman100(cos, d["llm_score"]), 2), round(lo, 1), round(hi, 1)])
            say(f"  {name:<28} {fmt_ci(cos, d['llm_score'])}")
        rows.append(["TF-IDF cosine", round(spearman100(d['tfidf_cosine'], d['llm_score']), 2), "", ""])
        pd.DataFrame(rows, columns=["model", "spearman_x100", "ci_low", "ci_high"]).to_csv(
            RESULTS / "ulo_baselines.csv", index=False)

    cols = [c for c in d.columns if c not in {"resid", "rank_gap"}]
    d[cols].to_csv(RESULTS / "ulo_scores.csv", index=False)
    (RESULTS / "ulo_evaluation.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS / 'ulo_evaluation.txt'} and {RESULTS / 'ulo_scores.csv'}")


if __name__ == "__main__":
    main()
