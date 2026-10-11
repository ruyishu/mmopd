import glob, re
import pandas as pd
def load_dev(limit_questions=None):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/eval/MMBench/en/dev-*.parquet"))], ignore_index=True)
    if limit_questions:
        keep = sorted((df["index"] % 1000000).unique())[:limit_questions]
        df = df[(df["index"] % 1000000).isin(keep)]
    return df.reset_index(drop=True)
def isval(x): return isinstance(x, str) and x.strip() and x.strip().lower() != "nan"
def build_prompt(r):
    opts = [(c, r[c]) for c in "ABCD" if c in r and isval(r[c])]
    p = (f"Hint: {r['hint']}\n" if "hint" in r and isval(r["hint"]) else "") + f"Question: {r['question']}\nOptions:\n" + "\n".join(f"{c}. {v}" for c, v in opts)
    return p + "\nAnswer with the option's letter from the given choices directly.", opts
def extract(t, opts):
    s = t.strip()
    m = re.match(r"^\(?([A-D])[\).:\s]|^([A-D])$", s)
    if m: return (m.group(1) or m.group(2)), "letter_start"
    m = re.search(r"\b(?:answer|option)\s*(?:is|:)?\s*\(?([A-D])\b", s, re.I)
    if m: return m.group(1).upper(), "letter_phrase"
    hits = [c for c, v in opts if str(v).strip().lower() in s.lower()]
    if len(hits) == 1: return hits[0], "option_text"
    return None, "fail"
def score(recs):
    g = {}
    for r in recs: g.setdefault(r["index"] % 1000000, []).append(r["correct"])
    orig = [r for r in recs if r["index"] < 1000000]
    how = {}
    for r in recs: how[r["method"]] = how.get(r["method"], 0) + 1
    return {"bench": "mmbench_en_dev", "n_rows": len(recs), "n_questions": len(g),
            "single_pass_acc": round(100 * sum(r["correct"] for r in orig) / max(len(orig), 1), 2),
            "circular_acc": round(100 * sum(all(v) for v in g.values()) / len(g), 2),
            "row_acc": round(100 * sum(r["correct"] for r in recs) / len(recs), 2),
            "unparseable_rows": how.get("fail", 0), "extract_methods": how}
