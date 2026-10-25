"""Grades a reply the way the practice answer keys define it.
must_include: facts joined by ' ; ' (ALL required); inside a fact ' / ' separates alternatives (ANY one).
must_not_include: ' ; ' separated phrases; none may appear."""
import csv, re, sys, os, unicodedata
from .config import dataset_dir

_DEV = str.maketrans("०१२३४५६७८९", "0123456789")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s).translate(_DEV).lower().replace("₹", " ").replace("rs.", " ")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return re.sub(r"\s+", " ", s)

def split_facts(spec: str):
    return [f.strip() for f in re.split(r"\s;\s", spec) if f.strip()] if spec else []

def check_include(reply: str, spec: str):
    r, missing = norm(reply), []
    for fact in split_facts(spec):
        alts = [a.strip() for a in re.split(r"\s/\s", fact) if a.strip()]
        if not any(norm(a) in r for a in alts): missing.append(fact)
    return missing

def check_exclude(reply: str, spec: str):
    r = norm(reply)
    return [p for p in split_facts(spec) if norm(p) in r]

def score_reply(reply: str, must_include: str, must_not_include: str) -> dict:
    miss, bad = check_include(reply, must_include), check_exclude(reply, must_not_include)
    return dict(passed=not miss and not bad, missing=miss, forbidden=bad)

def decision_ok(got: str, expected: str, acceptable: str = "") -> bool:
    ok = {x.strip() for x in (acceptable or expected).split(";") if x.strip()} | {expected}
    return got in ok

def recall(retrieved_ids, gold_ids) -> float:
    gold = [g for g in gold_ids if g]
    return 1.0 if not gold else sum(g in set(retrieved_ids) for g in gold) / len(gold)

def selftest(root=None):
    """Reference answers in questions.csv must pass the scorer. Sanity-checks the scorer itself."""
    root = root or dataset_dir()
    rows = list(csv.DictReader(open(os.path.join(root, "practice", "questions.csv"), encoding="utf-8", newline="")))
    fails = []
    for r in rows:
        res = score_reply(r["reference_answer"], r["must_include"], r["must_not_include"])
        if not res["passed"]: fails.append((r["id"], res))
    print(f"scorer self-test: {len(rows) - len(fails)}/{len(rows)} reference answers pass")
    for f in fails: print("  FAIL", f)
    return not fails

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
