# Audit: Do the parameters match history?
# We'll measure how often each ONES filter would eliminate the *actual next draw*
# when applied with the previous draw as the seed.
import re, pandas as pd
from collections import Counter
from pathlib import Path

# --- Load history (chronological) ---
lines = Path("/mnt/data/pwrbll.txt").read_text(encoding="utf-8").splitlines()
rows = []
pat = re.compile(r"^(.*?),\s+Powerball:\s+(\d+)")
for line in lines:
    m = pat.search(line)
    if not m:
        continue
    date_numbers, _ = m.groups()
    parts = date_numbers.split("\t")
    date = parts[0].strip()
    nums = re.findall(r"\d{2}", parts[-1])
    if not nums:
        # fallback
        nums = re.findall(r"\b\d{1,2}\b", line.split("Powerball:")[0])
        nums = [n.zfill(2) for n in nums[-5:]]
    nums = sorted(int(n) for n in nums[-5:])
    rows.append({"Date": date, "Numbers": nums})

df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

def ones_of(xs): return [x%10 for x in xs]
def tens_of(xs): return [x//10 for x in xs]

# --- Build pairwise seed->next ---
pairs = []
for i in range(len(df)-1):
    seed = df.loc[i,"Numbers"]
    nxt  = df.loc[i+1,"Numbers"]
    pairs.append((seed, nxt))

# --- Context helpers to evaluate ones filters ---
def multiset_shared(a,b):
    ca, cb = Counter(a), Counter(b)
    return sum((Counter(a) & Counter(b)).values())

def ctx_for_ones(seed,next_):
    seed_ones = ones_of(seed)
    combo_ones = ones_of(next_)
    ones_sum = sum(combo_ones)
    ones_even = sum(1 for d in combo_ones if d%2==0)
    ones_odd = 5 - ones_even
    ones_unique = len(set(combo_ones))
    ones_range = max(combo_ones) - min(combo_ones)
    ones_low = sum(1 for d in combo_ones if d<=4)
    ones_high = 5 - ones_low
    ctx = {
        "combo_ones": combo_ones,
        "seed_ones": seed_ones,
        "prev_seed_ones": ones_of(seed),  # same as seed here (we don't have 2-back in this quick audit)
        "prev_prev_seed_ones": [],
        "ones_sum": ones_sum,
        "seed_ones_sum": sum(seed_ones),
        "ones_even_count": ones_even,
        "ones_odd_count": ones_odd,
        "ones_unique_count": ones_unique,
        "ones_range": ones_range,
        "ones_low_count": ones_low,
        "ones_high_count": ones_high,
        "Counter": Counter,
        "shared_ones": multiset_shared,
        "hot_digits": [],
        "cold_digits": [],
        "due_digits": [],
    }
    return ctx

# --- Load the big ones filter file ---
ones_filters_path = Path("/mnt/data/pb_ones_filters_all.csv")
ones_filters = pd.read_csv(ones_filters_path, dtype=str).fillna("")

# Precompile expressions
compiled = []
for _, r in ones_filters.iterrows():
    fid = r["id"]; name=r["name"]; app=r["applicable_if"] or "True"; expr=r["expression"] or "False"
    try:
        app_c = compile(app, f"<applicable:{fid}>", "eval")
        expr_c = compile(expr, f"<expr:{fid}>", "eval")
        compiled.append((fid, name, app, expr, app_c, expr_c))
    except Exception as e:
        compiled.append((fid, name, app, expr, None, None))

# --- Evaluate elimination of true next draw ---
stats = []
for fid,name,app,expr,app_c,expr_c in compiled:
    applicable_cnt = 0
    eliminated_cnt = 0
    if app_c is None or expr_c is None:
        stats.append((fid,name,0,0,"syntax_error"))
        continue
    for seed,next_ in pairs:
        ctx = ctx_for_ones(seed,next_)
        try:
            if eval(app_c, ctx, ctx):
                applicable_cnt += 1
                if eval(expr_c, ctx, ctx):
                    eliminated_cnt += 1
        except Exception as e:
            # skip errors
            pass
    rate = (eliminated_cnt / applicable_cnt) if applicable_cnt>0 else 0.0
    stats.append((fid,name,applicable_cnt,eliminated_cnt,rate))

audit_df = pd.DataFrame(stats, columns=["id","name","applicable","eliminated","elim_rate"])
audit_df = audit_df.sort_values(["elim_rate","applicable"], ascending=[False, False])

# Save summary files
out_summary = Path("/mnt/data/ones_filters_history_audit_summary.csv")
out_topbad = Path("/mnt/data/ones_filters_history_top_mismatch.csv")
out_topgood = Path("/mnt/data/ones_filters_history_nearmiss_zero.csv")

audit_df.to_csv(out_summary, index=False)
audit_df.query("applicable>=10 and elim_rate>=0.25").head(100).to_csv(out_topbad, index=False)
audit_df.query("applicable>=10 and elim_rate<=0.05").head(100).to_csv(out_topgood, index=False)

out_summary, out_topbad, out_topgood, audit_df.shape
