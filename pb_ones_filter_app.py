
import streamlit as st
from collections import Counter
import os, pandas as pd

ONES_DOMAIN = '0123456789'
LOW_SET = set([0,1,2,3,4])

def load_filters_df():
    candidates = [
        "pb_ones_filters_all.csv",
        "pb_ones_filters_all_clean.csv",
        "pb_ones_foundational_filters.csv",
        "pb_ones_percentile_filters.csv",
    ]
    paths = [p for p in candidates if os.path.exists(p)]
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p, dtype=str).fillna(""))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["id","name","enabled","applicable_if","expression"])
    df = pd.concat(dfs, ignore_index=True)
    for col in ["id","name","enabled","applicable_if","expression"]:
        if col not in df.columns: df[col] = ""
    df = df[["id","name","enabled","applicable_if","expression"]]
    # Dequote expressions if needed
    def deq(s):
        s = (s or "").strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1].strip()
        return s
    df["expression"] = df["expression"].apply(deq)
    # Fill missing IDs and de-dup by id
    built_ids, out = set(), []
    auto_idx = 0
    for _, r in df.iterrows():
        fid = (r["id"] or "").strip()
        if not fid:
            fid = f"AUTOID_{auto_idx:05d}"; auto_idx += 1
        if fid in built_ids:
            continue
        built_ids.add(fid)
        out.append([fid, r["name"], (r["enabled"] or "False"), r["applicable_if"], r["expression"]])
    return pd.DataFrame(out, columns=["id","name","enabled","applicable_if","expression"])

def generate_combos(seed_ones: str, method: str) -> list:
    seed_ones = ''.join(sorted(seed_ones))
    combos_set = set()
    if method == '1-digit':
        for d in seed_ones:
            for a in ONES_DOMAIN:
                for b in ONES_DOMAIN:
                    for c in ONES_DOMAIN:
                        for e in ONES_DOMAIN:
                            key = ''.join(sorted(d + a + b + c + e))
                            combos_set.add(key)
    else:
        pairs = {''.join(sorted((seed_ones[i], seed_ones[j])))
                 for i in range(len(seed_ones)) for j in range(i+1, len(seed_ones))}
        for p in pairs:
            for a in ONES_DOMAIN:
                for b in ONES_DOMAIN:
                    for c in ONES_DOMAIN:
                        key = ''.join(sorted(p + a + b + c))
                        combos_set.add(key)
    return sorted(combos_set)

def all_ones_combos():
    # All 100k 5-digit multisets from 0..9 (sorted representation)
    # Build by product and sort; dedup by multiset key
    from itertools import product
    combos = set()
    for tpl in product(ONES_DOMAIN, repeat=5):
        combos.add(''.join(sorted(tpl)))
    return sorted(combos)

def multiset_shared(a,b):
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())

def build_ctx(seed, prev, prevprev, combo, hot_input, cold_input, due_digits):
    seed_ones = [int(x) for x in seed]
    prev_ones = [int(x) for x in prev] if prev else []
    prev_prev_ones = [int(x) for x in prevprev] if prevprev else []
    combo_ones = [int(c) for c in combo]

    ones_sum = sum(combo_ones)
    ones_even = sum(1 for d in combo_ones if d%2==0)
    ones_odd = 5 - ones_even
    ones_unique = len(set(combo_ones))
    ones_range = max(combo_ones) - min(combo_ones)
    ones_low = sum(1 for d in combo_ones if d in LOW_SET)
    ones_high = 5 - ones_low

    hot_digits = [int(x) for x in hot_input.split(',') if x.strip().isdigit() and 0 <= int(x) <= 9]
    cold_digits = [int(x) for x in cold_input.split(',') if x.strip().isdigit() and 0 <= int(x) <= 9]

    return {
        'combo_ones': combo_ones,
        'seed_ones': seed_ones,
        'prev_seed_ones': prev_ones,
        'prev_prev_seed_ones': prev_prev_ones,
        'ones_sum': ones_sum,
        'seed_ones_sum': sum(seed_ones),
        'ones_even_count': ones_even,
        'ones_odd_count': ones_odd,
        'ones_unique_count': ones_unique,
        'ones_range': ones_range,
        'ones_low_count': ones_low,
        'ones_high_count': ones_high,
        'Counter': Counter,
        'shared_ones': multiset_shared,
        'hot_digits': hot_digits,
        'cold_digits': cold_digits,
        'due_digits': list(due_digits or []),
    }

def eval_bool(expr_code, ctx):
    try:
        val = eval(expr_code, ctx, ctx)
        return bool(val) if isinstance(val, bool) else False
    except Exception:
        return False

def main():
    st.sidebar.header("🎯 Powerball Ones-Only — Manual Filter Runner (v7)")

    filters_df = load_filters_df()
    st.caption(f"{len(filters_df)} filters loaded." if not filters_df.empty else "No filters loaded.")

    # Inputs
    seed = st.sidebar.text_input("Seed ones (5 digits 0–9):", "16170").strip()
    prev = st.sidebar.text_input("Prev ones (optional):", "44753").strip()
    prevprev = st.sidebar.text_input("Prev-prev ones (optional):", "70139").strip()
    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot digits (comma sep):", "3,4")
    cold_input = st.sidebar.text_input("Cold digits (comma sep):", "6")

    # Due
    st.sidebar.subheader("Due digits")
    m = st.sidebar.slider("Auto window m", 1, 3, 2)
    due_mode = st.sidebar.radio("Due source", ["Auto", "Manual", "Auto ∪ Manual"], index=0)
    manual_due_text = st.sidebar.text_input("Manual due (0–9):", "")
    def digits(s): return [int(x) for x in s] if s else []
    seeds_chain = [seed, prev, prevprev]
    seen = set(); used = 0
    for s in seeds_chain:
        if s and used < m:
            seen.update(digits(s)); used += 1
    auto_due = [d for d in range(10) if d not in seen]
    manual_due = []
    if manual_due_text.strip():
        for tok in manual_due_text.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 0 <= v <= 9: manual_due.append(v)
    if due_mode == "Auto":
        due_set = auto_due
    elif due_mode == "Manual":
        due_set = manual_due
    else:
        due_set = sorted(set(auto_due) | set(manual_due))
    st.sidebar.write(f"**Current due set:** {{ {', '.join(map(str, due_set))} }}")

    # Tracking
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test combos (ones as 5 digits, e.g., 00123; one per line or comma-separated):", height=120)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    # Validation
    if len(seed) != 5 or (not seed.isdigit()):
        st.error("Seed must be exactly 5 digits."); return
    if prev and (len(prev) != 5 or (not prev.isdigit())):
        st.error("Prev must be 5 digits or blank."); return
    if prevprev and (len(prevprev) != 5 or (not prevprev.isdigit())):
        st.error("Prev-prev must be 5 digits or blank."); return

    # Generate & track
    combos = generate_combos(seed, method)
    generated_set = set(combos)

    # Option for count base
    st.sidebar.markdown("---")
    count_base = st.sidebar.radio("Init-cuts count base", ["Generated pool", "Full 100k space"], index=0)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations (in selected base)", value=True)

    if count_base == "Full 100k space":
        # Precompute once (cache not persisted here)
        all_combos = all_ones_combos()
        count_pool = all_combos
    else:
        count_pool = combos

    # Normalize tracked
    toks = []
    for line in track_text.splitlines():
        for token in line.replace(',',' ').split():
            toks.append(token.strip())
    normalized_tracked = []
    for tok in toks:
        digits_only = ''.join([c for c in tok if c.isdigit()])
        if len(digits_only) == 5:
            normalized_tracked.append(''.join(sorted(digits_only)))
    normalized_tracked = sorted(set(normalized_tracked))
    tracked_set = set(normalized_tracked)

    audit = { c: {"combo": c, "generated": (c in generated_set), "injected": False, "preserved": bool(preserve_tracked),
                  "survived": None, "eliminated": False, "eliminated_by": None, "eliminated_name": None,
                  "eliminated_order": None, "would_eliminate_by": None, "would_eliminate_name": None,
                  "would_eliminate_order": None}
              for c in normalized_tracked }
    if inject_tracked:
        for c in normalized_tracked:
            if c not in generated_set:
                combos.append(c); generated_set.add(c); audit[c]["injected"] = True

    # Compile filters
    compiled = []
    for i, r in filters_df.reset_index(drop=True).iterrows():
        fid = (r["id"] or "").strip() or f"AUTOID_{i:05d}"
        name = (r["name"] or "").strip()
        app = (r["applicable_if"] or "True").strip()
        expr = (r["expression"] or "False").strip()
        try:
            app_c = compile(app, f"<app:{fid}>", "eval")
            expr_c = compile(expr, f"<expr:{fid}>", "eval")
            enabled = (r.get("enabled","False").strip().lower()=="true")
            compiled.append((fid, name, app_c, expr_c, enabled))
        except Exception:
            continue

    # Initial counts in chosen base (strict booleans)
    def build_ctx_for(combo):
        return build_ctx(seed, prev, prevprev, combo, hot_input, cold_input, due_set)

    init_counts = {}
    for fid, name, app_c, expr_c, en in compiled:
        cnt = 0
        for combo in count_pool:
            ctx = build_ctx_for(combo)
            if eval_bool(app_c, ctx) and eval_bool(expr_c, ctx):
                cnt += 1
        init_counts[fid] = cnt

    st.header("🔧 Manual Filters (ones-only)")
    st.sidebar.markdown(f"**Generated (pre-filter):** {len(combos)} combos")

    # Sort by count in chosen base
    sorted_filters = sorted(compiled, key=lambda f: (init_counts.get(f[0],0) == 0, -init_counts.get(f[0],0)))
    display_filters = [f for f in sorted_filters if init_counts.get(f[0],0) > 0] if hide_zero else sorted_filters

    # Bulk select buttons
    keys = [f"chk_{fid}_{i}" for i,(fid, *_rest) in enumerate(display_filters)]
    st.session_state.setdefault("visible_filter_keys", keys)
    if st.session_state["visible_filter_keys"] != keys:
        st.session_state["visible_filter_keys"] = keys

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Select all (visible)"):
        for k in st.session_state["visible_filter_keys"]:
            st.session_state[k] = True
    if c2.button("Deselect all (visible)"):
        for k in st.session_state["visible_filter_keys"]:
            st.session_state[k] = False

    # Apply filters
    pool = list(combos)
    order_idx = 0
    for i,(fid, name, app_c, expr_c, en) in enumerate(display_filters):
        key = f"chk_{fid}_{i}"
        label = f"{fid}: {name} — init cuts {init_counts[fid]} ({count_base})"
        if key not in st.session_state:
            st.session_state[key] = en  # seed from CSV 'enabled'
        chk = st.checkbox(label, key=key)
        if chk:
            order_idx += 1
            survivors = []
            for combo in pool:
                ctx = build_ctx_for(combo)
                eliminate = eval_bool(app_c, ctx) and eval_bool(expr_c, ctx)
                if eliminate:
                    if combo in tracked_set and preserve_tracked:
                        info = audit.get(combo)
                        if info and info.get("would_eliminate_by") is None:
                            info["would_eliminate_by"] = fid
                            info["would_eliminate_name"] = name
                            info["would_eliminate_order"] = order_idx
                        survivors.append(combo); continue
                    if combo in tracked_set and not audit[combo]["eliminated"]:
                        audit[combo]["eliminated"] = True
                        audit[combo]["eliminated_by"] = fid
                        audit[combo]["eliminated_name"] = name
                        audit[combo]["eliminated_order"] = order_idx
                else:
                    survivors.append(combo)
            pool = survivors

    survivors_set = set(pool)
    for c in normalized_tracked:
        if c in audit:
            audit[c]["survived"] = (c in survivors_set)

    st.subheader(f"Remaining after manual filters: {len(pool)}")

    # Audit table
    if normalized_tracked:
        st.markdown("### 🔎 Tracked/Preserved Combos — Audit")
        rows = []
        for c in normalized_tracked:
            info = audit.get(c, {})
            rows.append({
                "combo": c,
                "generated": info.get("generated", False),
                "injected": info.get("injected", False),
                "preserved": info.get("preserved", False),
                "survived": info.get("survived", False),
                "eliminated": info.get("eliminated", False),
                "eliminated_by": info.get("eliminated_by"),
                "eliminated_order": info.get("eliminated_order"),
                "eliminated_name": info.get("eliminated_name"),
                "would_eliminate_by": info.get("would_eliminate_by"),
                "would_eliminate_order": info.get("would_eliminate_order"),
                "would_eliminate_name": info.get("would_eliminate_name"),
            })
        df_a = pd.DataFrame(rows, columns=[
            "combo","generated","injected","preserved","survived",
            "eliminated","eliminated_by","eliminated_order","eliminated_name",
            "would_eliminate_by","would_eliminate_order","would_eliminate_name"
        ])
        st.dataframe(df_a, use_container_width=True, hide_index=True)
        st.download_button("Download audit (CSV)", df_a.to_csv(index=False), file_name="pb_ones_audit_tracked.csv", mime="text/csv")

    # Survivors + downloads
    st.markdown("### ✅ Survivors")
    with st.expander("Show remaining combinations"):
        tracked_first = [c for c in pool if c in tracked_set]
        others = [c for c in pool if c not in tracked_set]
        if tracked_first:
            st.write("**Tracked survivors:**")
            for c in tracked_first: st.write(c)
            st.write("---")
        for c in others: st.write(c)

    df_out = pd.DataFrame({"ones_combo": pool})
    st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), "pb_ones_survivors.csv", "text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(pool), "pb_ones_survivors.txt", "text/plain")

if __name__ == "__main__":
    main()
