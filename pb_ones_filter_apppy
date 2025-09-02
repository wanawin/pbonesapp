
import streamlit as st
from itertools import product
from collections import Counter
import csv, os
import pandas as pd

ONES_DOMAIN = '0123456789'
LOW_SET = set([0,1,2,3,4])
HIGH_SET = set([5,6,7,8,9])

def load_filters(paths):
    filters = []
    if not isinstance(paths, (list, tuple)): paths = [paths]
    for path in paths:
        if not path or not os.path.exists(path): continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = {k.lower(): v for k, v in raw.items()}
                row['id'] = row.get('id', row.get('fid', '')).strip()
                for key in ('name','applicable_if','expression'):
                    if key in row and isinstance(row[key], str):
                        row[key] = row[key].strip().strip('"').strip("'")
                row['expression'] = (row.get('expression') or 'False').replace('!==','!=')
                row['expr_str'] = row['expression']
                applicable = row.get('applicable_if') or 'True'
                expr = row.get('expression') or 'False'
                try:
                    row['applicable_code'] = compile(applicable,'<applicable>','eval')
                    row['expr_code'] = compile(expr,'<expr>','eval')
                except SyntaxError as e:
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                row['enabled_default'] = (row.get('enabled','').lower() == 'true')
                filters.append(row)
    return filters

def generate_ones_combinations(seed_ones: str, method: str) -> list:
    seed_ones = ''.join(sorted(seed_ones))
    combos_set = set()
    if method == '1-digit':
        for d in seed_ones:
            for p in product(ONES_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(p)))
                combos_set.add(key)
    else:
        pairs = {''.join(sorted((seed_ones[i], seed_ones[j])))
                 for i in range(len(seed_ones)) for j in range(i+1, len(seed_ones))}
        for pair in pairs:
            for p in product(ONES_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(p)))
                combos_set.add(key)
    return sorted(combos_set)

def multiset_shared(a,b):
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())

def build_ctx(seed_ones_str, prev_ones_str, prev_prev_ones_str, combo_str, hot_input, cold_input, due_digits_param):
    seed_ones = [int(x) for x in seed_ones_str]
    prev_ones = [int(x) for x in prev_ones_str] if prev_ones_str else []
    prev_prev_ones = [int(x) for x in prev_prev_ones_str] if prev_prev_ones_str else []
    combo_ones = [int(c) for c in combo_str]

    ones_sum = sum(combo_ones)
    ones_even = sum(1 for d in combo_ones if d%2==0)
    ones_odd = 5 - ones_even
    ones_unique = len(set(combo_ones))
    ones_range = max(combo_ones) - min(combo_ones)
    ones_low = sum(1 for d in combo_ones if d in LOW_SET)
    ones_high = 5 - ones_low

    hot_digits = [int(x) for x in hot_input.split(',') if x.strip().isdigit() and 0 <= int(x) <= 9]
    cold_digits = [int(x) for x in cold_input.split(',') if x.strip().isdigit() and 0 <= int(x) <= 9]
    due_digits = list(due_digits_param) if due_digits_param is not None else []

    seed_ones_sum = sum(seed_ones) if seed_ones else 0

    ctx = {
        'combo_ones': combo_ones,
        'seed_ones': seed_ones,
        'prev_seed_ones': prev_ones,
        'prev_prev_seed_ones': prev_prev_ones,
        'ones_sum': ones_sum,
        'seed_ones_sum': seed_ones_sum,
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
        'due_digits': due_digits,
    }
    return ctx

def normalize_combo_text(text: str):
    raw_tokens = []
    for line in text.splitlines():
        for token in line.replace(',',' ').split():
            raw_tokens.append(token.strip())
    normalized, invalid = [], []
    for tok in raw_tokens:
        digits = [c for c in tok if c.isdigit()]
        if len(digits) != 5 or any(c not in ONES_DOMAIN for c in digits):
            invalid.append(tok); continue
        normalized.append(''.join(sorted(digits)))
    seen, out = set(), []
    for n in normalized:
        if n not in seen:
            out.append(n); seen.add(n)
    return out, invalid

def main():
    st.sidebar.header("🎯 Powerball Ones-Only — Manual Filter Runner")

    default_filters_path = "pb_ones_foundational_filters.csv"
    default_extra_path = "pb_ones_percentile_filters.csv"
    st.sidebar.caption("Filters default to ones-only foundational & percentile bands.")
    use_default = st.sidebar.checkbox("Use default ones filters", value=True)
    uploaded_filters = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])

    filter_paths = []
    if use_default and os.path.exists(default_filters_path): filter_paths.append(default_filters_path)
    if os.path.exists(default_extra_path): filter_paths.append(default_extra_path)
    if uploaded_filters is not None:
        upath = "user_ones_filters.csv"
        with open(upath, "wb") as f: f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)

    filters = load_filters(filter_paths)

    # Seed inputs
    seed = st.sidebar.text_input("Seed ones (Draw 1-back, 5 digits 0–9):", placeholder="e.g., 57999").strip()
    prev_seed = st.sidebar.text_input("Prev ones (Draw 2-back, 5 digits 0–9, optional):").strip()
    prev_prev = st.sidebar.text_input("Prev-prev ones (Draw 3-back, 5 digits 0–9, optional):").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot ones digits (comma-separated 0–9, optional):").strip()
    cold_input = st.sidebar.text_input("Cold ones digits (comma-separated 0–9, optional):").strip()

    # Due controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Due digits (ones)")
    m = st.sidebar.slider("Auto window m (use last m ones seeds)", min_value=1, max_value=3, value=2, step=1)
    due_mode = st.sidebar.radio("Due source", ["Auto (from last m)", "Manual override", "Auto + manual (union)"], index=0)
    manual_due_text = st.sidebar.text_input("Manual due digits (0–9, comma-separated)", value="")
    disable_due_filters_when_empty = st.sidebar.checkbox("Disable due-based filters when due set is empty", value=True)

    def digits_from_str(s): return [int(x) for x in s] if s else []
    seeds_chain = [seed, prev_seed, prev_prev]
    seen = set(); used = 0
    for s in seeds_chain:
        if s and used < m:
            seen.update(digits_from_str(s)); used += 1
    auto_due = [d for d in range(10) if d not in seen]

    manual_due = []
    if manual_due_text.strip():
        for tok in manual_due_text.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 0 <= v <= 9: manual_due.append(v)

    if due_mode == "Auto (from last m)":
        due_digits_current = auto_due
    elif due_mode == "Manual override":
        due_digits_current = manual_due
    else:
        due_digits_current = sorted(set(auto_due) | set(manual_due))

    st.sidebar.write(f"**Current due set:** {{ {', '.join(map(str, due_digits_current))} }}")

    # Track/test combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test combos (ones as 5 digits, e.g., 00123, 57999; one per line or comma-separated):", height=120)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    # Validate
    if len(seed) != 5 or (not seed.isdigit()) or any(c not in ONES_DOMAIN for c in seed):
        st.sidebar.error("Seed ones must be exactly 5 digits in 0–9 (e.g., 57999)."); return
    if prev_seed and (len(prev_seed) != 5 or (not prev_seed.isdigit()) or any(c not in ONES_DOMAIN for c in prev_seed)):
        st.sidebar.error("Prev ones must be 5 digits in 0–9 or left blank."); return
    if prev_prev and (len(prev_prev) != 5 or (not prev_prev.isdigit()) or any(c not in ONES_DOMAIN for c in prev_prev)):
        st.sidebar.error("Prev-prev ones must be 5 digits in 0–9 or left blank."); return

    combos = generate_ones_combinations(seed, method)

    tracked_norm, invalid_tokens = normalize_combo_text(track_text)
    if invalid_tokens:
        st.sidebar.warning(f"Ignored invalid entries: {', '.join(invalid_tokens[:5])}" + (" ..." if len(invalid_tokens)>5 else ""))
    tracked_set = set(tracked_norm)

    generated_set = set(combos)
    audit = { c: {"combo": c, "generated": (c in generated_set), "preserved": bool(preserve_tracked),
                  "injected": False, "eliminated": False, "eliminated_by": None,
                  "eliminated_name": None, "eliminated_order": None,
                  "would_eliminate_by": None, "would_eliminate_name": None, "would_eliminate_order": None}
              for c in tracked_norm }

    if inject_tracked:
        for c in tracked_norm:
            if c not in generated_set:
                combos.append(c); generated_set.add(c)
                audit[c]["injected"] = True

    # Initial counts
    init_counts = {flt['id']: 0 for flt in filters}
    for flt in filters:
        if disable_due_filters_when_empty and not due_digits_current and 'due_digits' in flt.get('expr_str',''):
            init_counts[flt['id']] = 0; continue
        ic = 0
        for combo in combos:
            ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx): ic += 1
            except Exception: pass
        init_counts[flt['id']] = ic

    st.sidebar.markdown(f"**Generated (pre-filter):** {len(combos)} combos")
    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=True)

    sorted_filters = sorted(filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))
    display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

    pool = list(combos)
    st.header("🔧 Manual Filters (ones-only)")
    order_index = 0
    dynamic_counts = {}
    for flt in display_filters:
        order_index += 1
        key = f"filter_{flt['id']}"
        default_checked = select_all and flt['enabled_default']
        checked = st.checkbox(f"{flt['id']}: {flt['name']} — init cuts {init_counts[flt['id']]}", key=key, value=default_checked)
        if checked:
            if disable_due_filters_when_empty and not due_digits_current and 'due_digits' in flt.get('expr_str',''):
                dynamic_counts[flt['id']] = 0; continue
            survivors = []
            dc = 0
            for combo in pool:
                ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
                eliminate = False
                try:
                    eliminate = eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx)
                except Exception: eliminate = False
                is_tracked = combo in tracked_set
                if eliminate:
                    if is_tracked and preserve_tracked:
                        if audit.get(combo) and audit[combo]["would_eliminate_by"] is None:
                            audit[combo]["would_eliminate_by"] = flt['id']
                            audit[combo]["would_eliminate_name"] = flt.get('name','')
                            audit[combo]["would_eliminate_order"] = order_index
                        survivors.append(combo); continue
                    dc += 1
                    if is_tracked and not audit[combo]["eliminated"]:
                        audit[combo]["eliminated"] = True
                        audit[combo]["eliminated_by"] = flt['id']
                        audit[combo]["eliminated_name"] = flt.get('name','')
                        audit[combo]["eliminated_order"] = order_index
                else:
                    survivors.append(combo)
            pool = survivors
            dynamic_counts[flt['id']] = dc

    st.subheader(f"Remaining after manual filters: {len(pool)}")
    survivors_set = set(pool)

    # Audit
    if tracked_norm:
        st.markdown("### 🔎 Tracked/Preserved Combos — Audit")
        rows = []
        for c in tracked_norm:
            info = audit.get(c, {})
            rows.append({
                "combo": c, "generated": info.get("generated", False),
                "survived": (c in survivors_set),
                "eliminated": info.get("eliminated", False),
                "eliminated_by": info.get("eliminated_by"), "eliminated_order": info.get("eliminated_order"),
                "eliminated_name": info.get("eliminated_name"),
                "would_eliminate_by": info.get("would_eliminate_by"),
                "would_eliminate_order": info.get("would_eliminate_order"),
                "would_eliminate_name": info.get("would_eliminate_name"),
                "injected": info.get("injected", False), "preserved": info.get("preserved", False),
            })
        df_audit = pd.DataFrame(rows, columns=[
            "combo","generated","survived","eliminated","eliminated_by",
            "eliminated_order","eliminated_name","would_eliminate_by",
            "would_eliminate_order","would_eliminate_name","injected","preserved"
        ])
        st.dataframe(df_audit, use_container_width=True)
        st.download_button("Download audit (CSV)", df_audit.to_csv(index=False), file_name="pb_ones_audit_tracked.csv", mime="text/csv")

    # Survivors
    st.markdown("### ✅ Survivors")
    with st.expander("Show remaining combinations"):
        tracked_survivors = [c for c in pool if c in tracked_set]
        if tracked_survivors:
            st.write("**Tracked survivors:**")
            for c in tracked_survivors:
                info = audit.get(c, {})
                if info and info.get("would_eliminate_by"):
                    st.write(f"{c} — ⚠ would be eliminated by {info['would_eliminate_by']} at step {info.get('would_eliminate_order')} ({info.get('would_eliminate_name')}) — preserved")
                else:
                    st.write(c)
            st.write("---")
        for c in pool:
            if c not in tracked_set: st.write(c)

    # Downloads
    df_out = pd.DataFrame({"ones_combo": pool})
    st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), file_name="pb_ones_survivors.csv", mime="text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(pool), file_name="pb_ones_survivors.txt", mime="text/plain")

if __name__ == "__main__":
    main()
