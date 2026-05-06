"""
Streamlit Safety Verifier
=========================
A web UI wrapping the Logic Oracle pipeline. Run with:

    streamlit run app.py

The app has four tabs:
  1. Rule Encoder    - enter prose, see parsed expression + BDD diagram
  2. Live Oracle     - flip sensor toggles, see PROCEED/STOP/YIELD instantly
  3. Verification    - rule-set sanity check (contradictions + dead state)
  4. Audit Trail     - browse every BDD path + auto-generated audit entry
"""

from __future__ import annotations

import os

import streamlit as st
import graphviz

from logic_oracle_av import (
    CANONICAL_RULES,
    CANONICAL_VARS,
    DECISION_PRIORITY,
    UID_TO_NAME,
    _VAR_OBJS,
    all_paths_to,
    bdd_node_count,
    build_bdds,
    check_dead_state,
    find_contradictions,
    generate_audit_entry,
    oracle_decide,
    parse_rule,
    parsed_to_bdd,
)
from pyeda.boolalg.bdd import BDDNODEONE, BDDNODEZERO  # type: ignore


st.set_page_config(
    page_title="AV Logic Oracle",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_bdd(bdd) -> graphviz.Digraph:
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="TB", bgcolor="white")
    g.attr("node", fontname="Helvetica", fontsize="14")

    seen: dict[int, str] = {}

    def add(node) -> str:
        if node is None:
            return ""
        nid = id(node)
        if nid in seen:
            return seen[nid]
        gid = f"n{len(seen)}"
        seen[nid] = gid
        if node is BDDNODEONE:
            g.node(gid, "1", shape="box", style="filled",
                   fillcolor="#7bd389", color="#2e7d32")
        elif node is BDDNODEZERO:
            g.node(gid, "0", shape="box", style="filled",
                   fillcolor="#ef9a9a", color="#b71c1c")
        else:
            label = UID_TO_NAME.get(node.root, "?")
            g.node(gid, label, shape="circle", style="filled",
                   fillcolor="#bbdefb", color="#1565c0")
            hi_id = add(node.hi)
            lo_id = add(node.lo)
            g.edge(gid, hi_id, style="solid",  color="#1565c0",
                   label=" 1", penwidth="1.6")
            g.edge(gid, lo_id, style="dashed", color="#666666",
                   label=" 0", penwidth="1.2")
        return gid

    add(bdd.node)
    return g


# ---------------------------------------------------------------------------
# Sidebar - identity
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Project")
    st.markdown("**The Logic Oracle**")
    st.markdown("*AV Edition - PJ3*")
    st.markdown("---")
    st.markdown("**Author**")
    st.markdown("Sultan Alzaabi")
    st.markdown("ID: 2022005844")
    st.markdown("---")
    st.markdown("**Course**")
    st.markdown("MATH 214 - Discrete Thinking")
    st.markdown("---")
    api_status = "available" if os.getenv("ANTHROPIC_API_KEY") else "regex fallback"
    st.markdown(f"**Parser mode:** `{api_status}`")


st.title("🚗 The Logic Oracle for Autonomous Vehicles")
st.caption(
    "Formal verification of AV driving rules using Binary Decision Diagrams + LLMs"
)


tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣  Rule Encoder",
    "2️⃣  Live Oracle",
    "3️⃣  Verification",
    "4️⃣  Audit Trail",
])


# ===========================================================================
# TAB 1 - RULE ENCODER
# ===========================================================================

with tab1:
    st.header("Encode a driving rule from prose")
    st.markdown(
        "Enter an AV safety rule in plain English. The parser produces a "
        "Boolean expression, the BDD engine reduces it, and Graphviz renders "
        "the resulting ROBDD."
    )

    default_text = (
        "Proceed only if the light is green, no pedestrian is in the crosswalk, "
        "the gap is safe, the road is dry, and the speed is within the limit, "
        "unless an emergency vehicle is detected."
    )
    prose = st.text_area("AV safety rule (prose)", value=default_text, height=110)

    if st.button("Parse and build BDD", type="primary"):
        parsed = parse_rule(prose, prefer_llm=True)
        st.subheader("Parsed output")
        st.json(parsed)

        try:
            bdd, var_map = parsed_to_bdd(parsed)
            st.success(
                f"BDD built successfully ({bdd_node_count(bdd)} nodes for "
                f"{len(var_map)} sensor variables)."
            )
            sat_paths = all_paths_to(bdd, 1)
            unsat_paths = all_paths_to(bdd, 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("BDD nodes", bdd_node_count(bdd))
            c2.metric("Satisfying paths", len(sat_paths))
            c3.metric("Falsifying paths", len(unsat_paths))

            st.subheader("ROBDD diagram")
            st.graphviz_chart(render_bdd(bdd))

            with st.expander("All satisfying paths (PROCEED scenarios)"):
                for p in sat_paths:
                    st.code(p)
            with st.expander("All falsifying paths (STOP/YIELD scenarios)"):
                for p in unsat_paths:
                    st.code(p)
        except Exception as exc:
            st.error(f"Could not build BDD: {exc}")


# ===========================================================================
# TAB 2 - LIVE ORACLE
# ===========================================================================

with tab2:
    st.header("Live Oracle - flip sensor toggles")
    st.markdown(
        "Choose a sensor reading and the Oracle returns PROCEED, STOP, YIELD, "
        "PULL_OVER, or DECELERATE - with an ISO 26262-style audit entry."
    )

    cols = st.columns(len(CANONICAL_VARS))
    sensor_state: dict[str, int] = {}
    for col, (var, desc) in zip(cols, CANONICAL_VARS.items()):
        with col:
            val = st.toggle(f"**{var}**: {desc}", value=False, key=f"toggle_{var}")
            sensor_state[var] = int(val)

    use_llm = st.checkbox("Use LLM for the audit entry "
                          "(falls back to template if API unavailable)",
                          value=False)

    decision = oracle_decide(sensor_state, explain_with_llm=use_llm)

    color_map = {
        "PROCEED":    "#2e7d32",
        "STOP":       "#b71c1c",
        "YIELD":      "#ef6c00",
        "PULL_OVER":  "#6a1b9a",
        "DECELERATE": "#1565c0",
        "NO_ACTION":  "#424242",
    }
    color = color_map.get(decision.decision, "#424242")
    st.markdown(
        f"<div style='padding:18px;border-radius:10px;"
        f"background:{color};color:white;font-size:32px;text-align:center;'>"
        f"<b>{decision.decision}</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Triggered by rule:** `{decision.triggering_rule}`")
    st.markdown("**Audit entry:**")
    st.info(decision.audit_entry)

    with st.expander("Show priority ladder"):
        st.markdown("The Oracle evaluates rules in this fixed priority order:")
        for i, (rule, action) in enumerate(DECISION_PRIORITY, 1):
            st.markdown(f"{i}. `{rule}`  →  **{action}**")


# ===========================================================================
# TAB 3 - VERIFICATION
# ===========================================================================

with tab3:
    st.header("Rule-set verification")
    st.markdown(
        "Pairwise contradiction check + dead-state detection over the canonical "
        "5-rule AV set."
    )

    pairs = find_contradictions(CANONICAL_RULES)
    dead = check_dead_state(CANONICAL_RULES)

    c1, c2 = st.columns(2)
    c1.metric("Contradicting rule pairs", len(pairs))
    c2.metric("Dead-state present?", "YES" if dead else "NO",
              delta=None,
              delta_color="off")

    st.subheader("Pairwise contradiction matrix")
    rule_names = list(CANONICAL_RULES)
    matrix_lines = ["| | " + " | ".join(rule_names) + " |"]
    matrix_lines.append("|" + "|".join(["---"] * (len(rule_names) + 1)) + "|")
    for r1 in rule_names:
        row = [r1]
        for r2 in rule_names:
            if r1 == r2:
                row.append("—")
            else:
                contradicts = (r1, r2) in pairs or (r2, r1) in pairs
                row.append("✗ UNSAT" if contradicts else "✓ SAT")
        matrix_lines.append("| " + " | ".join(row) + " |")
    st.markdown("\n".join(matrix_lines))

    st.subheader("Why the contradictions are a feature, not a bug")
    st.info(
        "The five canonical rules describe MUTUALLY EXCLUSIVE actions "
        "(proceed vs stop vs yield etc.). The contradictions PROVE that "
        "no two actions can fire at once - the priority resolver therefore "
        "always returns a single, well-defined decision."
    )


# ===========================================================================
# TAB 4 - AUDIT TRAIL
# ===========================================================================

with tab4:
    st.header("Audit trail")
    st.markdown(
        "Pick a rule and target outcome to enumerate every BDD path that "
        "produces that outcome - one audit entry per path."
    )
    rule = st.selectbox("Rule", list(CANONICAL_RULES))
    target = st.radio("Target", ["1 (rule fires)", "0 (rule does NOT fire)"])
    target_val = 1 if target.startswith("1") else 0

    bdd = build_bdds(CANONICAL_RULES)[rule]
    paths = all_paths_to(bdd, target_val)
    st.metric(f"BDD paths to terminal {target_val}", len(paths))

    for i, p in enumerate(paths, 1):
        sensor_state = {v: p.get(v, 0) for v in CANONICAL_VARS}
        decision = "TRIGGERED" if target_val == 1 else "NOT_TRIGGERED"
        audit = generate_audit_entry(
            sensor_state=sensor_state,
            decision=decision,
            rule_name=rule,
            use_llm=False,
        )
        with st.expander(f"Path #{i}  -  {p}"):
            st.code(audit)
