"""Optional Streamlit dashboard: browse ranked candidates per campaign.

Run with:  streamlit run src/matscout/viz/dashboard.py
Requires the `extra` optional dependencies (streamlit, plotly).
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:  # pragma: no cover - interactive
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="MatScout-4090", layout="wide")
    st.title("MatScout-4090 — candidate explorer")

    out = Path("outputs")
    ranked_path = out / "candidates_ranked.parquet"
    if not ranked_path.exists():
        st.warning("No outputs/candidates_ranked.parquet yet. Run `matscout run-campaign`.")
        return

    df = pd.read_parquet(ranked_path)
    campaigns = ["(all)"] + sorted({t for tags in df["tags"] for t in tags
                                    if t in {"battery", "semiconductor", "catalyst", "solar", "co2_capture"}})
    choice = st.sidebar.selectbox("Campaign", campaigns)
    if choice != "(all)":
        df = df[df["tags"].apply(lambda ts: choice in ts)]

    min_score = st.sidebar.slider("min final_score", 0.0, 1.0, 0.3, 0.05)
    df = df[df["final_score"] >= min_score].sort_values("final_score", ascending=False)

    st.metric("candidates", len(df))
    cols = ["reduced_formula", "final_score", "novelty_score", "ml_e_above_hull",
            "predicted_bandgap_ev", "uncertainty_score", "source"]
    st.dataframe(df[[c for c in cols if c in df.columns]].head(300), use_container_width=True)

    try:
        import plotly.express as px

        fig = px.scatter(df.head(500), x="ml_e_above_hull", y="novelty_score",
                         size="final_score", color="predicted_bandgap_ev",
                         hover_name="reduced_formula")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Install plotly for interactive scatter (pip install -e '.[extra]').")

    # ---- 3D structure viewer (plotly, no extra deps) ----
    st.subheader("Structure viewer")
    if "structure_cif" not in df.columns or df.empty:
        st.info("No structures available.")
        return
    labels = [f"{r.reduced_formula}  ({r.candidate_id})" for r in df.head(300).itertuples()]
    pick = st.selectbox("Candidate", labels)
    cid = pick.split("(")[-1].rstrip(")")
    row = df[df["candidate_id"] == cid].iloc[0]
    _render_structure(st, row)


def _render_structure(st, row) -> None:
    import plotly.graph_objects as go
    from pymatgen.core import Structure

    c1, c2 = st.columns([2, 1])
    with c2:
        st.metric("final score", f"{row['final_score']:.3f}")
        for k in ["predicted_bandgap_ev", "ml_e_above_hull", "ml_formation_energy_per_atom",
                  "novelty_score", "uncertainty_score", "dft_verified"]:
            if k in row and row[k] is not None:
                st.write(f"**{k}**: {row[k]}")
    with c1:
        try:
            s = Structure.from_str(row["structure_cif"], fmt="cif")
        except Exception:
            st.warning("Could not parse structure.")
            return
        xs, ys, zs, syms = [], [], [], []
        for site in s:
            xs.append(site.coords[0]); ys.append(site.coords[1]); zs.append(site.coords[2])
            syms.append(site.specie.symbol)
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        uniq = sorted(set(syms))
        cmap = {el: palette[i % len(palette)] for i, el in enumerate(uniq)}
        fig = go.Figure()
        for el in uniq:
            idx = [i for i, e in enumerate(syms) if e == el]
            fig.add_trace(go.Scatter3d(
                x=[xs[i] for i in idx], y=[ys[i] for i in idx], z=[zs[i] for i in idx],
                mode="markers", name=el,
                marker=dict(size=8, color=cmap[el], line=dict(width=0.5, color="black")),
            ))
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(aspectmode="data"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{s.composition.reduced_formula} · {len(s)} atoms · "
                   f"SG {s.get_space_group_info()[1]}")


if __name__ == "__main__":
    main()
