# MatScout-4090 — Findings (fixture-seed run)

_Generated 2026-05-23 from a local CHGNet+MACE ensemble run across all five campaigns._

> **Provenance / honesty note.** This run used the **bundled 24-prototype fixture seeds**
> (no Materials Project API key). e_above_hull is therefore unavailable (cohort-relative
> stability proxy used instead); bandgaps come from a composition surrogate (MAE 0.43 eV).
> Treat these as **hypotheses to verify by DFT**, not discoveries. With an MP key the seed
> pool becomes thousands of real stable structures and the candidates get substantially richer.

## Headline
The most promising, scientifically-defensible hits are in **solar absorbers**, where the
trained bandgap surrogate cleanly selects **Pb-free, earth-abundant chalcogenides/halides in
the 1.1–1.6 eV single-junction window** — matching the spec's hypotheses P2 (kesterite) and
P3 (Bi/Cu chalcogenide-halide). This is the campaign I'd take to DFT first.

---

## Solar absorbers — strongest results (take to DFT first)

| candidate | gap (eV) | final | why interesting | hypothesis |
|---|---|---|---|---|
| **Cu₂SnS₄** | 1.08 | 0.590 | CZTS-type kesterite derivative, earth-abundant, thin-film compatible | P2 |
| **Cu(SnBr₂)₂** | 1.33 | 0.600 | Sn(II) lone-pair + soft halide, ideal single-junction gap | P1/P3 |
| **Cu(SnI₂)₂** | 1.34 | 0.593 | as above, heavier halide | P3 |
| **Zn(CuS₂)₂** | 1.29 | 0.592 | Cu–Zn–S, CZTS chemistry, abundant | P2 |
| **Cu(BiI₂)₂ / Cu₂BiS₄** | 1.10–1.12 | 0.58 | Bi lone-pair, defect-tolerant, Pb-free perovskite alternative | P3 |

All are Pb-free (zero toxicity penalty), direct-gap-guessed, in the 1.1–1.6 eV window.
**Next step:** HSE06/GW gap + direct/indirect character (decks provided, see below); then defect
tolerance (antibonding VBM) and absorption coefficient.

## Battery — solid-electrolyte / multivalent candidates

| candidate | gap (eV) | final | note |
|---|---|---|---|
| MgAl₂O₄ (spinel) | 2.34 | 0.749 | stable wide-gap oxide framework; Mg multivalent host |
| MgS₂ / Mg₂S₃ | 3.8–4.1 | 0.67 | novel Mg–S, soft anion → lower migration barrier |
| MgCl₂ | 7.04 | 0.673 | wide electrochemical window |

Fixture seeds bias this toward simple binaries. **The real win here needs MP seeds** (NASICON,
garnet, argyrodite, LISICON) — exactly the spec's #1 "most likely to hit" target. Re-run with
`MP_API_KEY` for credible solid-electrolyte candidates.

## Semiconductor

| candidate | gap (eV) | note |
|---|---|---|
| **AlCuO₂** (delafossite) | 1.84 | real p-type transparent conductor family; top score 0.80 |
| MgAl₂O₄ / Al₂O₃ | 2.3 | wide-gap dielectrics |
| SrTiO₃ / TiO₂ | 2.8 | photocatalysis / wide-gap oxide |

## Catalyst (bulk pre-screen + surface proxy)

Bulk top: **TiMnO₃, MnO, MnAl₂O₄, TiFeO₃** (mixed-valence 3d-TM oxides, OER-relevant).
Surface analysis (`reports/catalyst_surface.md`, CHGNet (111) terminations):
- **TiO₂(111) is the best HER candidate** — dG(\*H) ≈ −0.2 eV, near the Sabatier optimum.
- All oxide OER overpotentials are high in this crude single-termination estimate (expected:
  polar oxide surfaces overbind; needs proper terminations + applied potential in DFT).

⚠️ Catalyst scores are confidence-capped (~0.5). Real activity needs surface DFT / OCP models.

## CO₂ capture
Top bulk hits (Al₂FeO₄, CaTiO₃, Al₂O₃) are **low-confidence** — this campaign requires porous
MOF/COF structures with pore descriptors (PLD/LCD/void fraction). **Action:** drop CoRE MOF /
QMOF CIFs (+ `descriptors.csv`) into `data/external/core_mof/`; the loader + scorer are ready.

## Wildcards (high novelty + high model uncertainty)
TiMnO₃ (unc 0.36), MnO (0.59), and Ti–Mn–Cu / Fe–Cu–W intermetallics surfaced as
high-uncertainty/high-novelty — the CHGNet↔MACE disagreement flags these as the most
"interesting but risky" picks, i.e. where DFT is most informative.

---

## Recommended next actions
1. **Add `MP_API_KEY`** and re-run — unlocks real e_above_hull + thousands of real seeds
   (biggest single quality lever; transforms battery especially).
2. **DFT-verify the solar shortlist** (Cu₂SnS₄, Cu(SnBr₂)₂, Cu/Bi halides) — decks in
   `outputs/dft_queue/solar/`; refine gaps with the **HSE06 decks** (`*/vasp/INCAR.hse06`).
3. **CO₂:** ingest a CoRE MOF subset to make that campaign meaningful.
4. **Catalyst:** run full surface phase diagrams (terminations/coverage/potential) on TiO₂ (HER)
   and the mixed-valence oxides (OER) — or wire an OCP/fairchem model.

See `outputs/reports/executive_summary.md` and per-campaign `*_shortlist.md` for full tables,
and `figures/` for Pareto fronts, stability-vs-novelty, and composition maps.
