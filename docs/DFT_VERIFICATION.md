# DFT verification guide

MatScout exports DFT-ready decks for the top candidates. The local pipeline (CHGNet/MACE
relaxation + surrogate scoring) is a **filter**; these decks are the verification step on a
cluster (MIT cluster, lab resources, or a collaborator's allocation).

## What's in each deck

```
outputs/dft_queue/<campaign>/<PRIORITY>_<rank>_<formula>_<id>/
  vasp/
    POSCAR              relaxed structure (ML-relaxed starting point)
    INCAR.relax         cell+ionic relaxation (ISIF=3); LDAU auto-enabled for TM oxides
    INCAR.static        static run (ISMEAR=-5 insulator / 1 metal), LORBIT=11 for DOS
    KPOINTS             Γ-centered, ~1000 k-points/atom density
    POTCAR.spec.txt     recommended PBE PAW labels (you supply the actual POTCARs)
    run_vasp.slurm      SLURM batch: relax -> static
  qe/
    relax.in            Quantum ESPRESSO vc-relax (ecutwfc 70 Ry, SSSP-style)
    run_qe.slurm        SLURM batch
  metadata.json         scores, e_hull/bandgap surrogates, uncertainty, provenance, priority
outputs/dft_queue/<campaign>/submit_queue.sh   submit all decks, priority A -> B -> C
```

## Priority tiers (see `metadata.json`)
- **A** — high score, low ML uncertainty, high novelty, e_above_hull < 50 meV/atom. Run first.
- **B** — high score, moderate uncertainty, e_above_hull < 100 meV/atom (or unknown without an MP key).
- **C** — scientifically interesting but higher uncertainty / harder synthesis.

## VASP

1. **Assemble POTCAR** (not shipped — licensed). In POSCAR species order, e.g. for `Na Cl`:
   ```bash
   cat $VASP_PP/Na_pv/POTCAR $VASP_PP/Cl/POTCAR > POTCAR
   ```
   Use the labels in `POTCAR.spec.txt`.
2. Edit `run_vasp.slurm`: set `--partition`, modules, and `vasp_std` path/MPI.
3. Submit one deck, or the whole queue:
   ```bash
   bash outputs/dft_queue/battery/submit_queue.sh        # priority A -> B -> C
   ```
4. Each job runs `INCAR.relax` (→ CONTCAR) then `INCAR.static`. Compare the static energy /
   formation energy / e_above_hull against the ML estimate in `metadata.json`.

## Quantum ESPRESSO

1. Put SSSP (efficiency or precision) `*.UPF` pseudopotentials under `qe/pseudo/`
   (names must match `relax.in`).
2. Edit `run_qe.slurm` (partition, modules, `pw.x`), then `sbatch run_qe.slurm`.

## After DFT
- Recompute e_above_hull with the DFT energy against a Materials Project phase diagram
  (set `MP_API_KEY` and use `matscout.validation.phase_diagram`).
- For semiconductors/solar, the surrogate gap is PBE-level at best — escalate the survivors to
  **HSE06 / GW** or experiment for the real gap.
- For catalysts, the bulk + single-termination surface proxy is only a prioritizer; run full
  surface phase diagrams (terminations, coverage, solvation, applied potential) on the winners.

## Recommended first batch
Start with all **priority-A** decks across campaigns, then **priority-B** for battery
(solid-electrolyte / coating candidates verify cleanly) and solar (Pb-free absorbers — check
the gap and direct/indirect character).
