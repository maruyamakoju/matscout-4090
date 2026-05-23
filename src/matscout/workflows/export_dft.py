"""Export top candidates as ready-to-run VASP + Quantum ESPRESSO input decks."""

from __future__ import annotations

import json

from pymatgen.core import Element, Structure
from pymatgen.io.vasp.inputs import Kpoints, Poscar
from rich.console import Console

from ..config.schema import CampaignConfig, GlobalConfig
from ..data.schemas import CandidateRecord
from .paths import Paths

console = Console()

# Recommended PBE PAW POTCAR labels (subset; default to bare symbol).
_POTCAR_REC = {
    "Li": "Li_sv", "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv",
    "Ca": "Ca_sv", "Sr": "Sr_sv", "Ba": "Ba_sv", "Sc": "Sc_sv", "Y": "Y_sv",
    "Ti": "Ti_pv", "V": "V_pv", "Nb": "Nb_pv", "Ta": "Ta_pv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co", "Ni": "Ni_pv", "Cu": "Cu_pv",
    "Zr": "Zr_sv", "Hf": "Hf_pv", "Mo": "Mo_pv", "W": "W_pv", "Ga": "Ga_d",
    "Ge": "Ge_d", "In": "In_d", "Sn": "Sn_d", "Bi": "Bi_d", "Pb": "Pb_d",
}
# Hubbard U values (eV) for common TM oxides (Materials Project-like).
_LDAU_U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Ni": 6.2, "V": 3.25, "W": 6.2, "Mo": 4.38}


def _priority(r: CandidateRecord) -> str:
    eh = r.ml_e_above_hull
    high_score = (r.final_score or 0) >= 0.5
    low_unc = r.uncertainty_score <= 0.05
    high_nov = r.novelty_score >= 0.5
    if high_score and low_unc and high_nov and (eh is not None and eh < 0.05):
        return "A"
    if high_score and r.uncertainty_score <= 0.10 and high_nov and (eh is None or eh < 0.10):
        return "B"
    return "C"


def _incar(structure: Structure, calc_mode: str, use_ldau: bool, metallic: bool = False) -> str:
    lines = [
        f"SYSTEM = MatScout {structure.composition.reduced_formula} {calc_mode}"
        f"{' metal' if metallic else ''}",
        "PREC = Accurate", "ENCUT = 520", "EDIFF = 1E-5", "ALGO = Normal",
        "LREAL = Auto", "LASPH = .TRUE.", "ISPIN = 2", "LWAVE = .FALSE.", "LCHARG = .FALSE.",
        f"ISMEAR = {1 if metallic else 0}", "SIGMA = 0.05",
    ]
    if calc_mode == "relax":
        lines += ["IBRION = 2", "ISIF = 3", "NSW = 200", "EDIFFG = -0.02"]
    else:  # static
        # tetrahedron method for insulators; keep Gaussian for metals
        lines += ["IBRION = -1", "NSW = 0", "LORBIT = 11"]
        lines += ["ISMEAR = 1" if metallic else "ISMEAR = -5"]
    if use_ldau:
        els = [str(e) for e in structure.composition.elements]
        ldauu = [str(_LDAU_U.get(e, 0.0)) for e in els]
        ldaul = ["2" if e in _LDAU_U else "-1" for e in els]
        lines += ["LDAU = .TRUE.", "LDAUTYPE = 2",
                  f"LDAUL = {' '.join(ldaul)}", f"LDAUU = {' '.join(ldauu)}",
                  f"LDAUJ = {' '.join('0.0' for _ in els)}", "LMAXMIX = 4"]
    return "\n".join(lines) + "\n"


def _potcar_spec(structure: Structure) -> str:
    els = [str(e) for e in structure.composition.elements]
    lines = ["# Recommended PBE PAW POTCAR labels (concatenate in this element order):"]
    lines += [f"{e}: {_POTCAR_REC.get(e, e)}" for e in els]
    lines.append("# Order matters and must match POSCAR species order.")
    return "\n".join(lines) + "\n"


def _qe_relax(structure: Structure) -> str:
    comp = structure.composition
    els = [str(e) for e in comp.elements]
    species_block = "\n".join(f"{e} {round(float(Element(e).atomic_mass), 3)} {e}.UPF" for e in els)
    cell = structure.lattice.matrix
    cell_block = "\n".join(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" for v in cell)
    pos_block = "\n".join(f"{site.specie.symbol} {site.frac_coords[0]:.8f} {site.frac_coords[1]:.8f} {site.frac_coords[2]:.8f}" for site in structure)
    return f"""&CONTROL
  calculation = 'vc-relax'
  prefix = '{comp.reduced_formula}'
  pseudo_dir = './pseudo'
  outdir = './out'
  forc_conv_thr = 1.0d-4
/
&SYSTEM
  ibrav = 0
  nat = {len(structure)}
  ntyp = {len(els)}
  ecutwfc = 70
  ecutrho = 560
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.4
/
&IONS
/
&CELL
/
ATOMIC_SPECIES
{species_block}
CELL_PARAMETERS angstrom
{cell_block}
ATOMIC_POSITIONS crystal
{pos_block}
K_POINTS automatic
4 4 4 0 0 0
"""


def _vasp_job_script(formula: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=ms_{formula}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=CHANGE_ME

# Assemble POTCAR first (see POTCAR.spec.txt) — concatenate POTCARs in POSCAR species order:
#   cat $VASP_PP/Na_pv/POTCAR $VASP_PP/Cl/POTCAR > POTCAR
module load vasp 2>/dev/null || true
set -e

# 1) cell + ionic relaxation
cp INCAR.relax INCAR
srun vasp_std
cp CONTCAR POSCAR

# 2) static (DOS / accurate energy)
cp INCAR.static INCAR
srun vasp_std
echo "done: {formula}"
"""


def _qe_job_script(formula: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=ms_{formula}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=CHANGE_ME

# Place SSSP pseudopotentials (*.UPF) under ./pseudo (names must match relax.in).
module load quantum-espresso 2>/dev/null || true
srun pw.x -in relax.in > relax.out
echo "done: {formula}"
"""


def export_one(record: CandidateRecord, out_dir, rank: int) -> dict:
    structure = record.get_structure()
    priority = _priority(record)
    folder = out_dir / f"{priority}_{rank:03d}_{record.reduced_formula}_{record.candidate_id.replace('::','-')}"
    folder.mkdir(parents=True, exist_ok=True)
    vasp = folder / "vasp"
    qe = folder / "qe"
    vasp.mkdir(exist_ok=True)
    qe.mkdir(exist_ok=True)

    use_ldau = ("O" in record.elements) and bool(set(record.elements) & set(_LDAU_U))
    metallic = record.predicted_bandgap_ev is not None and record.predicted_bandgap_ev < 0.1

    Poscar(structure).write_file(vasp / "POSCAR")
    (vasp / "INCAR.relax").write_text(_incar(structure, "relax", use_ldau, metallic), encoding="utf-8")
    (vasp / "INCAR.static").write_text(_incar(structure, "static", use_ldau, metallic), encoding="utf-8")
    Kpoints.automatic_density(structure, 1000).write_file(vasp / "KPOINTS")
    (vasp / "POTCAR.spec.txt").write_text(_potcar_spec(structure), encoding="utf-8")

    (qe / "relax.in").write_text(_qe_relax(structure), encoding="utf-8")
    (vasp / "run_vasp.slurm").write_text(_vasp_job_script(record.reduced_formula), encoding="utf-8")
    (qe / "run_qe.slurm").write_text(_qe_job_script(record.reduced_formula), encoding="utf-8")

    meta = {
        "candidate_id": record.candidate_id,
        "formula": record.reduced_formula,
        "source": record.source,
        "parent": record.parent_source_id or record.parent_mp_id,
        "priority": priority,
        "rank": rank,
        "final_score": record.final_score,
        "ml_e_above_hull": record.ml_e_above_hull,
        "predicted_bandgap_ev": record.predicted_bandgap_ev,
        "uncertainty_score": record.uncertainty_score,
        "novelty_score": record.novelty_score,
        "synthesizability_score": record.synthesizability_score,
        "ml_model": record.ml_model,
        "relaxation_converged": record.relaxation_converged,
        "ldau_enabled": use_ldau,
        "tags": record.tags,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"priority": priority, "folder": folder.name}


def export_dft_queue(ranked, campaign: CampaignConfig, g: GlobalConfig, top_k: int = 50) -> dict:
    import shutil

    paths = Paths(g)
    out_dir = paths.dft_queue(campaign.name)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # idempotent: clear previous run's decks
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"A": 0, "B": 0, "C": 0}
    folders = []
    for i, r in enumerate(ranked[:top_k], start=1):
        info = export_one(r, out_dir, i)
        counts[info["priority"]] += 1
        folders.append(info["folder"])
    # submit decks in priority order (A first), VASP by default
    submit = ["#!/bin/bash",
              "# Submit MatScout DFT decks in priority order (A -> B -> C).",
              "# Edit run_vasp.slurm partition/modules and assemble POTCAR before running.",
              "set -e", 'cd "$(dirname "$0")"']
    for folder in sorted(folders):  # priority prefix sorts A_ < B_ < C_
        submit.append(f'( cd "{folder}/vasp" && sbatch run_vasp.slurm )')
    sq = out_dir / "submit_queue.sh"
    sq.write_text("\n".join(submit) + "\n", encoding="utf-8")
    console.print(f"[green]DFT queue[/] {campaign.name}: {sum(counts.values())} decks "
                  f"(A={counts['A']} B={counts['B']} C={counts['C']}) -> {out_dir}")
    return counts
