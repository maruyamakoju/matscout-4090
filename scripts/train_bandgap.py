"""Train a composition -> bandgap surrogate and save to models/bandgap_surrogate.pkl.

Uses the matminer `matbench_expt_gap` dataset (experimental gaps, composition-only,
~4600 entries) featurized with Magpie element properties. `properties/bandgap.py`
auto-loads the saved model and falls back to its heuristic if absent.

Run:  .venv/Scripts/python.exe scripts/train_bandgap.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.simplefilter("ignore")


def main() -> None:
    import joblib
    import numpy as np
    from matminer.datasets import load_dataset
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    print("Loading matbench_expt_gap ...")
    df = load_dataset("matbench_expt_gap")
    # column names vary slightly across matminer versions
    comp_col = next(c for c in df.columns if "composition" in c.lower() or c.lower() == "formula")
    gap_col = next(c for c in df.columns if "gap" in c.lower())
    print(f"  {len(df)} entries; composition='{comp_col}', gap='{gap_col}'")

    def as_comp(x):
        return x if isinstance(x, Composition) else Composition(str(x))

    feat = ElementProperty.from_preset("magpie")
    print("Featurizing (Magpie) ...")
    X, y = [], []
    for c, g in zip(df[comp_col], df[gap_col]):
        try:
            X.append(feat.featurize(as_comp(c)))
            y.append(float(g))
        except Exception:
            continue
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"  feature matrix {X.shape}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42)
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_depth=8, l2_regularization=1.0,
        random_state=42,
    )
    print("Training HistGradientBoostingRegressor ...")
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    # fraction within 0.5 eV (used as a first-pass filter, so coarse accuracy is fine)
    within = float(np.mean(np.abs(pred - yte) <= 0.5))
    print(f"  holdout MAE={mae:.3f} eV  RMSE={rmse:.3f} eV  within-0.5eV={within:.1%}")

    # refit on all data for the shipped model
    model.fit(X, y)
    out = Path("models")
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "bandgap_surrogate.pkl")
    print(f"Saved -> {out / 'bandgap_surrogate.pkl'}")


if __name__ == "__main__":
    main()
