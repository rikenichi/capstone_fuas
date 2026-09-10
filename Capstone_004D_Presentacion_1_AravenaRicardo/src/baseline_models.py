"""Baseline supervisado 2024 para Capstone FUAS.

Reglas:
- Usa exclusivamente analytic_2024.parquet.
- 2025 NO se lee.
- Split interno 80/20 estratificado, random_state=42.
- Compara Logística, Árbol, Random Forest y GaussianNB.
- Dos variantes de COD_DEPE (1..6 vs agrupado) y con/sin ETNIA.
- Sin SMOTE, tuning, SHAP ni optimización de threshold.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from preprocessing import CodDepeGrouper, split_train_valid

THRESHOLD = 0.5
OUT_REPORTS = cfg.REPORTS_DIR
OUT_FIGURES = cfg.FIGURES_DIR / "baseline_2024"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

METAS = {
    "roc_auc": 0.80,
    "balanced_accuracy": 0.75,
    "recall": 0.80,
    "f1": 0.75,
    "fnr_max": 0.20,
}

FORBIDDEN = set(cfg.FORBIDDEN_FEATURES) | {
    cfg.TARGET,
    "ANIO_PROCESO", "TIPO_BENEFICIO", "BENEFICIOS_VALIDOS",
    "BENEFICIOS_ASIGNADOS_RAW", "TIENE_GRATUIDAD", "TIENE_BECA",
    "TIENE_FSCU", "TIENE_OTRO", "NOMBRE_REGION", "NOMBRE_COMUNA",
}


def load_only_2024() -> pd.DataFrame:
    """Único punto de carga. Deliberadamente no existe referencia a analytic_2025."""
    path = cfg.DATA_ANALYTIC / "analytic_2024.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass
    pkl = cfg.DATA_ANALYTIC / "analytic_2024.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    raise FileNotFoundError(path)


def feature_list(include_etnia: bool = True) -> list[str]:
    feats = ["NEM", "QUINTIL_SE4", "COD_DEPE", "EDAD", "GENERO", "NACIONALIDAD"]
    if include_etnia:
        feats.append("ETNIA")
    bad = sorted(set(feats) & FORBIDDEN)
    if bad:
        raise ValueError(f"Features prohibidas: {bad}")
    return feats


def build_variant_preprocessor(*, group_cod_depe: bool, include_etnia: bool, scale_numeric: bool):
    numeric = ["NEM", "EDAD"]
    ordinal = ["QUINTIL_SE4"]
    nominal = ["COD_DEPE", "GENERO", "NACIONALIDAD"] + (["ETNIA"] if include_etnia else [])

    num_pipe = StandardScaler() if scale_numeric else "passthrough"

    nom_steps = []
    if group_cod_depe:
        nom_steps.append(("cod_depe_group", CodDepeGrouper(col="COD_DEPE")))
    # Baseline: no agrupación de categorías raras. Especialmente ETNIA permanece intacta.
    nom_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)))
    nom_pipe = Pipeline(nom_steps)

    return ColumnTransformer(
        [
            ("num", num_pipe, numeric),
            ("ord", "passthrough", ordinal),
            ("nom", nom_pipe, nominal),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def model_specs():
    return {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=500, random_state=cfg.RANDOM_SEED, solver="lbfgs"),
            "scale": True,
            "dense": False,
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=cfg.RANDOM_SEED),
            "scale": False,
            "dense": False,
        },
        "random_forest": {
            "model": RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_leaf=10, random_state=cfg.RANDOM_SEED, n_jobs=-1),
            "scale": False,
            "dense": False,
        },
        "gaussian_nb": {
            "model": GaussianNB(),
            "scale": True,
            "dense": True,
        },
    }


def make_pipeline(model_name: str, *, group_cod_depe: bool, include_etnia: bool) -> Pipeline:
    spec = model_specs()[model_name]
    steps = [
        ("preprocess", build_variant_preprocessor(
            group_cod_depe=group_cod_depe,
            include_etnia=include_etnia,
            scale_numeric=spec["scale"],
        )),
    ]
    if spec["dense"]:
        steps.append(("dense", FunctionTransformer(to_dense, accept_sparse=True)))
    steps.append(("model", clone(spec["model"])))
    return Pipeline(steps)


def metrics_from_probs(y_true, y_prob, threshold: float = THRESHOLD):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(rec),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fnr": float(1.0 - rec),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_dummy(Xtr, Xva, ytr, yva, strategy: str):
    kwargs = {"strategy": strategy}
    if strategy == "stratified":
        kwargs["random_state"] = cfg.RANDOM_SEED
    m = DummyClassifier(**kwargs)
    m.fit(Xtr[["NEM"]], ytr)
    proba = m.predict_proba(Xva[["NEM"]])[:, list(m.classes_).index(1)]
    return m, proba, metrics_from_probs(yva, proba)


def run_baseline(save: bool = True):
    df = load_only_2024()
    # split_train_valid usa exclusivamente 2024 y mismas filas para todas las variantes.
    Xtr_all, Xva_all, ytr, yva = split_train_valid(df)

    results = []
    curve_data = {}

    for model_name in model_specs():
        for group_cod in (False, True):
            for include_etnia in (False, True):
                feats = feature_list(include_etnia)
                Xtr = Xtr_all[feats]
                Xva = Xva_all[feats]
                pipe = make_pipeline(model_name, group_cod_depe=group_cod, include_etnia=include_etnia)
                t0 = time.perf_counter()
                pipe.fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                prob = pipe.predict_proba(Xva)[:, 1]
                met = metrics_from_probs(yva, prob)
                row = {
                    "modelo": model_name,
                    "cod_depe": "agrupado_3" if group_cod else "original_1_6",
                    "etnia": "con" if include_etnia else "sin",
                    "threshold": THRESHOLD,
                    "fit_seconds": round(fit_s, 3),
                    **met,
                }
                results.append(row)
                key = f"{model_name}|{row['cod_depe']}|{row['etnia']}"
                curve_data[key] = prob
                print(key, {k: round(met[k], 4) for k in ("roc_auc","balanced_accuracy","recall","precision","f1","fnr")}, f"{fit_s:.1f}s", flush=True)

    # Dummies (sin variantes, solo referencia)
    for strategy in ("most_frequent", "stratified"):
        _, prob, met = evaluate_dummy(Xtr_all, Xva_all, ytr, yva, strategy)
        row = {
            "modelo": f"dummy_{strategy}", "cod_depe": "N/A", "etnia": "N/A",
            "threshold": THRESHOLD, "fit_seconds": 0.0, **met,
        }
        results.append(row)
        curve_data[row["modelo"]] = prob

    res = pd.DataFrame(results)
    res = res.sort_values(["roc_auc", "balanced_accuracy", "f1"], ascending=False).reset_index(drop=True)

    # Metas por fila
    res["cumple_auc"] = res["roc_auc"] >= METAS["roc_auc"]
    res["cumple_bal_acc"] = res["balanced_accuracy"] >= METAS["balanced_accuracy"]
    res["cumple_recall"] = res["recall"] >= METAS["recall"]
    res["cumple_f1"] = res["f1"] >= METAS["f1"]
    res["cumple_fnr"] = res["fnr"] <= METAS["fnr_max"]
    res["n_metas_cumplidas"] = res[["cumple_auc","cumple_bal_acc","cumple_recall","cumple_f1","cumple_fnr"]].sum(axis=1)

    # Mejores
    real = res[~res["modelo"].str.startswith("dummy_")].copy()
    best_auc = real.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
    best_recall = real.sort_values(["recall", "fnr", "roc_auc"], ascending=[False, True, False]).iloc[0].to_dict()
    # equilibrio: metas cumplidas, bal acc, f1, auc
    best_balance = real.sort_values(["n_metas_cumplidas","balanced_accuracy","f1","roc_auc"], ascending=False).iloc[0].to_dict()

    # Impactos promedio COD_DEPE y ETNIA por mismo modelo/contraparte
    impacts = {"cod_depe": {}, "etnia": {}}
    for model in model_specs():
        sub = real[real.modelo == model]
        for metric in ["roc_auc","balanced_accuracy","recall","f1"]:
            impacts["cod_depe"].setdefault(model, {})[metric] = float(sub[sub.cod_depe=="agrupado_3"][metric].mean() - sub[sub.cod_depe=="original_1_6"][metric].mean())
            impacts["etnia"].setdefault(model, {})[metric] = float(sub[sub.etnia=="con"][metric].mean() - sub[sub.etnia=="sin"][metric].mean())

    summary = {
        "anio": 2024,
        "n_train": int(len(Xtr_all)),
        "n_validation": int(len(Xva_all)),
        "target_rate_train": float(ytr.mean()),
        "target_rate_validation": float(yva.mean()),
        "threshold": THRESHOLD,
        "metas": METAS,
        "versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
        "best_auc": best_auc,
        "best_recall_fnr": best_recall,
        "best_balance": best_balance,
        "impactos_promedio": impacts,
        "results": res.to_dict(orient="records"),
        "nota": "2025 no fue leído ni utilizado en esta etapa.",
    }

    if save:
        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        csv_path = OUT_REPORTS / "baseline_2024.csv"
        json_path = OUT_REPORTS / "baseline_2024.json"
        res.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # Matriz de confusión para mejor equilibrio
        best_key = f"{best_balance['modelo']}|{best_balance['cod_depe']}|{best_balance['etnia']}"
        cm = np.array([[best_balance["tn"], best_balance["fp"]], [best_balance["fn"], best_balance["tp"]]])
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(cm)
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set_xticks([0,1], labels=["Pred 0","Pred 1"])
        ax.set_yticks([0,1], labels=["Real 0","Real 1"])
        ax.set_title(f"Matriz de confusión — {best_balance['modelo']}\n{best_balance['cod_depe']}, ETNIA {best_balance['etnia']}")
        fig.tight_layout(); fig.savefig(OUT_FIGURES / "confusion_mejor_equilibrio.png", dpi=160); plt.close(fig)

        # Seleccionar mejor variante por modelo para curvas
        selected = {}
        for model in model_specs():
            r = real[real.modelo==model].sort_values("roc_auc", ascending=False).iloc[0]
            key = f"{model}|{r['cod_depe']}|{r['etnia']}"
            selected[model] = (key, curve_data[key])

        fig, ax = plt.subplots(figsize=(7,5))
        for model, (_, prob) in selected.items():
            fpr, tpr, _ = roc_curve(yva, prob)
            auc = roc_auc_score(yva, prob)
            ax.plot(fpr, tpr, label=f"{model} AUC={auc:.3f}")
        ax.plot([0,1],[0,1], linestyle="--", label="azar")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC — mejores variantes baseline 2024"); ax.legend(); fig.tight_layout()
        fig.savefig(OUT_FIGURES / "roc_comparativa.png", dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,5))
        for model, (_, prob) in selected.items():
            prec, rec, _ = precision_recall_curve(yva, prob)
            ap = average_precision_score(yva, prob)
            ax.plot(rec, prec, label=f"{model} AP={ap:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall — mejores variantes baseline 2024"); ax.legend(); fig.tight_layout()
        fig.savefig(OUT_FIGURES / "pr_comparativa.png", dpi=160); plt.close(fig)

        # Resumen de métricas de mejores variantes por modelo
        best_rows = []
        for model in model_specs():
            best_rows.append(real[real.modelo==model].sort_values("roc_auc", ascending=False).iloc[0])
        bdf = pd.DataFrame(best_rows).set_index("modelo")
        metrics = ["roc_auc","balanced_accuracy","recall","precision","f1"]
        ax = bdf[metrics].plot(kind="bar", figsize=(10,5))
        ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Resumen métricas — mejor variante por modelo")
        ax.legend(loc="lower right"); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(OUT_FIGURES / "metricas_resumen.png", dpi=160); plt.close()

        # Confusion por cada modelo principal (mejor variante por AUC)
        for model in model_specs():
            r = real[real.modelo==model].sort_values("roc_auc", ascending=False).iloc[0]
            cmx = np.array([[r.tn, r.fp],[r.fn, r.tp]], dtype=int)
            fig, ax = plt.subplots(figsize=(5,4)); ax.imshow(cmx)
            for (i,j), v in np.ndenumerate(cmx): ax.text(j,i,f"{v:,}",ha="center",va="center")
            ax.set_xticks([0,1], labels=["Pred 0","Pred 1"]); ax.set_yticks([0,1], labels=["Real 0","Real 1"])
            ax.set_title(f"{model} — mejor AUC"); fig.tight_layout(); fig.savefig(OUT_FIGURES / f"confusion_{model}.png", dpi=160); plt.close(fig)

    return res, summary


if __name__ == "__main__":
    res, summary = run_baseline(save=True)
    cols = ["modelo","cod_depe","etnia","roc_auc","balanced_accuracy","recall","precision","f1","fnr","accuracy","n_metas_cumplidas"]
    print("\nRESULTADOS\n", res[cols].to_string(index=False))
    print("\nMEJOR AUC:", summary["best_auc"]["modelo"], summary["best_auc"]["roc_auc"])
    print("MEJOR EQUILIBRIO:", summary["best_balance"]["modelo"], summary["best_balance"]["n_metas_cumplidas"])
