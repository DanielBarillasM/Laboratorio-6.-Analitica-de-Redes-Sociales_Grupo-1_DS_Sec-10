"""Visualizaciones finales de contenido y sentimiento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {"negativo": "#DC2626", "neutral": "#64748B", "positivo": "#0F9D91"}
NAVY = "#102A43"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "font.family": "DejaVu Sans"})


def save_sentiment_overview(predictions: pd.DataFrame, output: Path) -> None:
    _style()
    counts = predictions["sentimiento"].value_counts().reindex(COLORS, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    bars = axes[0].bar(counts.index, counts.values, color=[COLORS[label] for label in counts.index])
    axes[0].bar_label(bars, labels=[f"{value}\n({100 * value / counts.sum():.1f}%)" for value in counts.values], padding=4)
    axes[0].set(title="Clasificación general", xlabel="Sentimiento", ylabel="Comentarios", ylim=(0, max(counts.values) * 1.2))
    axes[1].hist(predictions["confianza"], bins=np.linspace(0.33, 1.0, 18), color="#2563EB", edgecolor="white")
    axes[1].axvline(0.60, color="#F59E0B", linestyle="--", label="umbral de baja confianza")
    axes[1].set(title="Confianza de la clase ganadora", xlabel="Probabilidad máxima", ylabel="Comentarios", xlim=(0.3, 1.01))
    axes[1].legend()
    fig.suptitle("Sentimiento de comentarios con RoBERTuito", fontsize=16, weight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_confidence_plot(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary["nivel_confianza"].astype(str), summary["comentarios"], color=["#F59E0B", "#2563EB", "#0F9D91"])
    ax.bar_label(bars, labels=[f"{row.comentarios} ({row.porcentaje:.1f}%)" for row in summary.itertuples()], padding=4)
    ax.set(title="Diagnóstico de confianza del modelo", xlabel="Banda", ylabel="Comentarios", ylim=(0, max(summary["comentarios"]) * 1.2))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_group_sentiment(
    summary: pd.DataFrame,
    label_column: str,
    title: str,
    output: Path,
    top_n: int = 10,
) -> None:
    _style()
    work = summary[summary["muestra_interpretable"]].head(top_n).copy()
    if work.empty:
        work = summary.head(top_n).copy()
    work[label_column] = work[label_column].astype(str).str.slice(0, 48)
    work = work.sort_values("n_comentarios")
    fig_height = max(4.5, 0.55 * len(work) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    left = np.zeros(len(work))
    for label in ("negativo", "neutral", "positivo"):
        values = work[f"pct_{label}"].to_numpy()
        ax.barh(work[label_column], values, left=left, label=label.capitalize(), color=COLORS[label])
        left += values
    for index, row in enumerate(work.itertuples()):
        ax.text(101, index, f"n={row.n_comentarios}", va="center", fontsize=8, color="#475569")
    ax.set(title=title, xlabel="Distribución de comentarios (%)", ylabel="", xlim=(0, 112))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_community_sentiment(summary: pd.DataFrame, output: Path) -> None:
    work = summary.copy()
    work["comunidad_etiqueta"] = work["community"].map(
        lambda value: "Sin comunidad (aislados)" if int(value) == 0 else f"Comunidad {int(value)}"
    )
    save_group_sentiment(work, "comunidad_etiqueta", "Sentimiento por comunidad de coparticipación", output, top_n=len(work))
