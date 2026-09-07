"""Visualizaciones de centralidad, participantes puente y experimentos de remoción."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .visualization import COLORS, _stable_graph, setup_style


GREY = "#94A3B8"
MEASURES = (
    ("grado_ponderado", "Grado ponderado (weight)"),
    ("intermediacion", "Intermediación (distance)"),
    ("armonica", "Centralidad armónica (distance)"),
    ("pagerank", "PageRank (weight)"),
)


def short_label(value: object, width: int = 34) -> str:
    """Recorta etiquetas y descarta glifos que la tipografía base no puede dibujar."""

    text = str(value).encode("latin-1", "ignore").decode("latin-1").strip()
    text = " ".join(text.split()) or "(sin etiqueta)"
    return text if len(text) <= width else text[: width - 1] + "…"


def _unique_labels(frame: pd.DataFrame, width: int = 34) -> list[str]:
    """Etiquetas legibles y sin repeticiones para ejes categóricos."""

    labels: list[str] = []
    seen: dict[str, int] = {}
    for label, node_id in zip(frame["label"], frame["node_id"]):
        text = short_label(label, width)
        seen[text] = seen.get(text, 0) + 1
        labels.append(text if seen[text] == 1 else f"{text} [{str(node_id)[-4:]}]")
    return labels


def save_centrality_panel(centrality: pd.DataFrame, title: str, output: Path, top_n: int = 12) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    palette = (COLORS["blue"], COLORS["red"], COLORS["teal"], COLORS["orange"])
    for axis, (column, caption), color in zip(axes.ravel(), MEASURES, palette):
        top = centrality.nlargest(top_n, [column, "grado"]).iloc[::-1]
        axis.barh(_unique_labels(top), top[column], color=color)
        axis.set(title=caption, xlabel="Valor", ylabel="")
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle(title, fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _bridge_colors(graph: nx.Graph, structural: set[str], redundant: set[str]) -> list[str]:
    return [
        COLORS["red"] if node in structural else COLORS["orange"] if node in redundant else GREY
        for node in graph
    ]


def save_bridge_authors(
    projection: nx.Graph, bridges: pd.DataFrame, profiles: pd.DataFrame, output: Path, seed: int = 42
) -> None:
    """Ubica a los autores puente en la proyección y contrasta recurrencia con intermediación."""

    setup_style()
    structural = set(bridges.loc[bridges["es_punto_articulacion"], "node"])
    redundant = set(bridges.loc[~bridges["es_punto_articulacion"], "node"])
    active = [node for node, degree in projection.degree() if degree > 0]
    graph = _stable_graph(projection.subgraph(active).copy())
    pos = nx.spring_layout(graph, seed=seed, weight="weight", k=0.45, iterations=120)
    colors = _bridge_colors(graph, structural, redundant)
    sizes = [130 if node in structural else 70 if node in redundant else 22 for node in graph]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3, 2]})
    nx.draw_networkx_edges(graph, pos, ax=axes[0], alpha=0.16, width=0.7, edge_color="#64748B")
    nx.draw_networkx_nodes(graph, pos, ax=axes[0], node_color=colors, node_size=sizes, alpha=0.9)
    for node in sorted(structural, key=lambda item: -graph.degree(item))[:8]:
        if node in pos:
            axes[0].annotate(short_label(graph.nodes[node].get("label", node), 18), pos[node],
                             fontsize=7, color=COLORS["navy"], xytext=(4, 4), textcoords="offset points")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS["red"], label=f"Puente estructural ({len(structural)})"),
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS["orange"], label=f"Puente redundante ({len(redundant)})"),
        plt.Line2D([], [], marker="o", linestyle="", color=GREY, label="Resto de autores activos"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=9)
    axes[0].set_title("Autores puente en la proyección autor–autor", fontsize=13, weight="bold", color=COLORS["navy"])
    axes[0].axis("off")

    scatter = profiles.merge(
        bridges[["node", "es_punto_articulacion"]], on="node", how="left"
    ) if not bridges.empty else profiles.assign(es_punto_articulacion=False)
    scatter["es_punto_articulacion"] = scatter["es_punto_articulacion"].fillna(False)
    jitter = np.random.default_rng(seed).normal(0, 0.045, len(scatter))
    axes[1].scatter(
        scatter["videos_comentados"] + jitter, scatter["intermediacion"],
        c=[COLORS["red"] if flag else GREY for flag in scatter["es_punto_articulacion"]],
        s=[60 if flag else 22 for flag in scatter["es_punto_articulacion"]], alpha=0.8,
    )
    axes[1].set(
        title="Recurrencia no implica intermediación",
        xlabel="Videos comentados por el autor", ylabel="Intermediación (normalizada)",
    )
    axes[1].set_xticks(sorted(scatter["videos_comentados"].unique()))
    fig.suptitle("Participantes puente entre audiencias", fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_articulator_videos(
    projection: nx.Graph, articulators: pd.DataFrame, output: Path, seed: int = 42, top_n: int = 12
) -> None:
    """Muestra los videos que conectan audiencias y los que sostienen la conexión."""

    setup_style()
    structural = set(articulators.loc[articulators["es_punto_articulacion"], "node"])
    redundant = set(articulators.loc[~articulators["es_punto_articulacion"], "node"])
    active = [node for node, degree in projection.degree() if degree > 0]
    graph = _stable_graph(projection.subgraph(active).copy())
    pos = nx.spring_layout(graph, seed=seed, weight="weight", k=0.5, iterations=120)
    colors = _bridge_colors(graph, structural, redundant)
    sizes = [150 if node in structural else 80 if node in redundant else 30 for node in graph]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3, 2]})
    nx.draw_networkx_edges(graph, pos, ax=axes[0], alpha=0.3, width=1.0, edge_color="#64748B")
    nx.draw_networkx_nodes(graph, pos, ax=axes[0], node_color=colors, node_shape="s", node_size=sizes, alpha=0.9)
    for node, coordinates in pos.items():
        axes[0].annotate(short_label(graph.nodes[node].get("label", node), 24), coordinates,
                         fontsize=7, color=COLORS["navy"], xytext=(6, 5), textcoords="offset points")
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=COLORS["red"], label=f"Video articulador ({len(structural)})"),
        plt.Line2D([], [], marker="s", linestyle="", color=COLORS["orange"], label=f"Conector redundante ({len(redundant)})"),
        plt.Line2D([], [], marker="s", linestyle="", color=GREY, label="Video conectado sin rol de paso"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=9)
    axes[0].set_title("Videos articuladores en la proyección video–video", fontsize=13, weight="bold", color=COLORS["navy"])
    axes[0].axis("off")

    if articulators.empty:
        axes[1].text(0.5, 0.5, "Sin videos articuladores observados", ha="center", va="center")
        axes[1].axis("off")
    else:
        top = articulators.nlargest(top_n, ["videos_conectados", "autores_compartidos"]).iloc[::-1]
        positions = np.arange(len(top))
        width = 0.38
        axes[1].barh(positions + width / 2, top["videos_conectados"], height=width, color=COLORS["blue"], label="Videos conectados")
        axes[1].barh(positions - width / 2, top["autores_compartidos"], height=width, color=COLORS["teal"], label="Autores compartidos")
        axes[1].set_yticks(positions, _unique_labels(top, 30), fontsize=8)
        axes[1].xaxis.get_major_locator().set_params(integer=True)
        axes[1].set(title="Alcance de los videos conectores", xlabel="Conteo", ylabel="")
        axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Videos que conectan audiencias distintas", fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_removal_impact(impact: pd.DataFrame, output: Path, top_n: int = 8) -> None:
    """Compara fragmentación y componente mayor antes y después de remover cada nodo.

    Cada red ocupa su propia fila porque la proyección video–video parte de una
    fragmentación mucho mayor y sus magnitudes no son comparables en un solo eje.
    """

    setup_style()
    networks = list(dict.fromkeys(impact["red"])) if not impact.empty else []
    fig, axes = plt.subplots(max(len(networks), 1), 2, figsize=(16, 4.6 * max(len(networks), 1)), squeeze=False)
    if impact.empty:
        for axis in axes.ravel():
            axis.text(0.5, 0.5, "Sin nodos candidatos para remoción", ha="center", va="center")
            axis.axis("off")
    for row, red in enumerate(networks):
        subset = impact[impact["red"] == red]
        top = subset.nlargest(top_n, ["incremento_fragmentacion", "componentes_nuevos"]).iloc[::-1]
        positions = np.arange(len(top))
        colors = [COLORS["red"] if flag else COLORS["orange"] for flag in top["es_punto_articulacion"]]
        left, right = axes[row]
        left.barh(positions, 100 * top["incremento_fragmentacion"], color=colors)
        left.set_yticks(positions, _unique_labels(top, 30), fontsize=8)
        left.set(title=f"Incremento de fragmentación · {red.replace('_', ' ')}",
                 xlabel="Puntos porcentuales de pares desconectados", ylabel="")
        handles = [
            plt.Line2D([], [], marker="s", linestyle="", color=COLORS["red"], label="Punto de articulación"),
            plt.Line2D([], [], marker="s", linestyle="", color=COLORS["orange"], label="Nodo central sin articulación"),
        ]
        left.legend(handles=handles, fontsize=8, loc="lower right")

        width = 0.38
        right.barh(positions - width / 2, top["componente_mayor_antes"], height=width, color=GREY, label="Antes")
        right.barh(positions + width / 2, top["componente_mayor_despues"], height=width, color=COLORS["blue"], label="Después")
        right.set_yticks(positions, [f"{value:+.1f} %" for value in top["cambio_componente_mayor_pct"]], fontsize=8)
        right.set(title=f"Componente mayor antes y después · {red.replace('_', ' ')}",
                  xlabel="Nodos en la componente mayor", ylabel="")
        right.legend(fontsize=8)
    fig.suptitle("Experimentos de remoción de nodos", fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
