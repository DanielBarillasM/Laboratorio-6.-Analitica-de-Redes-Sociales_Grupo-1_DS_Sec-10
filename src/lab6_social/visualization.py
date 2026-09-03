"""Visualizaciones estáticas y reproducibles del avance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {"navy": "#102A43", "blue": "#2563EB", "teal": "#0F9D91", "orange": "#F59E0B", "red": "#DC2626"}


def _stable_graph(graph: nx.Graph) -> nx.Graph:
    """Normaliza el orden de inserción para que los layouts sean repetibles."""

    stable = nx.Graph(**graph.graph)
    for node in sorted(graph.nodes):
        stable.add_node(node, **graph.nodes[node])
    ordered_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: tuple(sorted((edge[0], edge[1]))),
    )
    for source, target, data in ordered_edges:
        stable.add_edge(source, target, **data)
    return stable


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "font.family": "DejaVu Sans"})


def save_eda_overview(videos: pd.DataFrame, comments: pd.DataFrame, per_video: pd.DataFrame, output: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    videos["category"].value_counts().head(8).sort_values().plot.barh(ax=axes[0, 0], color=COLORS["blue"])
    axes[0, 0].set(title="Videos por categoría", xlabel="Videos", ylabel="")
    top = per_video.head(10).sort_values("comentarios")
    axes[0, 1].barh(top["title"].str.slice(0, 42), top["comentarios"], color=COLORS["teal"])
    axes[0, 1].set(title="Videos con más comentarios", xlabel="Comentarios", ylabel="")
    axes[1, 0].hist(np.log10(videos["view_count"].clip(lower=1)), bins=25, color=COLORS["orange"], edgecolor="white")
    axes[1, 0].set(title="Distribución de visualizaciones", xlabel="log10(visualizaciones)", ylabel="Videos")
    axes[1, 1].hist(comments["reply_count"], bins=range(0, int(comments["reply_count"].max()) + 2), color=COLORS["red"], edgecolor="white")
    axes[1, 1].set(title="Respuestas por comentario", xlabel="Respuestas observadas", ylabel="Comentarios")
    fig.suptitle("Panorama descriptivo de la muestra de YouTube", fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_frequency_plot(unigrams: pd.DataFrame, bigrams: pd.DataFrame, hashtags: pd.DataFrame, output: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for axis, frame, label, color in (
        (axes[0], unigrams.head(15), "Palabras", COLORS["blue"]),
        (axes[1], bigrams.head(15), "Bigramas", COLORS["teal"]),
    ):
        ordered = frame.sort_values("frecuencia")
        axis.barh(ordered["ngram"], ordered["frecuencia"], color=color)
        axis.set(title=f"{label} frecuentes", xlabel="Frecuencia", ylabel="")
    if not hashtags.empty:
        ordered = hashtags.head(15).sort_values("frecuencia")
        axes[2].barh(ordered["hashtag"], ordered["frecuencia"], color=COLORS["orange"])
    axes[2].set(title="Hashtags frecuentes", xlabel="Frecuencia", ylabel="")
    fig.suptitle("Contenido observable en comentarios", fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_concentration(concentration: pd.DataFrame, output: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    for unit, group in concentration.groupby("unidad"):
        ax.plot(group["top_n"], group["participacion_pct"], marker="o", linewidth=2.5, label=unit.capitalize())
    ax.set(title="Concentración acumulada de comentarios", xlabel="Número de unidades principales", ylabel="Comentarios acumulados (%)", xticks=[1, 3, 5, 10], ylim=(0, 105))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_popularity(per_video: pd.DataFrame, output: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=per_video, x="view_count", y="comentarios", hue="category", size="autores_unicos", sizes=(20, 260), alpha=0.75, ax=ax, legend=False)
    ax.set_xscale("log")
    ax.set(title="Visibilidad y participación observada", xlabel="Visualizaciones (escala log)", ylabel="Comentarios recolectados")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_bipartite(graph: nx.Graph, output: Path, seed: int = 42) -> None:
    setup_style()
    active = [node for node, degree in graph.degree() if degree > 0]
    isolates = [node for node, degree in graph.degree() if degree == 0]
    active_graph = _stable_graph(graph.subgraph(active).copy())
    pos = nx.spring_layout(active_graph, seed=seed, weight="weight", k=0.32, iterations=100)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [4, 1]})
    author_nodes = [node for node in active_graph if active_graph.nodes[node]["node_type"] == "author"]
    video_nodes = [node for node in active_graph if active_graph.nodes[node]["node_type"] == "video"]
    nx.draw_networkx_edges(active_graph, pos, ax=axes[0], alpha=0.16, width=0.7, edge_color="#64748B")
    nx.draw_networkx_nodes(active_graph, pos, nodelist=author_nodes, ax=axes[0], node_size=18, node_color=COLORS["blue"], alpha=0.8, label="Autores")
    nx.draw_networkx_nodes(active_graph, pos, nodelist=video_nodes, ax=axes[0], node_size=100, node_color=COLORS["orange"], node_shape="s", label="Videos")
    axes[0].set_title("Componente con participación observada")
    axes[0].legend(markerscale=1.5)
    axes[0].axis("off")
    columns = 18
    isolated_x = np.arange(len(isolates)) % columns
    isolated_y = np.arange(len(isolates)) // columns
    axes[1].scatter(isolated_x, isolated_y, s=18, marker="s", color="#94A3B8", alpha=0.8)
    axes[1].set(title=f"{len(isolates)} videos aislados\n(cada cuadro es un nodo)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    fig.suptitle("Red bipartita completa autor–video", fontsize=17, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_projection(graph: nx.Graph, title: str, output: Path, seed: int = 42) -> None:
    setup_style()
    active = [node for node, degree in graph.degree() if degree > 0]
    active_graph = _stable_graph(graph.subgraph(active).copy())
    pos = nx.spring_layout(active_graph, seed=seed, weight="weight", k=0.45, iterations=120)
    weighted_degree = dict(active_graph.degree(weight="weight"))
    sizes = [25 + 8 * np.sqrt(weighted_degree[node]) for node in active_graph]
    widths = [0.3 + 0.35 * np.log1p(data.get("weight", 1)) for _, _, data in active_graph.edges(data=True)]
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(active_graph, pos, ax=ax, alpha=0.18, width=widths, edge_color="#64748B")
    nx.draw_networkx_nodes(active_graph, pos, ax=ax, node_size=sizes, node_color=COLORS["blue"], alpha=0.82)
    ax.set_title(title, fontsize=16, weight="bold", color=COLORS["navy"])
    ax.text(0.01, 0.01, f"Nodos aislados conservados en tablas/métricas: {graph.number_of_nodes() - len(active)}", transform=ax.transAxes, color="#475569")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_degree_distributions(degrees: pd.DataFrame, output: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (name, frame) in zip(axes, degrees.groupby("red")):
        counts = frame["degree"].value_counts().sort_index()
        axis.plot(counts.index, counts.values, marker="o", color=COLORS["teal"])
        axis.set_yscale("log")
        axis.set(title=name, xlabel="Grado", ylabel="Nodos (escala log)")
    fig.suptitle("Distribuciones de grado y concentración estructural", fontsize=16, weight="bold", color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_communities(graph: nx.Graph, membership: dict[str, int], output: Path, seed: int = 42) -> None:
    setup_style()
    active = _stable_graph(graph.subgraph(membership.keys()).copy())
    pos = nx.spring_layout(active, seed=seed, weight="weight", k=0.45, iterations=120)
    colors = [membership[node] for node in active]
    sizes = [25 + 7 * np.sqrt(active.degree(node, weight="weight")) for node in active]
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(active, pos, ax=ax, alpha=0.12, width=0.6, edge_color="#64748B")
    nx.draw_networkx_nodes(active, pos, ax=ax, node_color=colors, cmap="tab20", node_size=sizes, alpha=0.88)
    ax.set_title("Comunidades de coparticipación entre autores", fontsize=16, weight="bold", color=COLORS["navy"])
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
