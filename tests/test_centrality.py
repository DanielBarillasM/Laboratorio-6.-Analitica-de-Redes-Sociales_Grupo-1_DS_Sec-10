import networkx as nx
import pandas as pd
import pytest

from lab6_social.centrality import (
    add_distance,
    articulation_table,
    author_profiles,
    bridge_authors,
    centrality_measures,
    community_structure,
    removal_candidates,
    removal_impact,
)


def weighted_path() -> nx.Graph:
    """a -1- b -2- c: la arista b-c es el doble de fuerte, luego la mitad de larga."""

    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("b", "c", weight=2)
    return graph


def bowtie() -> nx.Graph:
    """Dos triángulos unidos por el nodo c, que es el único punto de articulación."""

    graph = nx.Graph()
    for source, target in (("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("c", "e")):
        graph.add_edge(source, target, weight=1)
    return graph


def test_distance_is_the_inverse_of_weight():
    prepared = add_distance(weighted_path())
    assert prepared["a"]["b"]["distance"] == 1.0
    assert prepared["b"]["c"]["distance"] == 0.5
    assert prepared["b"]["c"]["weight"] == 2


def test_distance_rejects_non_positive_weight():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0)
    with pytest.raises(ValueError):
        add_distance(graph)


def test_betweenness_marks_the_middle_of_a_path():
    table = centrality_measures(weighted_path(), "prueba").set_index("node")
    assert table.loc["b", "intermediacion"] == 1.0
    assert table.loc["a", "intermediacion"] == 0.0
    assert table.loc["c", "intermediacion"] == 0.0


def test_weighted_degree_uses_weight_and_harmonic_uses_distance():
    table = centrality_measures(weighted_path(), "prueba").set_index("node")
    assert table.loc["b", "grado_ponderado"] == 3.0
    assert table.loc["c", "grado_ponderado"] == 2.0
    # c está más cerca de b que a, porque su arista pesa más: la distancia es 1/peso.
    assert table.loc["c", "armonica"] > table.loc["a", "armonica"]


def test_star_centre_leads_every_measure():
    star = nx.star_graph(4)
    nx.set_edge_attributes(star, 1, "weight")
    table = centrality_measures(star, "estrella").set_index("node")
    for column in ("grado", "grado_ponderado", "intermediacion", "armonica", "pagerank"):
        assert table.loc[0, column] == table[column].max()


def test_harmonic_tolerates_disconnected_components():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("y", "z", weight=1)
    table = centrality_measures(graph, "dos componentes").set_index("node")
    assert (table["armonica"] > 0).all()
    assert table["componente_tamano"].tolist() == [2, 2, 2, 2]


def test_articulation_points_of_a_bowtie():
    points = articulation_table(bowtie(), "moño")
    assert points["node"].tolist() == ["c"]


def test_isolated_node_has_no_centrality_but_stays_in_the_table():
    graph = bowtie()
    graph.add_node("z")
    table = centrality_measures(graph, "moño").set_index("node")
    assert "z" in table.index
    assert table.loc["z", "grado"] == 0
    assert table.loc["z", "intermediacion"] == 0.0
    assert table.loc["z", "componente_tamano"] == 1


def test_removing_an_articulation_point_splits_the_network():
    row = removal_impact(bowtie(), ["c"], "moño").iloc[0]
    assert row["componentes_antes"] == 1
    assert row["componentes_despues"] == 2
    assert row["componentes_nuevos"] == 1
    assert row["componente_mayor_antes"] == 5
    assert row["componente_mayor_despues"] == 2
    assert row["aristas_eliminadas"] == 4
    assert row["cambio_componente_mayor_pct"] == pytest.approx(-60.0)
    assert row["fragmentacion_antes"] == pytest.approx(0.0)
    assert row["fragmentacion_despues"] == pytest.approx(2 / 3)
    assert row["incremento_fragmentacion"] == pytest.approx(2 / 3)
    assert bool(row["es_punto_articulacion"]) is True


def test_removing_a_redundant_node_keeps_the_components():
    row = removal_impact(bowtie(), ["a"], "moño").iloc[0]
    assert row["componentes_despues"] == 1
    assert row["componentes_nuevos"] == 0
    assert row["incremento_fragmentacion"] == pytest.approx(0.0)
    assert bool(row["es_punto_articulacion"]) is False


def test_removing_an_isolated_node_does_not_fragment():
    graph = bowtie()
    graph.add_node("z")
    row = removal_impact(graph, ["z"], "moño").iloc[0]
    assert row["componentes_antes"] == 2
    assert row["componentes_despues"] == 1
    assert row["componentes_nuevos"] == 0
    assert row["incremento_fragmentacion"] == pytest.approx(0.0)


def test_empty_graph_returns_empty_tables():
    graph = nx.Graph()
    assert centrality_measures(graph, "vacío").empty
    assert articulation_table(graph, "vacío").empty
    assert removal_impact(graph, [], "vacío").empty
    assert community_structure(graph, {}, pd.DataFrame()).empty


def test_graph_without_edges_still_produces_rows():
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    table = centrality_measures(graph, "sin aristas")
    assert len(table) == 2
    assert table["intermediacion"].sum() == 0.0
    assert table["pagerank"].sum() == pytest.approx(1.0)


def test_node_order_is_deterministic():
    forward = bowtie()
    backward = nx.Graph()
    for source, target, data in sorted(forward.edges(data=True), reverse=True):
        backward.add_edge(target, source, **data)
    left = centrality_measures(forward, "moño")
    right = centrality_measures(backward, "moño")
    pd.testing.assert_frame_equal(left, right)


def test_results_are_reproducible_across_runs():
    graph = bowtie()
    pd.testing.assert_frame_equal(centrality_measures(graph, "moño"), centrality_measures(graph, "moño"))
    pd.testing.assert_frame_equal(removal_impact(graph, ["a", "c"], "moño"), removal_impact(graph, ["a", "c"], "moño"))


def test_candidates_include_articulation_points_and_leaders():
    graph = bowtie()
    centrality = centrality_measures(graph, "moño")
    candidates = removal_candidates(centrality, articulation_table(graph, "moño"))
    assert candidates == sorted(graph.nodes)
    assert candidates == sorted(set(candidates))


def sample_comments() -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = pd.DataFrame([
        {"video_id": "v1", "channel_id": "c1", "category": "News", "source_query": "q1"},
        {"video_id": "v2", "channel_id": "c2", "category": "People", "source_query": "q2"},
    ])
    comments = pd.DataFrame([
        {"comment_id": "x1", "author_channel_id": "a1", "author_name": "A", "author_handle": "@a", "video_id": "v1"},
        {"comment_id": "x2", "author_channel_id": "a1", "author_name": "A", "author_handle": "@a", "video_id": "v2"},
        {"comment_id": "x3", "author_channel_id": "a2", "author_name": "B", "author_handle": "@b", "video_id": "v1"},
    ])
    return videos, comments


def test_author_profiles_separate_recurrence_from_volume():
    videos, comments = sample_comments()
    profiles = author_profiles(comments, videos).set_index("author_channel_id")
    assert profiles.loc["a1", "videos_comentados"] == 2
    assert profiles.loc["a1", "canales_distintos"] == 2
    assert profiles.loc["a1", "diversidad_canales"] == 1.0
    assert profiles.loc["a1", "entropia_participacion"] == pytest.approx(1.0)
    assert bool(profiles.loc["a1", "es_recurrente"]) is True
    assert bool(profiles.loc["a2", "es_recurrente"]) is False
    assert profiles.loc["a2", "entropia_participacion"] == 0.0


def test_bridges_separate_structural_from_redundant():
    graph = bowtie()
    nx.set_node_attributes(graph, {node: node for node in graph}, "node_id")
    centrality = centrality_measures(graph, "moño")
    points = articulation_table(graph, "moño")
    impact = removal_impact(graph, removal_candidates(centrality, points), "moño")
    profiles = pd.DataFrame({
        "node": sorted(graph.nodes), "label": sorted(graph.nodes), "author_handle": sorted(graph.nodes),
        "comentarios": 1, "videos_comentados": 1,
    })
    bridges = bridge_authors(centrality, profiles, points, impact)
    assert bridges["node"].tolist() == ["c"]
    assert bridges.iloc[0]["tipo_puente"].startswith("puente estructural")
    assert bridges.iloc[0]["componentes_nuevos_si_se_remueve"] == 1


def test_community_structure_flags_dependence_on_few_nodes():
    graph = bowtie()
    membership = {node: 1 for node in graph}
    centrality = centrality_measures(graph, "moño")
    structure = community_structure(graph, membership, centrality).iloc[0]
    assert structure["autores"] == 5
    assert structure["aristas_internas"] == 6
    assert structure["aristas_externas"] == 0
    assert structure["puntos_articulacion_internos"] == 1
    assert structure["caida_componente_mayor_pct"] == pytest.approx(60.0)
    assert bool(structure["dependiente_de_pocos_nodos"]) is True
    assert bool(structure["es_principal"]) is True
