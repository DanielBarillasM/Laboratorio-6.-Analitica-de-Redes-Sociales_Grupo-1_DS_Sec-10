import pandas as pd

from lab6_social.networks import build_bipartite_network, build_projections, graph_metrics


def sample_data():
    videos = pd.DataFrame([
        {"video_id": "v1", "title": "Uno", "channel_id": "c1", "channel_name": "Canal", "category": "News", "view_count": 10},
        {"video_id": "v2", "title": "Dos", "channel_id": "c1", "channel_name": "Canal", "category": "News", "view_count": 20},
        {"video_id": "v3", "title": "Tres", "channel_id": "c2", "channel_name": "Otro", "category": "People", "view_count": 5},
    ])
    comments = pd.DataFrame([
        {"comment_id": "x1", "author_channel_id": "a1", "author_name": "A", "author_handle": "@a", "video_id": "v1"},
        {"comment_id": "x2", "author_channel_id": "a1", "author_name": "A", "author_handle": "@a", "video_id": "v1"},
        {"comment_id": "x3", "author_channel_id": "a1", "author_name": "A", "author_handle": "@a", "video_id": "v2"},
        {"comment_id": "x4", "author_channel_id": "a2", "author_name": "B", "author_handle": "@b", "video_id": "v1"},
    ])
    return videos, comments


def test_bipartite_edge_weight_counts_comments():
    graph, nodes, edges = build_bipartite_network(*sample_data())
    assert graph["author::a1"]["video::v1"]["weight"] == 2
    assert len(nodes) == 5
    assert len(edges) == 3


def test_uncommented_video_is_preserved_as_isolate():
    graph, _, _ = build_bipartite_network(*sample_data())
    assert graph.degree("video::v3") == 0


def test_projection_weights_have_required_meaning():
    graph, _, _ = build_bipartite_network(*sample_data())
    authors, videos = build_projections(graph)
    assert authors["author::a1"]["author::a2"]["weight"] == 1
    assert videos["video::v1"]["video::v2"]["weight"] == 1


def test_metrics_include_isolates_and_components():
    graph, _, _ = build_bipartite_network(*sample_data())
    metrics = graph_metrics(graph, "test")
    assert metrics["nodos_aislados"] == 1
    assert metrics["componentes"] == 2

