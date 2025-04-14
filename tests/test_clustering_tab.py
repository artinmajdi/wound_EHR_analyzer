import pandas as pd
import pytest
import streamlit as st
from wound_analysis.dashboard_components.clustering_tab import ClusteringTab


def setup_session_state():
    if not hasattr(st, "session_state"):
        st.session_state = {}


def test_get_df_for_specific_cluster_all_data():
    setup_session_state()
    # Create a dummy dataframe with a 'Cluster' column
    df = pd.DataFrame({
        "value": [10, 20, 30],
        "Cluster": [0, 1, 0]
    })
    ct = ClusteringTab(df)
    st.session_state.df_w_cluster_tags = df
    # When no specific cluster is passed, it should return all data
    result = ct._get_df_for_specific_cluster()
    pd.testing.assert_frame_equal(result, df)
    assert st.session_state.get("selected_cluster") is None


def test_get_df_for_specific_cluster_specific():
    setup_session_state()
    # Create a dummy dataframe with a 'Cluster' column
    df = pd.DataFrame({
        "value": [10, 20, 30, 40],
        "Cluster": [0, 0, 1, 1]
    })
    ct = ClusteringTab(df)
    st.session_state.df_w_cluster_tags = df
    cluster_id = 1
    result = ct._get_df_for_specific_cluster(_cluster_id=cluster_id)
    expected = df[df["Cluster"] == cluster_id].copy()
    pd.testing.assert_frame_equal(result, expected)
    assert st.session_state.get("selected_cluster") == cluster_id


def test_get_cluster_df_integration(monkeypatch):
    setup_session_state()
    df = pd.DataFrame({
        "value": [100, 200, 300, 400],
        "Cluster": [1, 1, 0, 0]
    })
    ct = ClusteringTab(df)
    st.session_state.df_w_cluster_tags = df.copy()
    # Monkey-patch _get_user_selected_cluster to simulate user selecting cluster 1
    monkeypatch.setattr(ct, "_get_user_selected_cluster", lambda: 1)
    result = ct.get_cluster_df()
    expected = df[df["Cluster"] == 1].copy()
    pd.testing.assert_frame_equal(result, expected)
    assert ct._cluster_id == 1


def test_get_df_for_specific_cluster_no_data():
    setup_session_state()
    st.session_state.df_w_cluster_tags = None
    df = pd.DataFrame({
        "value": [10, 20],
        "Cluster": [0, 1]
    })
    ct = ClusteringTab(df)
    result = ct._get_df_for_specific_cluster()
    pd.testing.assert_frame_equal(result, pd.DataFrame())
