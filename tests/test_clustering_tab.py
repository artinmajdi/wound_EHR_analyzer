import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import streamlit as st
import plotly.graph_objects as go

from wound_analysis.dashboard_components.clustering_tab import ClusteringTab
from wound_analysis.utils.data_processor import WoundDataProcessor
from wound_analysis.utils.column_schema import DColumns

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'wound_area': [10, 20, 30, 40],
        'healing_rate': [0.1, 0.2, 0.3, 0.4],
        'highest_freq_absolute': [100, 200, 300, 400],
        'patient_id': ['P1', 'P1', 'P2', 'P2']
    })

@pytest.fixture
def mock_wound_processor(sample_data):
    """Create a mock WoundDataProcessor"""
    processor = MagicMock(spec=WoundDataProcessor)
    processor.df = sample_data
    return processor

@pytest.fixture
def clustering_tab(mock_wound_processor):
    """Create a ClusteringTab instance for testing"""
    return ClusteringTab(mock_wound_processor, selected_patient='P1')

def test_init(clustering_tab, sample_data):
    """Test initialization of ClusteringTab"""
    assert clustering_tab.df.equals(sample_data)
    assert isinstance(clustering_tab.CN, DColumns)
    assert clustering_tab.selected_patient == 'P1'
    assert clustering_tab.df_w_cluster_tags is None
    assert clustering_tab.selected_cluster is None
    assert clustering_tab.use_cluster_data is False

def test_get_cluster_analysis_settings(clustering_tab, monkeypatch):
    """Test cluster analysis settings with mocked streamlit inputs"""
    # Mock streamlit inputs
    def mock_multiselect(*args, **kwargs):
        return ['wound_area', 'healing_rate']

    def mock_number_input(*args, **kwargs):
        return 3

    def mock_selectbox(*args, **kwargs):
        return "K-Means"

    def mock_button(*args, **kwargs):
        return True

    monkeypatch.setattr(st, "multiselect", mock_multiselect)
    monkeypatch.setattr(st, "number_input", mock_number_input)
    monkeypatch.setattr(st, "selectbox", mock_selectbox)
    monkeypatch.setattr(st, "button", mock_button)

    settings = clustering_tab.get_cluster_analysis_settings()
    assert settings["n_clusters"] == 3
    assert settings["clustering_method"] == "K-Means"
    assert settings["run_clustering"] is True
    assert len(settings["cluster_features"]) > 0

def test_get_updated_df_no_clustering(clustering_tab, sample_data):
    """Test get_updated_df when no clustering has been performed"""
    result = clustering_tab.get_updated_df()
    assert result.equals(sample_data)

def test_get_updated_df_with_clustering(clustering_tab):
    """Test get_updated_df with clustering results"""
    # Mock clustering results
    clustered_df = clustering_tab.df.copy()
    clustered_df['Cluster'] = [0, 0, 1, 1]

    st.session_state.df_w_cluster_tags = clustered_df
    st.session_state.selected_cluster = 0
    clustering_tab.use_cluster_data = True

    result = clustering_tab.get_updated_df()
    assert len(result) == 2  # Only cluster 0 data
    assert all(result['Cluster'] == 0)

@pytest.mark.parametrize("empty_data", [
    pd.DataFrame(),  # Empty DataFrame
    pd.DataFrame({'wound_area': [], 'healing_rate': []})  # DataFrame with empty columns
])
def test_empty_dataset(empty_data):
    """Test behavior with empty dataset"""
    processor = MagicMock(spec=WoundDataProcessor)
    processor.df = empty_data

    tab = ClusteringTab(processor, selected_patient='P1')
    assert tab.df.empty

    # Mock session state to be empty
    st.session_state.df_w_cluster_tags = None
    result = tab.get_updated_df()
    assert result.empty

def test_missing_values(clustering_tab):
    """Test behavior with missing values"""
    df_with_na = clustering_tab.df.copy()
    df_with_na.loc[0, 'wound_area'] = np.nan

    processor = MagicMock(spec=WoundDataProcessor)
    processor.df = df_with_na

    tab = ClusteringTab(processor, selected_patient='P1')
    assert tab.df.isna().sum().sum() > 0  # Verify we have NaN values
    assert len(tab.get_updated_df()) == len(df_with_na)

@patch('streamlit.plotly_chart')
def test_display_cluster_distribution(mock_plotly_chart, clustering_tab):
    """Test cluster distribution display"""
    df_with_clusters = clustering_tab.df.copy()
    df_with_clusters['Cluster'] = [0, 0, 1, 1]

    # Create a mock figure that will be accepted by plotly
    mock_figure = go.Figure()
    with patch('plotly.express.bar', return_value=mock_figure):
        ClusteringTab._display_cluster_distribution(df_with_clusters)
        mock_plotly_chart.assert_called_once()

@patch('streamlit.plotly_chart')
def test_display_feature_importance(mock_plotly_chart, clustering_tab):
    """Test feature importance display"""
    feature_importance = {
        'wound_area': 0.5,
        'healing_rate': 0.3,
        'highest_freq_absolute': 0.2
    }

    # Create a mock figure that will be accepted by plotly
    mock_figure = go.Figure()
    with patch('plotly.graph_objects.Figure', return_value=mock_figure):
        ClusteringTab._display_feature_importance(feature_importance)
        mock_plotly_chart.assert_called_once()

def test_calculate_shap_values(clustering_tab):
    """Test SHAP value calculation"""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    features = ['feature1', 'feature2']

    with patch('shap.KernelExplainer') as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_explainer.return_value.expected_value = np.array([0.5, 0.5])

        shap_values, expected_values = clustering_tab._calculate_shap_values(df, features)

        assert isinstance(shap_values, np.ndarray)
        assert isinstance(expected_values, np.ndarray)
        assert shap_values.shape == (3, 2)  # 3 samples, 2 features

@patch('streamlit.plotly_chart')
@patch('streamlit.markdown')
def test_display_shap_analysis_all_data(mock_markdown, mock_plotly_chart, clustering_tab):
    """Test SHAP analysis display for 'All Data' view"""
    # Use real feature names from sample_data fixture
    features = ['wound_area', 'healing_rate', 'highest_freq_absolute']
    st.session_state.selected_cluster = "All Data"
    st.session_state.df_w_cluster_tags = clustering_tab.df.copy()
    st.session_state.shap_values = np.array([
        [0.1, 0.2, 0.3],
        [0.3, 0.4, 0.5],
        [0.5, 0.6, 0.7],
        [0.7, 0.8, 0.9]
    ])

    # Mock cluster settings
    clustering_tab._cluster_settings = {
        "cluster_features": features
    }

    # Call the method
    clustering_tab._display_shap_analysis()

    # Verify that plotly_chart was called with a figure
    mock_plotly_chart.assert_any_call(mock_plotly_chart.call_args[0][0], use_container_width=True)

    # Verify that appropriate markdown explanations were shown
    assert any("Overall SHAP Value Distribution" in str(call)
              for call in mock_markdown.call_args_list)

@patch('streamlit.plotly_chart')
@patch('streamlit.markdown')
def test_display_shap_analysis_specific_cluster(mock_markdown, mock_plotly_chart, clustering_tab):
    """Test SHAP analysis display for specific cluster view"""
    features = ['wound_area', 'healing_rate', 'highest_freq_absolute']
    st.session_state.selected_cluster = 0
    st.session_state.df_w_cluster_tags = pd.DataFrame({
        'wound_area': [10, 20, 30, 40],
        'healing_rate': [0.1, 0.2, 0.3, 0.4],
        'highest_freq_absolute': [100, 200, 300, 400],
        'Cluster': [0, 0, 1, 1]
    })
    st.session_state.shap_values = np.array([
        [0.1, 0.2, 0.3],
        [0.3, 0.4, 0.5],
        [0.5, 0.6, 0.7],
        [0.7, 0.8, 0.9]
    ])

    # Mock cluster settings
    clustering_tab._cluster_settings = {
        "cluster_features": features
    }

    # Call the method
    clustering_tab._display_shap_analysis()

    # Verify that plotly_chart was called with a figure
    mock_plotly_chart.assert_any_call(mock_plotly_chart.call_args[0][0], use_container_width=True)

    # Verify that appropriate markdown explanations were shown
    assert any(f"SHAP Values for Cluster {st.session_state.selected_cluster}" in str(call)
              for call in mock_markdown.call_args_list)

@patch('streamlit.info')
def test_display_shap_analysis_no_cluster(mock_info, clustering_tab):
    """Test SHAP analysis display when no cluster is selected"""
    # Mock session state with no cluster selected
    st.session_state.selected_cluster = None
    clustering_tab._cluster_settings = None

    # Call the method
    clustering_tab._display_shap_analysis()

    # Verify that info message was shown
    mock_info.assert_called_once_with("Please run clustering first to view SHAP analysis")

@patch('streamlit.info')
def test_display_shap_analysis_no_selection(mock_info, clustering_tab):
    """Test SHAP analysis display when cluster settings exist but no cluster is selected"""
    # Use real feature names from sample_data fixture
    features = ['wound_area', 'healing_rate', 'highest_freq_absolute']
    st.session_state.selected_cluster = None
    st.session_state.df_w_cluster_tags = clustering_tab.df.copy()
    st.session_state.shap_values = np.array([
        [0.1, 0.2, 0.3],
        [0.3, 0.4, 0.5],
        [0.5, 0.6, 0.7],
        [0.7, 0.8, 0.9]
    ])
    clustering_tab._cluster_settings = {
        "cluster_features": features
    }

    # Call the method
    clustering_tab._display_shap_analysis()

    # Verify that info message was shown
    mock_info.assert_called_once_with("Please select a cluster to view SHAP analysis")
