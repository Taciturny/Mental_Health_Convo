import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from src.core.data_loader import DataLoader
from src.core.embeddings_model import EmbeddingsModel
from src.core.qdrant_manager import QdrantManager
from src.core.search_engine import SearchEngine
from src.core.utils import (initialize_qdrant, is_relevant_query,
                            load_and_embed_data)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader("test_data.parquet")

    @patch("pandas.read_parquet")
    def test_get_total_rows(self, mock_read_parquet):
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        mock_read_parquet.return_value = mock_df

        self.assertEqual(self.data_loader.get_total_rows(), 10)

    @patch("pandas.read_parquet")
    def test_load_data_in_batches(self, mock_read_parquet):
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        mock_df.iloc.__getitem__.return_value.to_dict.return_value = [
            {"id": 1, "question": "Test?", "answer": "Answer"}
        ]
        mock_read_parquet.return_value = mock_df

        batches = list(self.data_loader.load_data_in_batches(batch_size=1))
        self.assertEqual(len(batches), 10)
        self.assertEqual(len(batches[0]), 1)
        self.assertEqual(batches[0][0]["id"], 1)


class TestEmbeddingsModel(unittest.TestCase):
    def setUp(self):
        self.embeddings_model = EmbeddingsModel.get_instance()

    @patch("src.core.embeddings_model.TextEmbedding")
    @patch("src.core.embeddings_model.LateInteractionTextEmbedding")
    def test_embeddings(self, mock_late, mock_dense):
        mock_dense_output = np.array(
            [
                0.025,
                0.019,
                -0.059,
                0.006,
                -0.060,
                0.033,
                0.018,
                0.007,
                0.056,
                0.073,
                0.030,
                -0.069,
            ],
            dtype=np.float32,
        )
        mock_late_output = np.array(
            [[0.100, 0.013, -0.109, -0.023, 0.098, 0.128, -0.038, 0.003]],
            dtype=np.float32,
        )

        mock_dense.return_value.embed.return_value = [mock_dense_output]
        mock_late.return_value.embed.return_value = [mock_late_output]

        # Patch the instance attributes
        self.embeddings_model.dense_embedding_model = mock_dense.return_value
        self.embeddings_model.late_interaction_embedding_model = mock_late.return_value

        inputs = ["Test input"]
        dense, late = self.embeddings_model.embeddings(inputs)

        np.testing.assert_array_almost_equal(dense[0], mock_dense_output, decimal=3)
        np.testing.assert_array_almost_equal(late[0], mock_late_output, decimal=3)

        mock_dense.return_value.embed.assert_called_once_with(inputs)
        mock_late.return_value.embed.assert_called_once_with(inputs)


class TestQdrantManager(unittest.TestCase):

    def setUp(self):
        self.host = "localhost"
        self.port = 6333
        self.url = "http://example.com"
        self.api_key = "test_api_key"

    @patch("src.core.qdrant_manager.QdrantClient")
    def test_init_with_host_port(self, mock_qdrant_client):
        _ = initialize_qdrant("test_collection")

        # Ensure that QdrantManager was initialized with localhost and the correct port
        mock_qdrant_client.assert_called_once_with(
            host="localhost", port=6333, timeout=600
        )

    @patch("src.core.qdrant_manager.QdrantClient")
    @patch("src.core.utils.settings")
    def test_init_with_url_api_key(self, mock_settings, mock_qdrant_client):
        # Set mock values for settings
        mock_settings.DEPLOYMENT_MODE = "cloud"
        mock_settings.QDRANT_URL = "http://example.com"
        mock_settings.QDRANT_API_KEY = "test_api_key"

        # Call the function that initializes QdrantManager in cloud mode
        _ = initialize_qdrant("test_collection")

        # Ensure QdrantClient was initialized with the correct URL and API key
        mock_qdrant_client.assert_called_once_with(
            url=mock_settings.QDRANT_URL,
            api_key=mock_settings.QDRANT_API_KEY,
            timeout=600,
        )

    @patch("src.core.qdrant_manager.QdrantClient")
    def test_create_collection(self, mock_qdrant_client):
        manager = QdrantManager(self.host, self.port)
        collection_name = "test_collection"
        manager.create_collection(collection_name)

        mock_qdrant_client().create_collection.assert_called_once()
        call_args = mock_qdrant_client().create_collection.call_args
        self.assertEqual(call_args[0][0], collection_name)
        self.assertIn("vectors_config", call_args[1])

    @patch("src.core.qdrant_manager.QdrantClient")
    def test_collection_exists(self, mock_qdrant_client):
        manager = QdrantManager(self.host, self.port)
        collection_name = "test_collection"

        # Mock the get_collections method
        mock_collection = MagicMock()
        mock_collection.name = collection_name
        mock_qdrant_client().get_collections.return_value.collections = [
            mock_collection
        ]

        self.assertTrue(manager.collection_exists(collection_name))
        self.assertFalse(manager.collection_exists("non_existent_collection"))

    @patch("src.core.qdrant_manager.QdrantClient")
    @patch("src.core.qdrant_manager.QdrantManager.collection_exists")
    @patch("src.core.qdrant_manager.QdrantManager.create_collection")
    def test_create_collection_if_not_exists(
        self,
        mock_create_collection,
        mock_collection_exists,
        mock_qdrant_client,
    ):
        manager = QdrantManager(self.host, self.port)
        collection_name = "test_collection"

        # Test when collection doesn't exist
        mock_collection_exists.return_value = False
        manager.create_collection_if_not_exists(collection_name)
        mock_create_collection.assert_called_once_with(collection_name)

        # Test when collection already exists
        mock_collection_exists.return_value = True
        mock_create_collection.reset_mock()
        manager.create_collection_if_not_exists(collection_name)
        mock_create_collection.assert_not_called()


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.search_engine = SearchEngine("test_collection")

    @patch("src.core.search_engine.QdrantClient")
    @patch("src.core.search_engine.EmbeddingsModel")
    def test_search_dense(self, mock_embeddings, mock_client):
        mock_embeddings.get_instance.return_value.embeddings.return_value = (
            [np.array([1, 2, 3])],
            None,
        )
        mock_client.return_value.query_points.return_value = MagicMock()

        # Replace the search engine's client with the mock
        self.search_engine.client = mock_client.return_value

        _ = self.search_engine.search_dense("test query")

        mock_client.return_value.query_points.assert_called_once()

    @patch("src.core.search_engine.QdrantClient")
    @patch("src.core.search_engine.EmbeddingsModel")
    def test_search_hybrid(self, mock_embeddings, mock_client):
        mock_embeddings.get_instance.return_value.embeddings.return_value = (
            [np.array([1, 2, 3])],
            [np.array([4, 5, 6])],
        )
        mock_client.return_value.query_points.return_value = MagicMock()

        # Replace the search engine's client with the mock
        self.search_engine.client = mock_client.return_value

        _ = self.search_engine.search_hybrid("test query")

        mock_client.return_value.query_points.assert_called_once()


class TestUtils(unittest.TestCase):

    @patch("src.core.utils.QdrantManager")
    def test_initialize_qdrant(self, mock_qdrant_manager):
        qdrant_manager = initialize_qdrant("test_collection")
        mock_qdrant_manager.return_value.create_collection_if_not_exists.assert_called_once_with(
            "test_collection"
        )
        self.assertIsNotNone(qdrant_manager)

    @patch("src.core.utils.DataLoader")
    @patch("src.core.utils.EmbeddingsModel")
    def test_load_and_embed_data(self, mock_embeddings_model, mock_data_loader):
        mock_data_loader.return_value.load_data_in_batches.return_value = [
            [{"question": "Test?"}]
        ]
        mock_embeddings_model.get_instance.return_value.embeddings.return_value = (
            [np.array([1, 2, 3])],
            [np.array([4, 5, 6])],
        )

        data, dense_embeddings, late_embeddings = load_and_embed_data(
            "test_file.parquet"
        )

        self.assertEqual(len(data), 1)
        self.assertEqual(len(dense_embeddings), 1)
        self.assertEqual(len(late_embeddings), 1)

    @patch("src.core.utils.sentiment_analyzer")
    def test_keyword_match(self, mock_sentiment_analyzer):
        # Test exact keyword match
        relevant_keywords = ["depressed", "anxiety", "treatment"]
        self.assertTrue(is_relevant_query("I'm feeling depressed", relevant_keywords))
        self.assertTrue(
            is_relevant_query("Looking for anxiety treatment", relevant_keywords)
        )

        # Test non-matching query
        self.assertFalse(
            is_relevant_query("What's the weather like today?", relevant_keywords)
        )

    @patch("src.core.utils.sentiment_analyzer")
    def test_sentiment_analysis(self, mock_sentiment_analyzer):
        # Mock sentiment analyzer for negative sentiment
        mock_sentiment_analyzer.return_value = [{"label": "NEGATIVE", "score": 0.8}]
        relevant_keywords = ["depressed", "anxiety", "treatment"]
        self.assertTrue(
            is_relevant_query("I feel terrible and hopeless", relevant_keywords)
        )

        # Mock sentiment analyzer for positive sentiment
        mock_sentiment_analyzer.return_value = [{"label": "POSITIVE", "score": 0.9}]
        self.assertFalse(is_relevant_query("I feel great today!", relevant_keywords))

        # Mock sentiment analyzer for negative sentiment but low score
        mock_sentiment_analyzer.return_value = [{"label": "NEGATIVE", "score": 0.6}]
        self.assertFalse(
            is_relevant_query("The weather is a bit gloomy", relevant_keywords)
        )


if __name__ == "__main__":
    unittest.main()
