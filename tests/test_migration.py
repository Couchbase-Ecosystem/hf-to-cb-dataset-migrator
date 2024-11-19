# test_migration.py

import unittest
from unittest.mock import patch, MagicMock, Mock
from hf_to_cb_dataset_migrator.migration import DatasetMigrator
from couchbase.exceptions import CouchbaseException, DocumentExistsException
from datasets import DatasetDict, Dataset, Split
from datasets.features import Features
from datasets.utils.version import Version
from typing import List, Dict, Any


class TestDatasetMigrator(unittest.TestCase):
    def setUp(self):
        # Initialize the DatasetMigrator with a token
        self.token = "test_token"
        self.migrator = DatasetMigrator(token=self.token)

    @patch('hf_to_cb_dataset_migrator.migration.get_dataset_config_names')
    def test_list_configs(self, mock_get_configs):
        # Mock the return value of get_dataset_config_names
        mock_get_configs.return_value = ['config1', 'config2']

        configs = self.migrator.list_configs(
            path='test_dataset',
            revision='1.0.0',
            download_config=None,
            download_mode=None,
            dynamic_modules_path=None,
            data_files=None,
        )

        self.assertEqual(configs, ['config1', 'config2'])
        mock_get_configs.assert_called_once()

    @patch('hf_to_cb_dataset_migrator.migration.get_dataset_split_names')
    def test_list_splits(self, mock_get_splits):
        # Mock the return value of get_dataset_split_names
        mock_get_splits.return_value = ['train', 'test', 'validation']

        splits = self.migrator.list_splits(
            path='test_dataset',
            config_name='config1',
            data_files=None,
            download_config=None,
            download_mode=None,
            revision='1.0.0',
        )

        self.assertEqual(splits, ['train', 'test', 'validation'])
        mock_get_splits.assert_called_once()

    @patch('hf_to_cb_dataset_migrator.migration.load_dataset')
    @patch.object(DatasetMigrator, 'connect')
    @patch.object(DatasetMigrator, 'close')
    @patch.object(DatasetMigrator, 'insert_multi')
    def test_migrate_dataset(
        self, mock_insert_multi, mock_close, mock_connect, mock_load_dataset
    ):
        # Mock dataset
        mock_dataset = Dataset.from_dict({
            'id': [1, 2],
            'text': ['Hello', 'World']
        })

        mock_load_dataset.return_value = mock_dataset

        result = self.migrator.migrate_dataset(
            path='test_dataset',
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            couchbase_bucket='bucket',
            cb_scope=None,
            cb_collection=None,
            id_fields='id',
            name=None,
            data_dir=None,
            data_files=None,
            split=None,
            cache_dir=None,
            features=None,
            download_config=None,
            download_mode=None,
            verification_mode=None,
            keep_in_memory=None,
            save_infos=False,
            revision='1.0.0',
            streaming=False,
            num_proc=None,
            storage_options=None,
            trust_remote_code=None,
            batch_size=1,
        )

        self.assertTrue(result)
        mock_connect.assert_called_once()
        mock_close.assert_called_once()
        mock_insert_multi.assert_called()
        mock_load_dataset.assert_called_once()

    @patch('hf_to_cb_dataset_migrator.migration.load_dataset')
    @patch.object(DatasetMigrator, 'connect')
    @patch.object(DatasetMigrator, 'close')
    @patch.object(DatasetMigrator, 'insert_multi')
    def test_migrate_dataset_exception(
        self, mock_insert_multi, mock_close, mock_connect, mock_load_dataset
    ):
        # Simulate an exception during migration
        mock_insert_multi.side_effect = CouchbaseException(message="Test Exception")

        # Mock dataset
        mock_dataset = Dataset.from_dict({
            'id': [1, 2],
            'text': ['Hello', 'World']
        })
        mock_load_dataset.return_value = mock_dataset

        with self.assertLogs('hf_to_cb_dataset_migrator.migration', level='ERROR') as cm:
            result = self.migrator.migrate_dataset(
                path='test_dataset',
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                couchbase_bucket='bucket',
                cb_scope=None,
                cb_collection=None,
                id_fields='id',
                name=None,
                data_dir=None,
                data_files=None,
                split=None,
                cache_dir=None,
                features=None,
                download_config=None,
                download_mode=None,
                verification_mode=None,
                keep_in_memory=None,
                save_infos=False,
                revision='1.0.0',
                streaming=False,
                num_proc=None,
                storage_options=None,
                trust_remote_code=None,
                batch_size=1,
            )

            self.assertFalse(result)
            self.assertIn("An error occurred during migration", cm.output[0])

    def test_insert_multi_success(self):
        # Mock insert_multi to simulate successful insertion
        self.migrator.collection = MagicMock()
        # Mock insert_multi to simulate successful insertion
        mock_result = MagicMock()
        mock_result.all_ok = True
        mock_result.exceptions = {}
        self.migrator.collection.insert_multi.return_value = mock_result

        batch = {
            'doc1': {'id': 1, 'text': 'Hello'},
            'doc2': {'id': 2, 'text': 'World'}
        }

        self.migrator.insert_multi(batch)
        self.migrator.collection.insert_multi.assert_called_once_with(batch)

    def test_insert_multi_failure(self):
        # Mock collection
        self.migrator.collection = MagicMock()
        # Mock insert_multi to simulate failure
        mock_result = MagicMock()
        mock_result.all_ok = False
        mock_result.exceptions = {
            'doc1': DocumentExistsException(),
            'doc2': CouchbaseException(message='Insert failed')
        }
        self.migrator.collection.insert_multi.return_value = mock_result

        batch = {
            'doc1': {'id': 1, 'text': 'Hello'},
            'doc2': {'id': 2, 'text': 'World'}
        }

        with self.assertRaises(Exception) as context:
            self.migrator.insert_multi(batch)

        self.assertIn("Failed to write some documents to Couchbase", str(context.exception))
        self.migrator.collection.insert_multi.assert_called_once_with(batch)

    @patch('hf_to_cb_dataset_migrator.migration.Cluster')
    def test_connect_success(self, mock_cluster_class):
        # Mock cluster connection
        mock_cluster = MagicMock()
        mock_cluster_class.return_value = mock_cluster
        mock_bucket = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket
        mock_bucket.default_collection.return_value = MagicMock()

        self.migrator.connect(
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            couchbase_bucket='bucket',
            cb_scope=None,
            cb_collection=None,
        )

        mock_cluster_class.assert_called_once()
        mock_cluster.wait_until_ready.assert_called_once()
        mock_cluster.bucket.assert_called_once_with('bucket')

    @patch('hf_to_cb_dataset_migrator.migration.Cluster')
    def test_connect_failure(self, mock_cluster_class):
        # Simulate connection failure
        mock_cluster_class.side_effect = CouchbaseException(message='Connection failed')

        with self.assertRaises(CouchbaseException):
            self.migrator.connect(
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                couchbase_bucket='bucket',
                cb_scope=None,
                cb_collection=None,
            )

    def test_close(self):
        # Test close method when cluster is connected
        self.migrator.cluster = MagicMock()
        self.migrator.close()
        self.assertIsNone(self.migrator.cluster)
        self.assertIsNone(self.migrator.collection)

    def test_close_without_connection(self):
        # Test close method when cluster is not connected
        self.migrator.cluster = None
        self.migrator.close()  # Should not raise any exception
        self.assertIsNone(self.migrator.cluster)
        self.assertIsNone(self.migrator.collection)


if __name__ == '__main__':
    unittest.main()