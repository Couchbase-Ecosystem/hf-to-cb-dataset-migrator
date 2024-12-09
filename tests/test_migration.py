# test_migration.py

import unittest
from unittest.mock import patch, MagicMock, Mock, call
from hf_to_cb_dataset_migrator.migration import DatasetMigrator
from couchbase.exceptions import (
    CouchbaseException,
    DocumentExistsException,
    ScopeAlreadyExistsException,
    CollectionAlreadyExistsException,
)
from datasets import Dataset, DatasetDict
from typing import Any


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
    @patch.object(DatasetMigrator, 'upsert_multi')
    def test_migrate_dataset(
        self, mock_upsert_multi, mock_close, mock_connect, mock_load_dataset
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
            cb_bucket='bucket',
            cb_scope='test_scope',
            cb_collection='test_collection',
            id_fields='id',
            name=None,
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

        self.assertIsNone(result)
        mock_connect.assert_called_once()
        mock_close.assert_called_once()
        mock_upsert_multi.assert_called()
        mock_load_dataset.assert_called_once()

    @patch('hf_to_cb_dataset_migrator.migration.load_dataset')
    @patch.object(DatasetMigrator, 'connect')
    @patch.object(DatasetMigrator, 'close')
    @patch.object(DatasetMigrator, 'upsert_multi')
    def test_migrate_dataset_exception(
        self, mock_upsert_multi, mock_close, mock_connect, mock_load_dataset
    ):
        # Simulate an exception during migration
        mock_upsert_multi.side_effect = CouchbaseException(message="Test Exception")

        # Mock dataset
        mock_dataset = Dataset.from_dict({
            'id': [1, 2],
            'text': ['Hello', 'World']
        })
        mock_load_dataset.return_value = mock_dataset

        with self.assertRaises(Exception) as cm:
            result = self.migrator.migrate_dataset(
                path='test_dataset',
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                cb_bucket='bucket',
                cb_scope='test_scope',
                cb_collection='test_collection',
                id_fields='id',
                batch_size=1
            )

            self.assertFalse(result)
            self.assertIn("Error processing split", cm.exception)

    def test_upsert_multi_success(self):
        # Mock upsert_multi to simulate successful insertion
        self.migrator.collection = MagicMock()
        # Mock upsert_multi to simulate successful insertion
        mock_result = MagicMock()
        mock_result.all_ok = True
        mock_result.exceptions = {}
        self.migrator.collection.upsert_multi.return_value = mock_result

        batch = {
            'doc1': {'id': 1, 'text': 'Hello'},
            'doc2': {'id': 2, 'text': 'World'}
        }

        self.migrator.upsert_multi(batch)
        self.migrator.collection.upsert_multi.assert_called_once_with(batch)

    def test_upsert_multi_failure(self):
        # Mock collection
        self.migrator.collection = MagicMock()
        # Mock upsert_multi to simulate failure
        mock_result = MagicMock()
        mock_result.all_ok = False
        mock_result.exceptions = {
            'doc1': DocumentExistsException(),
            'doc2': CouchbaseException(message='Upsert failed')
        }
        self.migrator.collection.upsert_multi.return_value = mock_result

        batch = {
            'doc1': {'id': 1, 'text': 'Hello'},
            'doc2': {'id': 2, 'text': 'World'}
        }

        with self.assertRaises(Exception) as context:
            self.migrator.upsert_multi(batch)

        self.assertIn("Failed to write some documents to Couchbase", str(context.exception))
        self.migrator.collection.upsert_multi.assert_called_once_with(batch)

    @patch('hf_to_cb_dataset_migrator.migration.Cluster')
    def test_connect_success(self, mock_cluster_class):
        # Mock cluster connection
        mock_cluster = MagicMock()
        mock_cluster_class.return_value = mock_cluster
        mock_bucket = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket

        # Mock collection manager and its methods
        mock_collection_manager = MagicMock()
        mock_bucket.collections.return_value = mock_collection_manager

        # Simulate that scope and collection do not exist initially
        # Use a function for side_effect to simulate multiple calls
        mock_scope = MagicMock()
        mock_scope.name = 'test_scope'
        mock_scope.collections = []
        mock_collection = MagicMock()
        mock_collection.name = 'test_collection'

        def get_all_scopes_side_effect():
            if not hasattr(self, 'get_all_scopes_call_count'):
                self.get_all_scopes_call_count = 0
            self.get_all_scopes_call_count += 1

            if self.get_all_scopes_call_count == 1:
                # Before scope creation
                return []
            elif self.get_all_scopes_call_count == 2:
                # After scope creation
                return [mock_scope]
            elif self.get_all_scopes_call_count == 3:
                # Before collection creation
                return [mock_scope]
            else:
                # After collection creation
                mock_scope.collections = [mock_collection]
                return [mock_scope]

        mock_collection_manager.get_all_scopes.side_effect = get_all_scopes_side_effect

        # Call the connect method
        self.migrator.connect(
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            cb_bucket='bucket',
            cb_scope='test_scope',
            cb_collection='test_collection',
        )

        # Assertions
        mock_cluster_class.assert_called_once()
        mock_cluster.wait_until_ready.assert_called_once()
        mock_cluster.bucket.assert_called_once_with('bucket')
        mock_bucket.collections.assert_called_once()
        mock_collection_manager.create_scope.assert_called_once_with('test_scope')

        # Since the collection did not exist, create_collection should be called
        mock_collection_manager.create_collection.assert_called_once()
        collection_spec = mock_collection_manager.create_collection.call_args[0][0]
        self.assertEqual(collection_spec.scope_name, 'test_scope')
        self.assertEqual(collection_spec.name, 'test_collection')

    @patch('hf_to_cb_dataset_migrator.migration.Cluster')
    def test_connect_create_scope_and_collection_exceptions(self, mock_cluster_class):
        # Mock cluster connection
        mock_cluster = MagicMock()
        mock_cluster_class.return_value = mock_cluster
        mock_bucket = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket

        # Mock collection manager and its methods
        mock_collection_manager = MagicMock()
        mock_bucket.collections.return_value = mock_collection_manager

        # Mock scope and collection
        mock_scope = MagicMock()
        mock_scope.name = 'test_scope'
        mock_scope.collections = []
        mock_collection = MagicMock()
        mock_collection.name = 'test_collection'

        def get_all_scopes_side_effect():
            if not hasattr(self, 'get_all_scopes_call_count'):
                self.get_all_scopes_call_count = 0
            self.get_all_scopes_call_count += 1

            if self.get_all_scopes_call_count == 1:
                # Before scope creation
                return []
            elif self.get_all_scopes_call_count == 2:
                # After scope creation (even though it raises exception)
                return [mock_scope]
            elif self.get_all_scopes_call_count == 3:
                # Before collection creation
                return [mock_scope]
            else:
                # After collection creation (even though it raises exception)
                mock_scope.collections = [mock_collection]
                return [mock_scope]

        mock_collection_manager.get_all_scopes.side_effect = get_all_scopes_side_effect

        # Simulate exceptions when creating scope and collection
        mock_collection_manager.create_scope.side_effect = ScopeAlreadyExistsException()
        mock_collection_manager.create_collection.side_effect = CollectionAlreadyExistsException()

        # Call the connect method
        self.migrator.connect(
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            cb_bucket='bucket',
            cb_scope='test_scope',
            cb_collection='test_collection',
        )

        # Assertions
        mock_collection_manager.create_scope.assert_called_once_with('test_scope')
        mock_collection_manager.create_collection.assert_called_once()
        collection_spec = mock_collection_manager.create_collection.call_args[0][0]
        self.assertEqual(collection_spec.scope_name, 'test_scope')
        self.assertEqual(collection_spec.name, 'test_collection')

    @patch('hf_to_cb_dataset_migrator.migration.Cluster')
    def test_connect_scope_and_collection_exist(self, mock_cluster_class):
        # Mock cluster connection
        mock_cluster = MagicMock()
        mock_cluster_class.return_value = mock_cluster
        mock_bucket = MagicMock()
        mock_cluster.bucket.return_value = mock_bucket

        # Mock collection manager and its methods
        mock_collection_manager = MagicMock()
        mock_bucket.collections.return_value = mock_collection_manager

        # Simulate that scope and collection already exist
        mock_collection = MagicMock()
        mock_collection.name = 'test_collection'
        mock_scope = MagicMock()
        mock_scope.name = 'test_scope'
        mock_scope.collections = [mock_collection]
        mock_collection_manager.get_all_scopes.return_value = [mock_scope]

        # Call the connect method
        self.migrator.connect(
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            cb_bucket='bucket',
            cb_scope='test_scope',
            cb_collection='test_collection',
        )

        # Assertions
        mock_cluster_class.assert_called_once()
        mock_cluster.wait_until_ready.assert_called_once()
        mock_cluster.bucket.assert_called_once_with('bucket')
        mock_bucket.collections.assert_called_once()

        # Since the scope and collection exist, create_scope and create_collection should not be called
        mock_collection_manager.create_scope.assert_not_called()
        mock_collection_manager.create_collection.assert_not_called()

    @patch('hf_to_cb_dataset_migrator.migration.Cluster')
    def test_connect_failure(self, mock_cluster_class):
        # Simulate connection failure
        mock_cluster_class.side_effect = CouchbaseException(message='Connection failed')

        with self.assertRaises(Exception) as e:
            self.migrator.connect(
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                cb_bucket='bucket',
                cb_scope='test_scope',
                cb_collection='test_collection',
            )
        self.assertEqual(str(e.exception), str(Exception(f"Failed to connect to Couchbase cluster: {mock_cluster_class.side_effect}")))    

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

    def test_list_fields(self):
        # Mock dataset with known fields
        mock_dataset = Dataset.from_dict({
            'id': [1, 2],
            'text': ['Hello', 'World'],
            'label': [0, 1]
        })

        with patch('hf_to_cb_dataset_migrator.migration.load_dataset') as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            
            fields = self.migrator.list_fields(
                path='test_dataset',
                revision='1.0.0'
            )
            
            self.assertEqual(fields, ['id', 'text', 'label'])
            mock_load_dataset.assert_called_once()

    def test_migrate_dataset_with_id_fields(self):
        # Mock dataset with multiple fields
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.column_names = ['user_id', 'post_id']
        mock_dataset.__iter__.return_value = [
            {'user_id': 'u1', 'post_id': 'p1', 'text': 'Hello'},
            {'user_id': 'u2', 'post_id': 'p2', 'text': 'World'}
        ]

        with patch('hf_to_cb_dataset_migrator.migration.load_dataset') as mock_load_dataset, \
             patch.object(DatasetMigrator, 'connect') as mock_connect, \
             patch.object(DatasetMigrator, 'close') as mock_close, \
             patch.object(DatasetMigrator, 'upsert_multi') as mock_upsert_multi:
            
            mock_load_dataset.return_value = mock_dataset
            
            result = self.migrator.migrate_dataset(
                path='test_dataset',
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                cb_bucket='bucket',
                cb_scope='test_scope',
                cb_collection='test_collection',
                id_fields='user_id,post_id',
                batch_size=1
            )

            self.assertIsNone(result)
            expected_calls = [
                call({'u1_p1': {'user_id': 'u1', 'post_id': 'p1', 'text': 'Hello'}}),
                call({'u2_p2': {'user_id': 'u2', 'post_id': 'p2', 'text': 'World'}})
            ]
            mock_upsert_multi.assert_has_calls(expected_calls)

    def test_migrate_dataset_with_splits(self):
        # Mock datasets with splits
        train_dataset = MagicMock(spec=Dataset)
        train_dataset.column_names = ['id']
        train_dataset.__iter__.return_value = [
            {'id': 1, 'text': 'Train'}
        ]
        
        test_dataset = MagicMock(spec=Dataset)
        test_dataset.__iter__.return_value = [
            {'id': 2, 'text': 'Test'}
        ]
        
        mock_dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        with patch('hf_to_cb_dataset_migrator.migration.load_dataset') as mock_load_dataset, \
             patch.object(DatasetMigrator, 'connect') as mock_connect, \
             patch.object(DatasetMigrator, 'close') as mock_close, \
             patch.object(DatasetMigrator, 'upsert_multi') as mock_upsert_multi:
            
            mock_load_dataset.return_value = mock_dataset_dict
            
            result = self.migrator.migrate_dataset(
                path='test_dataset',
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                cb_bucket='bucket',
                cb_scope='test_scope',
                cb_collection='test_collection',
                id_fields='id',
                batch_size=1
            )

            self.assertIsNone(result)
            expected_calls = [
                call({'1': {'id': 1, 'text': 'Train', 'split': 'train'}}),
                call({'2': {'id': 2, 'text': 'Test', 'split': 'test'}})
            ]
            mock_upsert_multi.assert_has_calls(expected_calls)

    def test_migrate_dataset_batch_processing(self):
        # Mock dataset with multiple records
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.column_names = ['id', 'text']  # Add column_names attribute
        
        # Create a list of records that will be returned during iteration
        records = [{'id': i, 'text': f'Text{i}'} for i in range(1, 6)]
        mock_dataset.__iter__.return_value = iter(records)  # Use iter() to make it a proper iterator

        with patch('hf_to_cb_dataset_migrator.migration.load_dataset') as mock_load_dataset, \
             patch.object(DatasetMigrator, 'connect') as mock_connect, \
             patch.object(DatasetMigrator, 'close') as mock_close, \
             patch.object(DatasetMigrator, 'upsert_multi') as mock_upsert_multi:
            
            mock_load_dataset.return_value = mock_dataset
            mock_upsert_multi.return_value = None  # Ensure upsert_multi doesn't raise any exceptions
            
            result = self.migrator.migrate_dataset(
                path='test_dataset',
                cb_url='couchbase://localhost',
                cb_username='user',
                cb_password='pass',
                cb_bucket='bucket',
                cb_scope='test_scope',
                cb_collection='test_collection',
                id_fields='id',
                batch_size=2
            )

            self.assertIsNone(result)
            expected_calls = [
                call({
                    '1': {'id': 1, 'text': 'Text1'},
                    '2': {'id': 2, 'text': 'Text2'}
                }),
                call({
                    '3': {'id': 3, 'text': 'Text3'},
                    '4': {'id': 4, 'text': 'Text4'}
                }),
                call({
                    '5': {'id': 5, 'text': 'Text5'}
                })
            ]
            mock_upsert_multi.assert_has_calls(expected_calls)

    def test_list_fields_with_split(self):
        # Create mock dataset
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.column_names = ['field1', 'field2']
        
        # Mock load_dataset function
        with patch('hf_to_cb_dataset_migrator.migration.load_dataset') as mock_load:
            mock_load.return_value = mock_dataset
            
            migrator = DatasetMigrator()
            fields = migrator.list_fields(
                path='test_dataset',
                split='train'
            )
            
            # Verify load_dataset was called with correct parameters
            mock_load.assert_called_once_with(
                path='test_dataset',
                name=None,
                data_files=None,
                revision=None,
                use_auth_token=None,
                split='train'
            )
            
            assert fields == ['field1', 'field2']

    def test_list_fields_with_dataset_dict(self):
        # Create mock dataset dictionary
        mock_train_dataset = MagicMock(spec=Dataset)
        mock_train_dataset.column_names = ['field1', 'field2']
        
        mock_dataset_dict = DatasetDict({
            'train': mock_train_dataset
        })
        
        # Mock load_dataset function
        with patch('hf_to_cb_dataset_migrator.migration.load_dataset') as mock_load:
            mock_load.return_value = mock_dataset_dict
            
            migrator = DatasetMigrator()
            fields = migrator.list_fields(
                path='test_dataset'
            )
            
            # Verify load_dataset was called with correct parameters
            mock_load.assert_called_once_with(
                path='test_dataset',
                name=None,
                data_files=None,
                revision=None,
                use_auth_token=None,
                split=None
            )
            
            assert fields == ['field1', 'field2']


if __name__ == '__main__':
    unittest.main()