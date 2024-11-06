# tests/test_migration.py

import unittest
from unittest.mock import patch, MagicMock
from hf_to_cb_dataset_migrator.migration import DatasetMigrator
from datasets import DatasetDict, Dataset, IterableDatasetDict
from datasets.features import Features

class TestDatasetMigrator(unittest.TestCase):
    @patch('my_cli.migration.load_dataset')
    @patch('my_cli.migration.get_dataset_config_names')
    @patch('my_cli.migration.get_dataset_split_names')
    @patch('my_cli.migration.DatasetMigrator.connect')
    @patch('my_cli.migration.DatasetMigrator.close')
    @patch('my_cli.migration.DatasetMigrator.insert_multi')
    def test_migrate_dataset_with_all_parameters(
        self, mock_insert_multi, mock_close, mock_connect, mock_get_splits, mock_get_configs, mock_load_dataset
    ):
        # Setup the mock return values
        mock_get_configs.return_value = ['config1', 'config2']
        mock_get_splits.return_value = ['train', 'test']
        mock_dataset = MagicMock(spec=DatasetDict)
        mock_dataset.items.return_value = [('train', [{'id': 1, 'text': 'sample'}])]
        mock_load_dataset.return_value = mock_dataset

        migrator = DatasetMigrator(api_key='api_key_value')
        result = migrator.migrate_dataset(
            path='dataset',
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            couchbase_bucket='bucket',
            cb_scope='scope',
            cb_collection='collection',
            id_fields='id',
            name='config1',
            data_dir='/path/to/data_dir',
            data_files=['/path/to/file1', '/path/to/file2'],
            split='train',
            cache_dir='/path/to/cache',
            features=Features({'text': 'string'}),
            download_config='download_config_value',
            download_mode='force_redownload',
            verification_mode='all_checks',
            keep_in_memory=True,
            save_infos=True,
            revision='main',
            token='token_value',
            streaming=False,
            num_proc=4,
            storage_options={'option_key': 'option_value'},
            trust_remote_code=True,
            custom_arg='custom_value'
        )
        self.assertTrue(result)
        mock_get_configs.assert_called_once()
        mock_get_splits.assert_called_once()
        mock_load_dataset.assert_called_once()
        mock_connect.assert_called_once()
        mock_insert_multi.assert_called()
        mock_close.assert_called_once()