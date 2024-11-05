# tests/test_cli.py

import unittest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import json
from hf_to_cb_dataset_migrator.cli import main

class TestCLI(unittest.TestCase):
    @patch('my_cli.cli.DatasetMigrator')
    def test_migrate_with_all_parameters(self, mock_migrator_class):
        # Mock the migrator instance
        mock_migrator = mock_migrator_class.return_value
        mock_migrator.migrate_dataset.return_value = True

        runner = CliRunner()
        result = runner.invoke(main, [
            'migrate',
            '--path', 'dataset',
            '--name', 'config1',
            '--data-dir', '/path/to/data_dir',
            '--data-files', '/path/to/file1', '/path/to/file2',
            '--split', 'train',
            '--cache-dir', '/path/to/cache',
            '--features', 'your_features',
            '--download-config', 'your_download_config',
            '--download-mode', 'force_redownload',
            '--verification-mode', 'all_checks',
            '--keep-in-memory',
            '--save-infos',
            '--revision', 'main',
            '--token', 'your_token',
            '--no-streaming',
            '--num-proc', '4',
            '--storage-options', '{"option_key": "option_value"}',
            '--trust-remote-code',
            '--id-fields', 'id,title',
            '--cb-url', 'couchbase://localhost',
            '--cb-username', 'username',
            '--cb-password', 'password',
            '--cb-bucket', 'bucket',
            '--cb-scope', 'scope',
            '--cb-collection', 'collection',
            '--api-key', 'api_key_value',
            '--config-kwargs', 'key1=value1', 'key2=value2'
        ])
        self.assertEqual(result.exit_code, 0)
        mock_migrator_class.assert_called_with(api_key='api_key_value')
        mock_migrator.migrate_dataset.assert_called_once()
        self.assertIn('Migration completed successfully.', result.output)

    @patch('my_cli.cli.DatasetMigrator')
    def test_migrate_invalid_config(self, mock_migrator_class):
        # Mock the get_dataset_config_names function
        with patch('my_cli.cli.get_dataset_config_names') as mock_get_configs:
            mock_get_configs.return_value = ['config1', 'config2']
            mock_migrator = mock_migrator_class.return_value

            runner = CliRunner()
            result = runner.invoke(main, [
                'migrate',
                '--path', 'dataset',
                '--name', 'invalid_config',
                '--cb-url', 'couchbase://localhost',
                '--cb-username', 'user',
                '--cb-password', 'pass',
                '--cb-bucket', 'bucket'
            ])
            self.assertEqual(result.exit_code, 0)
            mock_get_configs.assert_called_once()
            self.assertIn("Invalid configuration name 'invalid_config'. Available configurations are: ['config1', 'config2']", result.output)
            mock_migrator.migrate_dataset.assert_not_called()

    @patch('my_cli.cli.DatasetMigrator')
    def test_migrate_invalid_split(self, mock_migrator_class):
        # Mock the get_dataset_config_names and get_dataset_split_names functions
        with patch('my_cli.cli.get_dataset_config_names') as mock_get_configs, \
             patch('my_cli.cli.get_dataset_split_names') as mock_get_splits:
            mock_get_configs.return_value = ['config1']
            mock_get_splits.return_value = ['train', 'test']
            mock_migrator = mock_migrator_class.return_value

            runner = CliRunner()
            result = runner.invoke(main, [
                'migrate',
                '--path', 'dataset',
                '--name', 'config1',
                '--split', 'invalid_split',
                '--cb-url', 'couchbase://localhost',
                '--cb-username', 'user',
                '--cb-password', 'pass',
                '--cb-bucket', 'bucket'
            ])
            self.assertEqual(result.exit_code, 0)
            mock_get_configs.assert_called_once()
            mock_get_splits.assert_called_once()
            self.assertIn("Invalid split name 'invalid_split'. Available splits are: ['train', 'test']", result.output)
            mock_migrator.migrate_dataset.assert_not_called()