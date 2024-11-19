# test_cli.py

import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from hf_to_cb_dataset_migrator.cli import main
from hf_to_cb_dataset_migrator.migration import DatasetMigrator


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_list_configs_cmd(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.list_configs.return_value = ['config1', 'config2']
        mock_migrator_class.return_value = mock_migrator

        result = self.runner.invoke(
            main,
            ['list-configs', '--path', 'test_dataset']
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Available configurations for 'test_dataset':", result.output)
        self.assertIn("- config1", result.output)
        self.assertIn("- config2", result.output)
        mock_migrator.list_configs.assert_called_once_with(
            'test_dataset',
        )

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_list_splits_cmd(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.list_splits.return_value = ['train', 'test', 'validation']
        mock_migrator_class.return_value = mock_migrator

        result = self.runner.invoke(
            main,
            ['list-splits', '--path', 'test_dataset', '--name', 'config1']
        )
        print(result.output)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Available splits for dataset 'test_dataset' with config 'config1':", result.output)
        self.assertIn("- train", result.output)
        self.assertIn("- test", result.output)
        self.assertIn("- validation", result.output)
        mock_migrator.list_splits.assert_called_once_with(
            'test_dataset',
            config_name='config1',
        )

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_migrate_cmd(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.migrate_dataset.return_value = True
        mock_migrator_class.return_value = mock_migrator

        # Mock input prompts for Couchbase credentials
        inputs = '\n'.join(['couchbase://localhost', 'user', 'pass', 'bucket'])

        result = self.runner.invoke(
            main,
            ['migrate', '--path', 'test_dataset', '--id-fields', 'id'],
            input=inputs
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting migration of dataset 'test_dataset' to Couchbase bucket 'bucket'...", result.output)
        self.assertIn("Migration completed successfully.", result.output)
        mock_migrator.migrate_dataset.assert_called_once()

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_migrate_cmd_failure(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.migrate_dataset.return_value = False
        mock_migrator_class.return_value = mock_migrator

        # Mock input prompts for Couchbase credentials
        inputs = '\n'.join(['couchbase://localhost', 'user', 'pass', 'bucket'])

        result = self.runner.invoke(
            main,
            ['migrate', '--path', 'test_dataset', '--id-fields', 'id'],
            input=inputs
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting migration of dataset 'test_dataset' to Couchbase bucket 'bucket'...", result.output)
        self.assertIn("Migration failed.", result.output)
        mock_migrator.migrate_dataset.assert_called_once()

    def test_main_help(self):
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("CLI tool to interact with Hugging Face datasets and migrate them to Couchbase.", result.output)

    def test_list_configs_help(self):
        result = self.runner.invoke(main, ['list-configs', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("List all configuration names for a given dataset.", result.output)

    def test_list_splits_help(self):
        result = self.runner.invoke(main, ['list-splits', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("List all available splits for a given dataset and configuration.", result.output)

    def test_migrate_help(self):
        result = self.runner.invoke(main, ['migrate', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Migrate datasets from Hugging Face to Couchbase.", result.output)


if __name__ == '__main__':
    unittest.main()