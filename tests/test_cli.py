import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from hf_to_cb_dataset_migrator.cli import main


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_list_configs_cmd(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.list_configs.return_value = ['config1', 'config2']
        mock_migrator_class.return_value = mock_migrator

        # Test with minimum options
        result = self.runner.invoke(main, ['list-configs', '--path', 'test_dataset'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Available configurations for 'test_dataset':", result.output)
        self.assertIn("- config1", result.output)
        self.assertIn("- config2", result.output)

        # Test with all options
        result = self.runner.invoke(main, ['list-configs', '--path', 'test_dataset', '--token', 'dummy_token', '--json-output'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"config1"', result.output)
        self.assertIn('"config2"', result.output)

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_list_splits_cmd(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.list_splits.return_value = ['train', 'test', 'validation']
        mock_migrator_class.return_value = mock_migrator

        # Test with minimum options
        result = self.runner.invoke(main, ['list-splits', '--path', 'test_dataset', '--name', 'config1'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Available splits for dataset 'test_dataset' with config 'config1':", result.output)

        # Test with all options
        result = self.runner.invoke(main, ['list-splits', '--path', 'test_dataset', '--name', 'config1', '--token', 'dummy_token', '--json-output'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"train"', result.output)
        self.assertIn('"test"', result.output)
        self.assertIn('"validation"', result.output)

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_migrate_cmd(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.migrate_dataset.return_value = True
        mock_migrator_class.return_value = mock_migrator

        # Mock input prompts for Couchbase credentials
        inputs = '\n'.join(['couchbase://localhost', 'user', 'pass', 'bucket'])

        # Test with minimum options
        result = self.runner.invoke(main, ['migrate', '--path', 'test_dataset', '--id-fields', 'id', '--cb-scope', 'test-scope'], input=inputs)
        # Assert that the migrate_dataset method was called with the correct arguments
        mock_migrator.migrate_dataset.assert_called_once_with(
            path='test_dataset',
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            cb_bucket='bucket',
            cb_scope='test-scope', 
            cb_collection=None, 
            id_fields='id',
            name=None, 
            data_files=None, 
            split=None, 
            cache_dir=None, 
            download_config=None, 
            download_mode=None, 
            verification_mode=None, 
            keep_in_memory=False, 
            save_infos=False, 
            revision=None, 
            token=None, 
            streaming=True, 
            num_proc=None, 
            storage_options=None, 
            trust_remote_code=None,
            batch_size=1000
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Migration completed successfully.", result.output)
    
        # Reset mock for next test
        mock_migrator.reset_mock()

        # Test with all options
        result = self.runner.invoke(
            main,
            [
                'migrate',
                '--path', 'test_dataset',
                '--id-fields', 'id',
                '--split', 'train',
                '--token', 'dummy_token',
                '--cb-url', 'couchbase://localhost',
                '--cb-username', 'user',
                '--cb-password', 'pass',
                '--cb-bucket', 'bucket',
                '--cb-scope', 'scope',
                '--cb-collection', 'collection',
                '--debug'
            ]
        )

        mock_migrator.migrate_dataset.assert_called_once_with(
        path='test_dataset',
        split='train',
        id_fields='id',
        token='dummy_token',
        cb_url='couchbase://localhost',
        cb_username='user',
        cb_password='pass',
        cb_bucket='bucket',
        cb_scope='scope',
        cb_collection='collection',
        name=None, 
        data_files=None,
        cache_dir=None, 
        download_config=None, 
        download_mode=None, 
        verification_mode=None, 
        keep_in_memory=False, 
        save_infos=False, 
        revision=None,
        streaming=True, 
        num_proc=None, 
        storage_options=None, 
        trust_remote_code=None,
        batch_size=1000
    )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting migration of dataset 'test_dataset' to Couchbase bucket 'bucket'...", result.output)
        self.assertIn("Migration completed successfully.", result.output)

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_migrate_cmd_failure(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.migrate_dataset.side_effect = Exception("Migration failed")
        mock_migrator_class.return_value = mock_migrator

        result = self.runner.invoke(
            main,
            [
                'migrate',
                '--path', 'test_dataset',
                '--id-fields', 'id',
                '--cb-url', 'couchbase://localhost',
                '--cb-username', 'user',
                '--cb-password', 'pass',
                '--cb-bucket', 'bucket',
                '--cb-scope', 'scope',
                '--cb-collection', 'collection'
            ]
        )

        mock_migrator.migrate_dataset.assert_called_once_with(
            path='test_dataset',
            id_fields='id',
            cb_url='couchbase://localhost',
            cb_username='user',
            cb_password='pass',
            cb_bucket='bucket',
            cb_scope='scope',
            cb_collection='collection',
            split=None,
            token=None,
            name=None,
            data_files=None,
            cache_dir=None,
            download_config=None,
            download_mode=None,
            verification_mode=None,
            keep_in_memory=False,
            save_infos=False,
            revision=None,
            streaming=True,
            num_proc=None,
            storage_options=None,
            trust_remote_code=None,
            batch_size=1000
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Starting migration of dataset 'test_dataset' to Couchbase bucket 'bucket'...", result.output)
        self.assertIn("Migration failed", result.output)

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


    @patch.dict('os.environ', {'MOCK_CLI_FOR_CBMIGRATE': 'true'})
    def test_pre_function(self):
        # Mock environment to enable pre_function execution
        result = self.runner.invoke(
            main,
            [
                'list-configs',  # Command to test
                '--path', 'test_dataset',
                '--token', 'dummy_token',
                '--json-output',
            ]
        )

        # Assert pre_function executed and outputs all options in JSON format
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"path": "test_dataset"', result.output)
        self.assertIn('"token": "dummy_token"', result.output)
        self.assertIn('"json_output": true', result.output)

    @patch.dict('os.environ', {'MOCK_CLI_FOR_CBMIGRATE': 'true'})
    def test_pre_function_with_migrate(self):
        # Test the migrate command to ensure pre_function works across commands
        result = self.runner.invoke(
            main,
            [
                'migrate',
                '--path', 'test_dataset',
                '--id-fields', 'id',
                '--cb-url', 'couchbase://localhost',
                '--cb-username', 'user',
                '--cb-password', 'pass',
                '--cb-bucket', 'test_bucket',
                '--cb-scope', 'scope'
            ]
        )

        # Assert pre_function executed and outputs all options in JSON format
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"path": "test_dataset"', result.output)
        self.assertIn('"id_fields": "id"', result.output)
        self.assertIn('"cb_url": "couchbase://localhost"', result.output)
        self.assertIn('"cb_username": "user"', result.output)
        self.assertIn('"cb_password": "pass"', result.output)
        self.assertIn('"cb_bucket": "test_bucket"', result.output)    

    @patch('hf_to_cb_dataset_migrator.cli.DatasetMigrator')
    def test_list_fields_with_split(self, mock_migrator_class):
        mock_migrator = MagicMock()
        mock_migrator.list_fields.return_value = ['field1', 'field2']
        mock_migrator_class.return_value = mock_migrator

        # Test with minimum options
        result = self.runner.invoke(main, [
            'list-fields',
            '--path', 'test_dataset',
            '--split', 'train',
            '--token', 'dummy_token'
        ])

        # Assert that the list_fields method was called with the correct arguments
        mock_migrator.list_fields.assert_called_once_with(
            path='test_dataset',
            name=None,
            data_files=None,
            revision=None,
            split='train'
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Fields:", result.output)
        self.assertIn("field1", result.output)
        self.assertIn("field2", result.output)

        # Reset mock for next test
        mock_migrator.reset_mock()

        # Test with all options
        result = self.runner.invoke(
            main,
            [
                'list-fields',
                '--path', 'test_dataset',
                '--name', 'config1',
                '--split', 'train',
                '--token', 'dummy_token',
                '--revision', '1.0.0',
                '--data-files', 'file1.csv',
                '--data-files', 'file2.csv',
                '--json-output'
            ]
        )

        mock_migrator.list_fields.assert_called_once_with(
            path='test_dataset',
            name='config1',
            data_files=['file1.csv', 'file2.csv'],
            revision='1.0.0',
            split='train'
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"field1"', result.output)
        self.assertIn('"field2"', result.output)


if __name__ == '__main__':
    unittest.main()