# test_integration.py

import os
import subprocess
import json
import pytest
from click.testing import CliRunner
from cli import main

# Retrieve environment variables
COUCHBASE_URL = os.getenv('COUCHBASE_URL')
COUCHBASE_USERNAME = os.getenv('COUCHBASE_USERNAME')
COUCHBASE_PASSWORD = os.getenv('COUCHBASE_PASSWORD')
COUCHBASE_BUCKET = os.getenv('COUCHBASE_BUCKET')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

DATASET_PATH = 'McAuley-Lab/Amazon-Reviews-2023'

@pytest.fixture(scope='module')
def cli_runner():
    return CliRunner()

def test_list_configs(cli_runner):
    """
    Test the 'list-configs' command with various options.
    """
    # Without token
    result = cli_runner.invoke(
        main,
        ['list-configs', '--path', DATASET_PATH]
    )
    assert result.exit_code == 0
    print("Available configurations:")
    print(result.output)

    # With token if required
    if HUGGINGFACE_TOKEN:
        result = cli_runner.invoke(
            main,
            ['list-configs', '--path', DATASET_PATH, '--token', HUGGINGFACE_TOKEN]
        )
        assert result.exit_code == 0
        print("Available configurations with token:")
        print(result.output)

    # Output in JSON format
    result = cli_runner.invoke(
        main,
        ['list-configs', '--path', DATASET_PATH, '--json-output']
    )
    assert result.exit_code == 0
    configs = json.loads(result.output)
    assert isinstance(configs, list)
    print("Configurations in JSON format:")
    print(configs)

def test_list_splits(cli_runner):
    """
    Test the 'list-splits' command with various options.
    """
    # First, get a configuration name
    result = cli_runner.invoke(
        main,
        ['list-configs', '--path', DATASET_PATH, '--json-output']
    )
    assert result.exit_code == 0
    configs = json.loads(result.output)
    config_name = configs[0] if configs else None

    # Without config name
    result = cli_runner.invoke(
        main,
        ['list-splits', '--path', DATASET_PATH]
    )
    assert result.exit_code == 0
    print("Available splits without config name:")
    print(result.output)

    # With config name
    if config_name:
        result = cli_runner.invoke(
            main,
            ['list-splits', '--path', DATASET_PATH, '--name', config_name]
        )
        assert result.exit_code == 0
        print(f"Available splits for config '{config_name}':")
        print(result.output)

    # Output in JSON format
    result = cli_runner.invoke(
        main,
        ['list-splits', '--path', DATASET_PATH, '--json-output']
    )
    assert result.exit_code == 0
    splits = json.loads(result.output)
    assert isinstance(splits, list)
    print("Splits in JSON format:")
    print(splits)

def test_migrate(cli_runner):
    """
    Test the 'migrate' command with various combinations of options.
    """
    # Prepare input for Couchbase credentials
    couchbase_inputs = '\n'.join([
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET
    ])

    # Common options
    revision = 'main'
    save_infos = True
    verification_mode = 'basic_checks'  # Options: 'no_checks', 'basic_checks', 'all_checks'

    # 1. Test with only --path
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH,
            '--split', 'train[:1%]',  # Use a small subset for testing
            '--id-fields', 'review_id',
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--revision', revision,
            '--verification-mode', verification_mode,
            '--save-infos' if save_infos else '--no-save-infos',
            '--token', HUGGINGFACE_TOKEN,
            '--streaming', 'False',
            '--num-proc', '1',
            '--storage-options', '{}',
            '--trust-remote-code',
            '--cb-scope', 'my_scope',
            '--cb-collection', 'my_collection'
        ],
        input=couchbase_inputs
    )
    assert result.exit_code == 0
    assert "Migration completed successfully." in result.output
    print("Migration with only --path output:")
    print(result.output)

    # 2. Test with --path and --name
    # Get available configurations to choose a valid name
    configs_result = cli_runner.invoke(
        main,
        ['list-configs', '--path', DATASET_PATH, '--json-output']
    )
    assert configs_result.exit_code == 0
    configs = json.loads(configs_result.output)
    config_name = configs[0] if configs else None

    if config_name:
        result = cli_runner.invoke(
            main,
            [
                'migrate',
                '--path', DATASET_PATH,
                '--name', config_name,
                '--split', 'train[:1%]',
                '--id-fields', 'review_id',
                '--cache-dir', '/tmp/hf_dataset_cache_dir',
                '--revision', revision,
                '--verification-mode', verification_mode,
                '--save-infos' if save_infos else '--no-save-infos',
                '--token', HUGGINGFACE_TOKEN,
                '--streaming', 'False',
                '--num-proc', '1',
                '--storage-options', '{}',
                '--trust-remote-code',
                '--cb-scope', 'my_scope',
                '--cb-collection', 'my_collection'
            ],
            input=couchbase_inputs
        )
        assert result.exit_code == 0
        assert "Migration completed successfully." in result.output
        print(f"Migration with --path and --name='{config_name}' output:")
        print(result.output)

    # 3. Test with --data-files and file patterns
    # Assuming the dataset allows specifying data files (e.g., local files or remote patterns)
    # For the purpose of this test, we'll use a placeholder pattern
    data_files_pattern = 's3://path/to/dataset/files/*.json'  # Update if applicable

    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH,
            '--data-files', data_files_pattern,
            '--split', 'train[:1%]',
            '--id-fields', 'review_id',
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--revision', revision,
            '--verification-mode', verification_mode,
            '--save-infos' if save_infos else '--no-save-infos',
            '--token', HUGGINGFACE_TOKEN,
            '--streaming', 'False',
            '--num-proc', '1',
            '--storage-options', '{}',
            '--trust-remote-code',
            '--cb-scope', 'my_scope',
            '--cb-collection', 'my_collection'
        ],
        input=couchbase_inputs
    )
    if result.exit_code == 0:
        assert "Migration completed successfully." in result.output
        print("Migration with --data-files and file pattern output:")
        print(result.output)
    else:
        print("Migration with --data-files failed (expected if data_files not supported):")
        print(result.output)

    # 4. Test with all possible options (excluding --data-dir)
    download_config = json.dumps({
        'resume_download': True,
        'use_etag': True
    })

    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH,
            '--name', config_name if config_name else 'default',
            '--data-files', data_files_pattern,
            '--split', 'train[:1%]',
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--download-config', download_config,
            '--download-mode', 'reuse_dataset_if_exists',
            '--verification-mode', 'all_checks',
            '--keep-in-memory',
            '--save-infos' if save_infos else '--no-save-infos',
            '--revision', revision,
            '--token', HUGGINGFACE_TOKEN,
            '--streaming', 'False',
            '--num-proc', '2',
            '--storage-options', '{}',
            '--trust-remote-code',
            '--id-fields', 'review_id',
            '--cb-scope', 'my_scope',
            '--cb-collection', 'my_collection'
        ],
        input=couchbase_inputs
    )
    if result.exit_code == 0:
        assert "Migration completed successfully." in result.output
        print("Migration with all options output:")
        print(result.output)
    else:
        print("Migration with all options failed:")
        print(result.output)