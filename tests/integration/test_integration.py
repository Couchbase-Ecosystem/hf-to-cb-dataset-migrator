# test_integration.py

import os
import subprocess
import json
import pytest
from click.testing import CliRunner
from hf_to_cb_dataset_migrator.cli import main

# Retrieve environment variables
COUCHBASE_URL = os.getenv('COUCHBASE_URL')
COUCHBASE_USERNAME = os.getenv('COUCHBASE_USERNAME')
COUCHBASE_PASSWORD = os.getenv('COUCHBASE_PASSWORD')
COUCHBASE_BUCKET = os.getenv('COUCHBASE_BUCKET')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

DATASET_PATH_WITH_NAME = 'McAuley-Lab/Amazon-Reviews-2023'
DATASET_PATH_NO_NAME = 'stanfordnlp/imdb'  # Dataset that doesn't require a config name

@pytest.fixture(scope='module')
def cli_runner():
    return CliRunner()

def test_list_configs_with_name(cli_runner):
    """
    Test the 'list-configs' command for a dataset that requires a config name.
    """
    # With token if required
    if HUGGINGFACE_TOKEN:
        result = cli_runner.invoke(
            main,
            ['list-configs', '--path', DATASET_PATH_WITH_NAME, '--token', HUGGINGFACE_TOKEN]
        )
    else:
        result = cli_runner.invoke(
            main,
            ['list-configs', '--path', DATASET_PATH_WITH_NAME]
        )

    assert result.exit_code == 0
    print("Available configurations for dataset requiring name:")
    print(result.output)

def test_list_splits_with_name(cli_runner):
    """
    Test the 'list-splits' command for a dataset that requires a config name.
    """
    # Get available configurations
    if HUGGINGFACE_TOKEN:
        result = cli_runner.invoke(
            main,
            ['list-configs', '--path', DATASET_PATH_WITH_NAME, '--json-output', '--token', HUGGINGFACE_TOKEN]
        )
    else:
        result = cli_runner.invoke(
            main,
            ['list-configs', '--path', DATASET_PATH_WITH_NAME, '--json-output']
        )

    assert result.exit_code == 0
    configs = json.loads(result.output)
    config_name = configs[0] if configs else None

    assert config_name is not None, "No configurations found for the dataset requiring name."

    # List splits with config name
    result = cli_runner.invoke(
        main,
        ['list-splits', '--path', DATASET_PATH_WITH_NAME, '--name', config_name]
    )
    assert result.exit_code == 0
    print(f"Available splits for dataset '{DATASET_PATH_WITH_NAME}' with config '{config_name}':")
    print(result.output)


def test_list_fields(cli_runner):
    """
    Test the 'list-fields' command for datasets with and without configuration names.
    """
    # Dataset without configuration name
    result = cli_runner.invoke(
        main,
        ['list-fields', '--path', DATASET_PATH_NO_NAME]
    )
    assert result.exit_code == 0
    print(f"Fields for dataset '{DATASET_PATH_NO_NAME}':")
    print(result.output)

    # Dataset with configuration name
    result = cli_runner.invoke(
        main,
        ['list-configs', '--path', DATASET_PATH_WITH_NAME, '--json-output']
    )
    assert result.exit_code == 0
    configs = json.loads(result.output)
    config_name = configs[0] if configs else None
    assert config_name is not None, "No configurations found for the dataset requiring name."

    result = cli_runner.invoke(
        main,
        ['list-fields', '--path', DATASET_PATH_WITH_NAME, '--name', config_name]
    )
    assert result.exit_code == 0
    print(f"Fields for dataset '{DATASET_PATH_WITH_NAME}' with config '{config_name}':")
    print(result.output)


def test_migrate_with_name(cli_runner):
    """
    Test the 'migrate' command for a dataset that requires a config name.
    """
    # Prepare input for Couchbase credentials
    couchbase_inputs = '\n'.join([
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET
    ])

    config_name = "0core_last_out_All_Beauty"
    # Common options
    revision = 'main'
    save_infos = True
    verification_mode = 'basic_checks'  # Options: 'no_checks', 'basic_checks', 'all_checks'

    # Test migration with --path and --name
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_WITH_NAME,
            '--name', config_name,
            '--split', 'train[:1%]',  # Use a small subset for testing
            '--id-fields', "user_id",
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--revision', revision,
            '--verification-mode', verification_mode,
            '--save-infos' if save_infos else '--no-save-infos',
            '--token', HUGGINGFACE_TOKEN,
            '--no-streaming',
            '--num-proc', '1',
            '--storage-options', '{}',
            '--trust-remote-code',
            '--cb-scope', 'my_scope_with_name',
            '--cb-collection', 'my_collection_with_name'
        ],
        input=couchbase_inputs
    )
    print(result.output)
    assert result.exit_code == 0
    assert "Migration completed successfully." in result.output
    print(f"Migration output for dataset '{DATASET_PATH_WITH_NAME}' with config '{config_name}':")
    print(result.output)

def test_list_splits_no_name(cli_runner):
    """
    Test the 'list-splits' command for a dataset that does not require a config name.
    """
    result = cli_runner.invoke(
        main,
        ['list-splits', '--path', DATASET_PATH_NO_NAME]
    )
    assert result.exit_code == 0
    print(f"Available splits for dataset '{DATASET_PATH_NO_NAME}':")
    print(result.output)

def test_migrate_no_name(cli_runner):
    """
    Test the 'migrate' command for a dataset that does not require a config name.
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

    # Test migration with only --path (no --name)
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_NO_NAME,
            #'--split', 'train',  # Use a small subset for testing
            #'--id-fields', 'text',
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--revision', revision,
            '--verification-mode', verification_mode,
            '--save-infos' if save_infos else '--no-save-infos',
            '--token', HUGGINGFACE_TOKEN,
            '--streaming',
            '--storage-options', '{}',
            '--trust-remote-code',
            '--cb-scope', 'my_scope_no_name',
            '--cb-collection', 'my_collection_no_name'
        ],
        input=couchbase_inputs
    )
    assert result.exit_code == 0
    assert "Migration completed successfully." in result.output
    print(f"Migration output for dataset '{DATASET_PATH_NO_NAME}' without config name:")
    print(result.output)
