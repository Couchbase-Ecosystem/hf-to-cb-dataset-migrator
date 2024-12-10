# test_integration.py

import os
import subprocess
import json
import pytest
from click.testing import CliRunner
from hf_to_cb_dataset_migrator.cli import main
from typing import List
import logging
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions, KnownConfigProfiles
from datasets import Dataset, DatasetDict
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Retrieve environment variables
COUCHBASE_URL = os.getenv('COUCHBASE_URL')
COUCHBASE_USERNAME = os.getenv('COUCHBASE_USERNAME')
COUCHBASE_PASSWORD = os.getenv('COUCHBASE_PASSWORD')
COUCHBASE_BUCKET = os.getenv('COUCHBASE_BUCKET')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

DATASET_PATH_WITH_NAME = 'McAuley-Lab/Amazon-Reviews-2023'
DATASET_PATH_NO_NAME = 'stanfordnlp/imdb'  # Dataset that doesn't require a config name

def check_env_variables():
    """Check if all required environment variables are set"""
    required_vars = {
        'COUCHBASE_URL': COUCHBASE_URL,
        'COUCHBASE_USERNAME': COUCHBASE_USERNAME,
        'COUCHBASE_PASSWORD': COUCHBASE_PASSWORD,
        'COUCHBASE_BUCKET': COUCHBASE_BUCKET
    }
    missing = [var for var, val in required_vars.items() if not val]
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")

@pytest.fixture(scope='module')
def cli_runner():
    return CliRunner()

@pytest.fixture(scope='module')
def couchbase_inputs():
    """Fixture to provide Couchbase connection inputs"""
    check_env_variables()
    return '\n'.join([
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET,
    ])

@pytest.fixture(scope='module')
def cleanup_couchbase():
    """Fixture to clean up test data before and after tests"""
    # Setup - could add code to clean existing test collections if needed
    yield
    # Teardown - clean up test collections
    # Add cleanup code here if needed

@pytest.fixture(scope='function')
def cleanup_collection():
    """Fixture to clean up test collection before and after each test"""
    def _cleanup_collection(scope_name: str, collection_name: str):
        auth = PasswordAuthenticator(COUCHBASE_USERNAME, COUCHBASE_PASSWORD)
        cluster_opts = ClusterOptions(auth)
        cluster_opts.apply_profile(KnownConfigProfiles.WanDevelopment)
        cluster = Cluster(COUCHBASE_URL, cluster_opts)
        bucket = cluster.bucket(COUCHBASE_BUCKET)
        
        try:
            # Drop and recreate collection
            mgr = bucket.collections()
            mgr.drop_collection(collection_name, scope_name)
        except Exception:
            pass  # Collection might not exist
            
    return _cleanup_collection

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


def test_migrate_with_invalid_config(cli_runner, couchbase_inputs, cleanup_collection):
    """Test migration with invalid configuration name"""
    cleanup_collection('test_scope', 'test_collection')
    
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_WITH_NAME,
            '--name', 'invalid_config',
            '--split', 'train[:1%]',
            '--cb-scope', 'test_scope',
        ],
        input=couchbase_inputs,
        catch_exceptions=False
    )
    assert result.exit_code != 0
    assert "invalid_config" in result.output

def test_migrate_with_all_options(cli_runner, couchbase_inputs, cleanup_collection):
    """Test migration with all possible valid options"""
    cleanup_collection('test_scope', 'test_collection')
    
    config_name = "0core_last_out_All_Beauty"
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_WITH_NAME,
            '--name', config_name,
            '--split', 'train[:1%]',
            '--id-fields', 'user_id',
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--revision', 'main',
            '--verification-mode', 'all_checks',
            '--save-infos',
            '--token', HUGGINGFACE_TOKEN,
            '--no-streaming',
            '--num-proc', '2',
            '--storage-options', '{"anon": true}',
            '--trust-remote-code',
            '--cb-scope', 'test_scope',
            '--cb-collection', 'test_collection',
            '--cb-batch-size', '500',
            '--download-mode', 'force_redownload',
            '--debug'
        ],
        input=couchbase_inputs
    )
    assert "Migration completed successfully." in result.output
    assert result.exit_code == 0

    expected_fields = ['user_id']
    assert validate_migrated_data(
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET,
        'test_scope',
        'test_collection',
        expected_fields,
        expected_count=1
    )

def test_migrate_with_streaming(cli_runner, couchbase_inputs, cleanup_collection):
    """Test migration with streaming mode"""
    cleanup_collection('test_scope_stream', 'test_collection_stream')
    
    config_name = "0core_last_out_All_Beauty"
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_WITH_NAME,
            '--name', config_name,
            '--split', 'train',
            '--id-fields', 'user_id',
            '--verification-mode', 'basic_checks',
            '--cb-scope', 'test_scope_stream',
            '--cb-collection', 'test_collection_stream'
        ],
        input=couchbase_inputs
    )
    
    assert result.exit_code == 0
    assert "Migration completed successfully." in result.output

    expected_fields = ['user_id']
    assert validate_migrated_data(
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET,
        'test_scope_stream',
        'test_collection_stream',
        expected_fields,
        expected_count=8151
    )

def test_migrate_with_memory_retention(cli_runner, couchbase_inputs, cleanup_collection):
    """Test migration with keep-in-memory option"""
    cleanup_collection('test_scope_memory', 'test_collection_memory')
    
    config_name = "0core_last_out_All_Beauty"
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_WITH_NAME,
            '--name', config_name,
            '--split', 'train',
            '--id-fields', 'user_id',
            '--keep-in-memory',
            '--verification-mode', 'no_checks',
            '--cb-scope', 'test_scope_memory',
            '--cb-collection', 'test_collection_memory'
        ],
        input=couchbase_inputs
    )
    
    assert result.exit_code == 0
    assert "Migration completed successfully." in result.output

    expected_fields = ['user_id']
    assert validate_migrated_data(
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET,
        'test_scope_memory',
        'test_collection_memory',
        expected_fields,
        expected_count=8151
    )

def test_migrate_invalid_inputs(cli_runner, couchbase_inputs):
    """Test migration with invalid inputs"""
    
    # Test with invalid split
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_NO_NAME,
            '--split', 'invalid_split',
            '--cb-scope', 'test_scope',
            '--cb-collection', 'test_collection'
        ],
        input=couchbase_inputs
    )
    assert result.exit_code != 0
    assert "invalid_split" in result.output

    # Test with invalid scope/collection names
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_NO_NAME,
            '--split', 'train',
            '--cb-scope', 'invalid/scope',
            '--cb-collection', 'test_collection'
        ],
        input=couchbase_inputs
    )
    assert result.exit_code != 0
    assert "invalid/scope" in result.output

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

def test_migrate_no_name(cli_runner, couchbase_inputs, cleanup_collection):
    """Test migration for dataset without config name"""
    cleanup_collection('my_scope_no_name', 'my_collection_no_name')
    
    result = cli_runner.invoke(
        main,
        [
            'migrate',
            '--path', DATASET_PATH_NO_NAME,
            '--split', 'train',  # Use a small subset for testing
            '--cache-dir', '/tmp/hf_dataset_cache_dir',
            '--revision', 'main',
            '--verification-mode', 'basic_checks',
            '--save-infos',
            '--token', HUGGINGFACE_TOKEN,
            '--storage-options', '{}',
            '--trust-remote-code',
            '--cb-scope', 'my_scope_no_name',
            '--cb-collection', 'my_collection_no_name'
        ],
        input=couchbase_inputs
    )
    print(result.output)
    assert "Migration completed successfully." in result.output
    assert result.exit_code == 0
    

    # Validate the migrated data
    expected_fields = ['text', 'label']
    assert validate_migrated_data(
        COUCHBASE_URL,
        COUCHBASE_USERNAME,
        COUCHBASE_PASSWORD,
        COUCHBASE_BUCKET,
        'my_scope_no_name',
        'my_collection_no_name',
        expected_fields,
        expected_count=19903
    )

def validate_migrated_data(
    cb_url: str,
    cb_username: str,
    cb_password: str,
    cb_bucket: str,
    cb_scope: str,
    cb_collection: str,
    expected_fields: List[str],
    expected_count: int
) -> bool:
    """
    Validates the migrated data in Couchbase.
    
    Args:
        cb_url: Couchbase cluster URL
        cb_username: Couchbase username
        cb_password: Couchbase password
        cb_bucket: Bucket name
        cb_scope: Scope name
        cb_collection: Collection name
        expected_fields: List of fields that should be present in documents
        expected_count: Expected number of documents
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    from couchbase.options import ClusterOptions

    try:
        # Connect to Couchbase
        auth = PasswordAuthenticator(cb_username, cb_password)
        cluster_opts = ClusterOptions(auth)
        cluster_opts.apply_profile(KnownConfigProfiles.WanDevelopment)
        cluster = Cluster(cb_url, cluster_opts)
        bucket = cluster.bucket(cb_bucket)
        scope = bucket.scope(cb_scope)
        collection = scope.collection(cb_collection)

        # Query to count documents
        query = f'SELECT COUNT(*) as count FROM `{cb_bucket}`.`{cb_scope}`.`{cb_collection}`'
        result = cluster.query(query)
        count = next(result.rows()).get('count', 0)
        assert count >= expected_count, f"Expected {expected_count} documents, found {count}"

        # Query to validate document structure
        query = f'SELECT * FROM `{cb_bucket}`.`{cb_scope}`.`{cb_collection}` LIMIT 1'
        result = cluster.query(query)
        sample_doc = next(result.rows()).get(cb_collection)
        
        # Check if all expected fields are present
        for field in expected_fields:
            assert field in sample_doc, f"Field {field} not found in document"

        return True
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise e

def test_list_fields_integration_with_split(cli_runner):
    """Test list-fields command with split option using a public dataset"""
    # Use a small, public dataset
    result = cli_runner.invoke(main, [
        'list-fields',
        '--path', 'rotten_tomatoes',  # Using the rotten_tomatoes dataset as it's small and public
        '--split', 'train'
    ])
    
    assert result.exit_code == 0
    # Check for known fields in the rotten_tomatoes dataset
    assert 'text' in result.output
    assert 'label' in result.output
