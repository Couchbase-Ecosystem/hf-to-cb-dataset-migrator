# hf_to_cb_dataset_migrator/cli.py

import click
import json
import os
import sys
from hf_to_cb_dataset_migrator.migration import DatasetMigrator
from hf_to_cb_dataset_migrator.utils import generate_help
from typing import Any, Optional
import logging
import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_TIMEOUT = 60
DEFAULT_BATCH_SIZE = 1000
MOCK_CLI_ENV_VAR = 'MOCK_CLI_FOR_CBMIGRATE'
RUN_FROM_CBMIGRATE_ENV_VAR = 'RUN_FROM_CBMIGRATE'

prog_name = "cbmigrate hugging-face" if os.getenv(RUN_FROM_CBMIGRATE_ENV_VAR, "false") == "true" else "hf_to_cb_dataset_migrator"

def pre_function(ctx: click.Context) -> None:
    options = ctx.params
    click.echo(json.dumps(options, indent=2))


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """CLI tool to interact with Hugging Face datasets and migrate them to Couchbase."""
    pass

list_configs_help = generate_help("List all configuration names for a given dataset.",[
    f"{prog_name} list-configs --path dataset.",
    "List configurations for a public dataset.",
    f"{prog_name} list-configs --path my-dataset --token YOUR_HF_TOKEN",
    "List configurations for a private dataset using a token.",
    f"{prog_name} list-configs --path dataset --json-output",
    "Output configurations in JSON format."
])
@main.command('list-configs',help=list_configs_help)
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--download-config', default=None, help='Specific download configuration parameters (optional).')
@click.option('--download-mode', type=click.Choice(['reuse_dataset_if_exists', 'force_redownload']), default=None,
              help='Download mode (optional).')
@click.option('--dynamic-modules-path', default=None, help='Path to dynamic modules (optional).')
@click.option('--data-files', default=None, multiple=True, help='Path(s) to source data file(s) (optional).')
@click.option('--token',default=None, help='Use authentication token for private datasets.')
@click.option('--json-output', is_flag=True, help='Output the configurations in JSON format.')
@click.option('--debug', is_flag=True, help='Enable debug output.')
@click.option('--trust-remote-code', is_flag=True, default=None,
              help='Allow loading arbitrary code from the dataset repository (optional).')
@click.pass_context
def list_configs_cmd(ctx, path, revision, download_config, download_mode, dynamic_modules_path,
                     data_files, token, json_output, debug, trust_remote_code):
    # this is code for mock the cli library for cbmigrate testing
    if os.getenv(MOCK_CLI_ENV_VAR, "false") == "true":
        pre_function(ctx)
        return
    if json_output:
        logging.basicConfig(level=logging.ERROR)   
    elif debug:
        logging.basicConfig(level=logging.DEBUG)

    migrator = DatasetMigrator(token=token)
    
    download_config_dict = None
    if download_config:
        try:
            download_config_dict = json.loads(download_config)
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing download_config JSON: {e}", err=True)
            sys.exit(1)
        

    download_kwargs = {
        'revision': revision,
        'download_config': download_config_dict,
        'download_mode': download_mode,
        'dynamic_modules_path': dynamic_modules_path,
        'data_files': data_files if data_files else None,
        'trust_remote_code': trust_remote_code,
    }
    # Remove None values
    download_kwargs = {k: v for k, v in download_kwargs.items() if v is not None}

    try:
        configs = migrator.list_configs(path, **download_kwargs)
        if configs:
            if json_output:
                click.echo(json.dumps(configs, indent=2))
            else:
                click.echo(f"Available configurations for '{path}':")
                for config in configs:
                    click.echo(f"- {config}")
        else:
            click.echo(f"No configurations found for dataset '{path}' or dataset not found.")            
    except Exception as e:
        click.echo(f"Error listing configurations: {str(e)}", err=True)
        sys.exit(1)


list_splits_help = generate_help("List all available splits for a given dataset and configuration.", [
    f"{prog_name} list-splits --path dataset",
    "List splits for a public dataset",
    f"{prog_name} list-splits --path dataset --name config-name",
    "List splits for a dataset with a configuration name",
    f"{prog_name} list-splits --path my-private-dataset --token YOUR_HF_TOKEN",
    "List splits for a private dataset using a token",
    f"{prog_name} list-splits --path dataset --json-output",
    "Output splits in JSON format",
])
@main.command('list-splits', help=list_splits_help)
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--name', 'config_name', default=None, help='Configuration name of the dataset (optional).')
@click.option('--data-files', default=None, multiple=True, help='Path(s) to source data file(s) (optional).')
@click.option('--download-config', default=None, help='Specific download configuration parameters (optional).')
@click.option('--download-mode', type=click.Choice(['reuse_dataset_if_exists', 'force_redownload']), default=None,
              help='Download mode (optional).')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--token', default=None, help='Authentication token for private datasets (optional).')
@click.option('--json-output', is_flag=True, help='Output the splits in JSON format.')
@click.option('--debug', is_flag=True, help='Enable debug output.')
@click.option('--trust-remote-code', is_flag=True, default=None,
              help='Allow loading arbitrary code from the dataset repository (optional).')
@click.pass_context
def list_splits_cmd(ctx, path, config_name, data_files, download_config, download_mode, revision, token,
                    json_output, debug, trust_remote_code):
    
    # this is code for mock the cli library for cbmigrate testing
    if os.getenv(MOCK_CLI_ENV_VAR, "false") == "true":
        pre_function(ctx)
        return
    
    if json_output:
        logging.basicConfig(level=logging.ERROR)   
    elif debug:
        logging.basicConfig(level=logging.DEBUG)
        
    download_config_dict = None
    if download_config:
        try:
            download_config_dict = json.loads(download_config)
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing download_config JSON: {e}", err=True)
            sys.exit(1)

    migrator = DatasetMigrator(token=token)

    config_kwargs = {
        'data_files': data_files if data_files else None,
        'download_config': download_config_dict,
        'download_mode': download_mode,
        'revision': revision,
        'trust_remote_code': trust_remote_code,
    }
    # Remove None values
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    try:
        splits = migrator.list_splits(path, config_name=config_name, **config_kwargs)
        if splits:
            if json_output:
                click.echo(json.dumps(splits, indent=2))
            else:
                config_name_display = config_name if config_name else "default"
                click.echo(f"Available splits for dataset '{path}' with config '{config_name_display}':")
                for split in splits:
                    click.echo(f"- {split}")
        else:
            click.echo(f"No splits found for dataset '{path}' with config '{config_name}' or dataset not found.")
    except Exception as e:
        click.echo(f"Error listing splits: {str(e)}", err=True)
        sys.exit(1)

list_fields_help = generate_help("List the fields (columns) of a dataset.", [
    f"{prog_name} list-fields --path dataset",
    "List fields for a public dataset",
    f"{prog_name} list-fields --path dataset --name config-name",
    "List fields for a dataset with a configuration name",
    f"{prog_name} list-fields --path my-private-dataset --token YOUR_HF_TOKEN",
    "List fields for a private dataset using a token",
    f"{prog_name} list-fields --path dataset --json-output",
    "Output fields in JSON format",
])
@main.command('list-fields', help=list_fields_help)
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--name', help='Name of the dataset configuration (optional).')
@click.option('--data-files', multiple=True, help='Paths to source data files (optional).')
@click.option('--download-config', default=None, help='Specific download configuration parameters (optional).')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--token', default=None, help='Hugging Face token for private datasets (optional).')
@click.option('--split', default=None, help='Which split of the data to load (optional).')
@click.option('--json-output', is_flag=True, help='Output the fields in JSON format.')
@click.option('--debug', is_flag=True, help='Enable debug output.')
@click.option('--trust-remote-code', is_flag=True, default=None,
              help='Allow loading arbitrary code from the dataset repository (optional).')
@click.pass_context
def list_fields(ctx, path, name, data_files, download_config, revision, token, split, json_output, debug, trust_remote_code):
    
    # this is code for mock the cli library for cbmigrate testing
    if os.getenv(MOCK_CLI_ENV_VAR, "false") == "true":
        pre_function(ctx)
        return
    
    if json_output:
        logging.basicConfig(level=logging.ERROR)   
    elif debug:
        logging.basicConfig(level=logging.DEBUG)

    download_config_dict = None
    if download_config:
        try:
            download_config_dict = json.loads(download_config)
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing download_config JSON: {e}", err=True)
            sys.exit(1)

    migrator = DatasetMigrator(token=token)

    try:
        fields = migrator.list_fields(
            path=path,
            name=name,
            data_files=list(data_files) if data_files else None,
            download_config=download_config_dict,
            revision=revision,
            split=split,
            trust_remote_code=trust_remote_code
        )
        if json_output:
            click.echo(json.dumps(fields))
        else:
            click.echo("Fields:")
            for field in fields:
                click.echo(field)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

cb_opts = "--cb-url couchbase://localhost --cb-username user --cb-password pass "  \
"--cb-bucket my_bucket --cb-scope my_scope --cb-collection my_collection"
migrate_help = generate_help("Migrate datasets from Hugging Face to Couchbase.", [
    f"{prog_name} migrate --path dataset --id-fields id_field {cb_opts}",
    "Migrate the default split of a public dataset with minimal options",
    
    f"{prog_name} migrate --path dataset --name config-name --id-fields id_field {cb_opts}",
    "Migrate a dataset with a specific configuration name",
    
    f"{prog_name} migrate --path dataset --data-files file1.csv --data-files file2.csv --id-fields id_field {cb_opts}",
   "Migrate a dataset by specifying data files",
    
    f"{prog_name} migrate --path dataset --split train --id-fields id_field {cb_opts}",
    "Migrate a specific split of the dataset",
    
    f"{prog_name} migrate --path dataset --cache-dir /path/to/cache --id-fields id_field {cb_opts}",
    "Migrate a dataset using a custom cache directory",
    
    f"{prog_name} migrate --path dataset --download-mode force_redownload --id-fields id_field {cb_opts}",
    "Migrate a dataset with specific download configuration",
    
    f"{prog_name} migrate --path dataset --verification-mode all_checks --id-fields id_field {cb_opts}",
    "# Migrate a dataset with all verification checks",
    
    f"{prog_name} migrate --path dataset --keep-in-memory --id-fields id_field {cb_opts}",
    "Migrate a dataset and keep it in memory",
    
    f"{prog_name} migrate --path dataset --save-infos --id-fields id_field {cb_opts}",
    "Migrate a dataset and save dataset information",
    
    f"{prog_name} migrate --path dataset --no-streaming --id-fields id_field {cb_opts}",
    "# Migrate a dataset with streaming mode disabled",
    
    f"{prog_name} migrate --path dataset --trust-remote-code --id-fields id_field {cb_opts}",
    "Migrate a dataset allowing execution of remote code",
    
    f"{prog_name} migrate --path dataset --id-fields field1,field2 {cb_opts}",
    "Migrate a dataset using multiple fields as document ID",
    
    f"{prog_name} migrate --path my-private-dataset --id-fields id_field --token YOUR_HF_TOKEN {cb_opts}",
    "Migrate a private dataset using a Hugging Face token",
    f"{prog_name} migrate --path dataset --revision 1.0.0 --id-fields id_field {cb_opts}",
    "Migrate a specific revision of a dataset",
    f"{prog_name} migrate --path dataset --num-proc 4 --id-fields id_field {cb_opts}",
    "Migrate a dataset using multiple processes",
])
@main.command(help=migrate_help)
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--name', default=None, help='Configuration name of the dataset (optional).')
#@click.option('--data-dir', default=None, help='Directory with the data files (optional).')
@click.option('--data-files', default=None, multiple=True, help='Path(s) to source data file(s) (optional).')
@click.option('--split', default=None, help='Which split of the data to load (optional).')
@click.option('--cache-dir', default=None, help='Cache directory to store the datasets (optional).')
#@click.option('--features', default=None, help='Set of features to use (optional).')
@click.option('--download-config', default=None, help='Specific download configuration parameters (optional).')
@click.option('--download-mode', type=click.Choice(['reuse_dataset_if_exists', 'force_redownload']), default=None,
              help='Download mode (optional).')
@click.option('--verification-mode', type=click.Choice(['no_checks', 'basic_checks', 'all_checks']), default=None,
              help='Verification mode (optional).')
@click.option('--keep-in-memory', is_flag=True, default=False, help='Keep the dataset in memory (optional).')
@click.option('--save-infos', is_flag=True, default=False, help='Save dataset information (default: False).')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--token', default=None, help='Authentication token for private datasets (optional).')
@click.option('--no-streaming', is_flag=True, default=False, help='Disable streaming mode for dataset loading (default: False).')
@click.option('--num-proc', default=None, type=int, help='Number of processes to use (optional).')
@click.option('--storage-options', default=None, help='Storage options for remote filesystems (optional).')
@click.option('--trust-remote-code', is_flag=True, default=None,
              help='Allow loading arbitrary code from the dataset repository (optional).')
@click.option('--id-fields', default=None, help='Comma-separated list of field names to use as document ID.')
@click.option('--cb-url', prompt='Couchbase URL', help='Couchbase cluster URL (e.g., couchbase://localhost).')
@click.option('--cb-username', prompt='Couchbase username', help='Username for Couchbase authentication.')
@click.option('--cb-password', prompt=True, hide_input=True, confirmation_prompt=False,
              help='Password for Couchbase authentication.')
@click.option('--cb-bucket', prompt='Couchbase bucket name', help='Couchbase bucket to store data.')
@click.option('--cb-scope', prompt='Couchbase scope name', help='Couchbase scope name.')
@click.option('--cb-collection', prompt='Couchbase collection name', help='Couchbase collection name (optional).')
@click.option('--cb-batch-size', default=DEFAULT_BATCH_SIZE, type=int, help=f'Number of documents to insert per batch (default: {DEFAULT_BATCH_SIZE}).')
@click.option('--debug', is_flag=True, help='Enable debug output.')
@click.pass_context
def migrate(
    ctx, path, name, 
    #data_dir, 
    data_files, split, cache_dir, 
    #features, 
    download_config, download_mode,
    verification_mode, keep_in_memory, save_infos, revision, token, no_streaming, num_proc, storage_options,
    trust_remote_code, id_fields, cb_url, cb_username, cb_password, cb_bucket, cb_scope, cb_collection,
    debug, cb_batch_size):

    # this is code for mock the cli library for cbmigrate testing
    if os.getenv(MOCK_CLI_ENV_VAR, "false") == "true":
        pre_function(ctx)
        return

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    download_config_dict = None
    if download_config:
        try:
            download_config_dict = json.loads(download_config)
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing download_config JSON: {e}", err=True)
            sys.exit(1)


    click.echo(f"Starting migration of dataset '{path}' to Couchbase bucket '{cb_bucket}' scope '{cb_scope}' collection '{cb_collection}'...")
    migrator = DatasetMigrator(token=token)

    # Prepare data_files
    data_files = list(data_files) if data_files else None
    try:
        migrator.migrate_dataset(
            path=path,
            cb_url=cb_url,
            cb_username=cb_username,
            cb_password=cb_password,
            cb_bucket=cb_bucket,
            cb_scope=cb_scope,
            cb_collection=cb_collection,
            id_fields=id_fields,
            name=name,
            #data_dir=data_dir,
            data_files=data_files,
            split=split,
            cache_dir=cache_dir,
            #features=features,
            download_config=download_config_dict,
            download_mode=download_mode,
            verification_mode=verification_mode,
            keep_in_memory=keep_in_memory,
            save_infos=save_infos,
            revision=revision,
            token=token,
            streaming=not no_streaming,
            num_proc=num_proc,
            storage_options=json.loads(storage_options) if storage_options else None,
            trust_remote_code=trust_remote_code,
            batch_size=cb_batch_size,
        )
        click.echo("Migration completed successfully.")
    except Exception as e:
        click.echo(f"Migration failed: {str(e)}", err=True)
        sys.exit(1)
    

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(prog_name=prog_name)