# hf_to_cb_dataset_migrator/cli.py

import click
import json
import os
from hf_to_cb_dataset_migrator.migration import DatasetMigrator
from typing import Any

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """CLI tool to interact with Hugging Face datasets and migrate them to Couchbase."""
    pass

@main.command('list-configs')
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--download-config', default=None, help='Specific download configuration parameters (optional).')
@click.option('--download-mode', type=click.Choice(['reuse_dataset_if_exists', 'force_redownload']), default=None,
              help='Download mode (optional).')
@click.option('--dynamic-modules-path', default=None, help='Path to dynamic modules (optional).')
@click.option('--data-files', default=None, multiple=True, help='Path(s) to source data file(s) (optional).')
@click.option('--token',default=None, help='Use authentication token for private datasets.')
@click.option('--json-output', is_flag=True, help='Output the configurations in JSON format.')
def list_configs_cmd(path, revision, download_config, download_mode, dynamic_modules_path,
                     data_files, token, json_output):
    """List all configuration names for a given dataset."""
    migrator = DatasetMigrator(token=token)
    download_kwargs = {
        'revision': revision,
        'download_config': json.load(download_config) if download_config else None,
        'download_mode': download_mode,
        'dynamic_modules_path': dynamic_modules_path,
        'data_files': data_files if data_files else None,
    }
    # Remove None values
    download_kwargs = {k: v for k, v in download_kwargs.items() if v is not None}

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

@main.command('list-splits')
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--name', 'config_name', default=None, help='Configuration name of the dataset (optional).')
@click.option('--data-files', default=None, multiple=True, help='Path(s) to source data file(s) (optional).')
@click.option('--download-config', default=None, help='Specific download configuration parameters (optional).')
@click.option('--download-mode', type=click.Choice(['reuse_dataset_if_exists', 'force_redownload']), default=None,
              help='Download mode (optional).')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--token', default=None, help='Authentication token for private datasets (optional).')
@click.option('--json-output', is_flag=True, help='Output the splits in JSON format.')
def list_splits_cmd(path, config_name, data_files, download_config, download_mode, revision, token,
                    json_output):
    """List all available splits for a given dataset and configuration."""
    migrator = DatasetMigrator(token=token)

    config_kwargs = {
        'data_files': data_files if data_files else None,
        'download_config': json.load(download_config) if download_config else None,
        'download_mode': download_mode,
        'revision': revision,
    }
    # Remove None values
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

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

@main.command('list-fields')
@click.option('--path', required=True, help='Path or name of the dataset.')
@click.option('--name', help='Name of the dataset configuration (optional).')
@click.option('--data-files', multiple=True, help='Paths to source data files (optional).')
@click.option('--revision', default=None, help='Version of the dataset script to load (optional).')
@click.option('--token', default=None, help='Hugging Face token for private datasets (optional).')
@click.option('--json-output', is_flag=True, help='Output the fields in JSON format.')
def list_fields(path, name, data_files, revision, token, json_output):
    """List the fields (columns) of a dataset."""
    migrator = DatasetMigrator(token=token)

    try:
        fields = migrator.list_fields(
            path=path,
            name=name,
            data_files=list(data_files) if data_files else None,
            revision=revision
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

@main.command()
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
@click.option('--streaming/--no-streaming', default=True, help='Load the dataset in streaming mode (default: True).')
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
@click.option('--cb-scope', default=None, help='Couchbase scope name (optional).')
@click.option('--cb-collection', default=None, help='Couchbase collection name (optional).')
def migrate(
    path, name, 
    #data_dir, 
    data_files, split, cache_dir, 
    #features, 
    download_config, download_mode,
    verification_mode, keep_in_memory, save_infos, revision, token, streaming, num_proc, storage_options,
    trust_remote_code, id_fields, cb_url, cb_username, cb_password, cb_bucket, cb_scope, cb_collection
):
    """Migrate datasets from Hugging Face to Couchbase."""
    click.echo(f"Starting migration of dataset '{path}' to Couchbase bucket '{cb_bucket}'...")
    migrator = DatasetMigrator(token=token)

    # Prepare data_files
    data_files = list(data_files) if data_files else None

    result = migrator.migrate_dataset(
        path=path,
        cb_url=cb_url,
        cb_username=cb_username,
        cb_password=cb_password,
        couchbase_bucket=cb_bucket,
        cb_scope=cb_scope,
        cb_collection=cb_collection,
        id_fields=id_fields,
        name=name,
        #data_dir=data_dir,
        data_files=data_files,
        split=split,
        cache_dir=cache_dir,
        #features=features,
        download_config=download_config,
        download_mode=download_mode,
        verification_mode=verification_mode,
        keep_in_memory=keep_in_memory,
        save_infos=save_infos,
        revision=revision,
        token=token,
        streaming=streaming,
        num_proc=num_proc,
        storage_options=json.loads(storage_options) if storage_options else None,
        trust_remote_code=trust_remote_code,
    )
    if result:
        click.echo("Migration completed successfully.")
    else:
        click.echo("Migration failed.")

if __name__ == '__main__':
    prog_name = "cbmigrate hugging-face" if os.getenv('RUN_FROM_CBMIGRATE', "false") == "true" else None
    main(prog_name=prog_name)