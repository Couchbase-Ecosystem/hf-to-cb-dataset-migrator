# Hugging Face Dataset to Couchbase Migrator CLI

A command-line tool to interact with Hugging Face datasets and migrate them to Couchbase, with support for streaming data.

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Commands

The CLI provides the following commands:

### 1. List Configurations

Lists all available configurations for a dataset.

```bash
hf_to_cb_dataset_migrator list-configs --path dataset
```

Flags:

- `--path`: Path or name of the dataset (required)
- `--revision`: Version of the dataset script to load
- `--download-config`: Specific download configuration parameters
- `--download-mode`: Download mode (reuse_dataset_if_exists or force_redownload)
- `--token`: Authentication token for private datasets
- `--json-output`: Output the configurations in JSON format
- `--debug`: Enable debug output

### 2. List Splits

Lists all available splits for a dataset.

```bash
hf_to_cb_dataset_migrator list-splits --path dataset
```

Flags:

- `--path`: Path or name of the dataset (required)
- `--name`: Configuration name of the dataset
- `--data-files`: Path(s) to source data file(s)
- `--token`: Authentication token for private datasets
- `--json-output`: Output the splits in JSON format
- `--debug`: Enable debug output

### 3. List Fields

Lists all fields (columns) in a dataset.

```bash
hf_to_cb_dataset_migrator list-fields --path dataset
```

Flags:

- `--path`: Path or name of the dataset (required)
- `--name`: Name of the dataset configuration
- `--data-files`: Paths to source data files
- `--revision`: Version of the dataset script to load
- `--token`: Hugging Face token for private datasets
- `--json-output`: Output the fields in JSON format
- `--debug`: Enable debug output

### 4. Migrate Dataset

Migrates data from Hugging Face to Couchbase.

```bash
hf_to_cb_dataset_migrator migrate \
    --path dataset \
    --id-fields id_field \
    --cb-url couchbase://localhost \
    --cb-username user \
    --cb-password pass \
    --cb-bucket my_bucket \
    --cb-scope my_scope \
    --cb-collection my_collection
```

Flags:

- `--path`: Path or name of the dataset (required)
- `--id-fields`: Comma-separated list of field names to use as document ID (required)
- `--cb-url`: Couchbase cluster URL (required)
- `--cb-username`: Couchbase username (required)
- `--cb-password`: Couchbase password (required)
- `--cb-bucket`: Couchbase bucket name (required)
- `--cb-scope`: Couchbase scope name (required)
- `--name`: Configuration name of the dataset
- `--data-files`: Path(s) to source data file(s)
- `--split`: Which split of the data to load
- `--cache-dir`: Cache directory for datasets
- `--download-mode`: Download mode (reuse_dataset_if_exists or force_redownload)
- `--verification-mode`: Verification mode (no_checks, basic_checks, or all_checks)
- `--keep-in-memory`: Keep dataset in memory
- `--save-infos`: Save dataset information
- `--revision`: Version of the dataset script to load
- `--token`: Authentication token for private datasets
- `--no-streaming`: Disable streaming mode
- `--num-proc`: Number of processes to use
- `--trust-remote-code`: Allow execution of remote code
- `--cb-collection`: Couchbase collection name
- `--cb-batch-size`: Number of documents to insert per batch (default: 1000)
- `--debug`: Enable debug output

## Examples

- List configurations for a public dataset:

```bash
hf_to_cb_dataset_migrator list-configs --path dataset
```

- List configurations for a private dataset:

```bash
hf_to_cb_dataset_migrator list-configs --path my-dataset --token YOUR_HF_TOKEN
```

- List splits for a dataset with specific configuration:

```bash
hf_to_cb_dataset_migrator list-splits --path dataset --name config-name
```

- List fields in JSON format:

```bash
hf_to_cb_dataset_migrator list-fields --path dataset --json-output
```

- Migrate a dataset with multiple ID fields:

```bash
hf_to_cb_dataset_migrator migrate \
    --path dataset \
    --id-fields field1,field2 \
    --cb-url couchbase://localhost \
    --cb-username user \
    --cb-password pass \
    --cb-bucket my_bucket \
    --cb-scope my_scope \
    --cb-collection my_collection
```

- Migrate a specific split with streaming enabled:

```bash
hf_to_cb_dataset_migrator migrate \
    --path dataset \
    --split train \
    --id-fields id_field \
    --cb-url couchbase://localhost \
    --cb-username user \
    --cb-password pass \
    --cb-bucket my_bucket \
    --cb-scope my_scope
```

## Error Handling

The CLI will exit with a non-zero status code if an error occurs during execution. Error messages will be displayed on stderr.

## Logging

- Use `--debug` flag with any command to enable debug-level logging
- JSON output options are available for machine-readable output
- Progress information is displayed during migration

## Authentication

- For private Hugging Face datasets, use the `--token` option
- Couchbase credentials are required for migration operations
- Credentials can be provided via command-line options
