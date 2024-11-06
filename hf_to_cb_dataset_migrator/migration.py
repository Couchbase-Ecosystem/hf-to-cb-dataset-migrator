# my_cli/migration.py

import couchbase.collection
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from datasets.utils.logging import set_verbosity_error
from datasets.download import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from datasets.features import Features
from datasets.utils import Version
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset, Split
import couchbase
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, KnownConfigProfiles
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException, DocumentExistsException
from couchbase.result import MultiMutationResult
from datetime import timedelta
import uuid
import logging
from typing import Any, Dict, List, Union, Optional, Sequence, Mapping

logger = logging.getLogger(__name__)

class DatasetMigrator:
    def __init__(self, token: Optional[str] = None):
        """
        Initializes the DatasetMigrator with optional API key for Hugging Face.

        :param token: Hugging Face Token for accessing private datasets (optional)
        """
        self.token = token
        self.cluster = None
        self.collection = None

    def connect(self, cb_url: str, cb_username: str, cb_password: str, couchbase_bucket: str,
                cb_scope: Optional[str] = None, cb_collection: Optional[str] = None):
        """Establishes a connection to the Couchbase cluster and gets the collection."""
        cluster_opts = ClusterOptions(
            PasswordAuthenticator(cb_username, cb_password),
        )
        cluster_opts.apply_profile(KnownConfigProfiles.WanDevelopment)
        self.cluster = Cluster(cb_url, cluster_opts)
        self.cluster.wait_until_ready(timedelta(seconds=60))  # Wait until cluster is ready
        bucket = self.cluster.bucket(couchbase_bucket)

        # Get the collection
        if cb_scope and cb_collection:
            scope = bucket.scope(cb_scope)
            self.collection = scope.collection(cb_collection)
        else:
            self.collection = bucket.default_collection()

    def close(self):
        """Closes the connection to the Couchbase cluster."""
        if self.cluster:
            self.cluster.close()
            self.cluster = None
            self.collection = None

    def list_configs(self, path: str,
                     revision: Union[str, Version, None] = None,
                     download_config: Optional[Dict] = None,
                     download_mode: Union[DownloadMode, str, None] = None,
                     dynamic_modules_path: Optional[str] = None,
                     data_files: Union[Dict, List, str, None] = None,
                     **config_kwargs: Any) -> Optional[List[str]]:
        """
        Lists all configuration names for a specified dataset.
        Parameters:
            path (str): The path or name of the dataset.
            revision (Union[str, Version, None], optional): The version or revision of the dataset script to load.
            download_config (Optional[Dict], optional): Dictionary of download configuration parameters.
            download_mode (Union[DownloadMode, str, None], optional): Specifies the download mode.
            dynamic_modules_path (Optional[str], optional): Path to dynamic modules for custom processing.
            data_files (Union[Dict, List, str, None], optional): Paths to source data files.
            config_kwargs (Any): Additional keyword arguments for dataset configuration.

        Returns:
            Optional[List[str]]: A list of configuration names if successful; None if an error occurs.
        """
        try:
            set_verbosity_error()  # Suppress warnings
            # Include API key if provided
            if self.token:
                config_kwargs['token'] = self.token

            configs = get_dataset_config_names(
                path,
                revision=revision,
                download_config=DownloadConfig(**download_config) if download_config else None,
                download_mode=download_mode,
                dynamic_modules_path=dynamic_modules_path,
                data_files=data_files,
                **config_kwargs,
            )
            return configs
        except Exception as e:
            logger.error(f"An error occurred while fetching configs: {e}")
            return None

    def list_splits(self, path: str,
                    config_name: Optional[str] = None,
                    data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None] = None,
                    download_config: Optional[Dict] = None,
                    download_mode: Union[DownloadMode, str, None] = None,
                    revision: Union[str, Version, None] = None,
                    **config_kwargs: Any) -> Optional[List[str]]:
        """
        List all available splits for a given dataset and configuration.

        Parameters:
            path (str): Path or name of the dataset.
            config_name (str, optional): Configuration name of the dataset.
            data_files (Union[Dict, List, str, None], optional): Path(s) to source data file(s).
            download_config (Optional[Dict], optional): Specific download configuration parameters.
            download_mode (Union[DownloadMode, str, None], optional): Specifies the download mode.
            revision (Union[str, Version, None], optional): Version of the dataset script to load.
            config_kwargs (Any): Additional keyword arguments for configuration.

        Returns:
            Optional[List[str]]: A list of split names if successful; None if an error occurs.
        """
        try:
            set_verbosity_error()  # Suppress warnings
            # Include token if provided
            if self.token:
                config_kwargs['token'] = self.token

            splits = get_dataset_split_names(
                path,
                config_name=config_name,
                data_files=data_files,
                download_config=DownloadConfig(**download_config) if download_config else None,
                download_mode=download_mode,
                revision=revision,
                **config_kwargs
            )
            return splits
        except Exception as e:
            logger.error(f"An error occurred while fetching splits: {e}")
            return None

    def migrate_dataset(
        self,
        path: str,
        cb_url: str,
        cb_username: str,
        cb_password: str,
        couchbase_bucket: str,
        cb_scope: Optional[str] = None,
        cb_collection: Optional[str] = None,
        id_fields: Optional[str] = None,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        #features: Optional[Features] = None,
        download_config: Optional[Dict] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict] = None,
        trust_remote_code: Optional[bool] = None,
        **config_kwargs: Any,
    ) -> bool:
        """
        Migrates a Hugging Face dataset to Couchbase using batch insertion.

        This function accepts all parameters from the `load_dataset` function to customize dataset loading.

        Parameters:
            path (str): Path or name of the dataset.
            cb_url (str): Couchbase cluster URL (e.g., couchbase://localhost).
            cb_username (str): Username for Couchbase authentication.
            cb_password (str): Password for Couchbase authentication.
            couchbase_bucket (str): Couchbase bucket to store data.
            cb_scope (str, optional): Couchbase scope name.
            cb_collection (str, optional): Couchbase collection name.
            id_fields (str): Comma-separated list of field names to use as document ID.
            name (str, optional): Configuration name of the dataset.
            data_dir (str, optional): Directory with the data files.
            data_files (Union[Dict, List, str, None], optional): Path(s) to source data file(s).
            split (str, optional): Which split of the data to load.
            cache_dir (str, optional): Cache directory to store the datasets.
            features (Optional[Features], optional): Set of features to use.
            download_config (Optional[Dict], optional): Specific download configuration parameters.
            download_mode (Union[DownloadMode, str, None], optional): Specifies the download mode.
            verification_mode (str, optional): Verification mode.
            keep_in_memory (bool, optional): Whether to keep the dataset in memory.
            save_infos (bool, default=False): Whether to save dataset information.
            revision (Union[str, Version, None], optional): Version of the dataset script to load.
            streaming (bool, default=False): Whether to load the dataset in streaming mode.
            num_proc (int, optional): Number of processes to use.
            storage_options (Dict, optional): Storage options for remote filesystems.
            trust_remote_code (bool, optional): Allow loading arbitrary code from the dataset repository.
            config_kwargs (Any): Additional keyword arguments for dataset configuration.

        Returns:
            bool: True if migration is successful, False otherwise.
        """
        try:
            # Include token if provided
            if self.token:
                config_kwargs['token'] = self.token
            print(config_kwargs)
            # Prepare parameters for load_dataset
            load_kwargs = {
                'path': path,
                'name': name,
                'data_dir': data_dir,
                'data_files': data_files,
                'split': split,
                'cache_dir': cache_dir,
                #'features': features,
                'download_config': DownloadConfig(**download_config) if download_config else None,
                'download_mode': download_mode,
                'verification_mode': verification_mode,
                'keep_in_memory': keep_in_memory,
                'save_infos': save_infos,
                'revision': revision,
                'streaming': streaming,
                'num_proc': num_proc,
                'storage_options': storage_options,
                'trust_remote_code': trust_remote_code,
                **config_kwargs
            }

            # Remove None values
            load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

            # Parse id_fields into a list
            if id_fields:
                id_fields_list = [field.strip() for field in id_fields.split(',') if field.strip()]
                if not id_fields_list:
                    id_fields_list = None
            else:
                id_fields_list = None

            # Load the dataset from Hugging Face
            dataset = load_dataset(**load_kwargs)

            # Establish Couchbase connection
            self.connect(cb_url, cb_username, cb_password, couchbase_bucket, cb_scope, cb_collection)

            total_records = 0

            # Function to construct document ID
            def construct_doc_id(example):
                if id_fields_list:
                    try:
                        id_values = [str(example[field]) for field in id_fields_list]
                        doc_id = '_'.join(id_values)
                        return doc_id
                    except KeyError as e:
                        raise ValueError(f"Field '{e.args[0]}' not found in the dataset examples.")
                else:
                    return str(uuid.uuid4())

            # If dataset is a dict (multiple splits), iterate over each split
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                for split_name, split_dataset in dataset.items():
                    print(f"Processing split '{split_name}'...")
                    batch = {}
                    for example in split_dataset:
                        doc_id = construct_doc_id(example)
                        # Include split name in the document
                        example_with_split = dict(example)
                        example_with_split['split'] = split_name
                        batch[doc_id] = example_with_split
                        total_records += 1

                        # Batch insert every 1000 documents
                        if len(batch) >= 1000:
                            self.insert_multi(batch)
                            batch.clear()
                            print(f"{total_records} records migrated...")
                    # Insert remaining documents in batch
                    if batch:
                        self.insert_multi(batch)
                        print(f"{total_records} records migrated...")
                        batch.clear()
            else:
                # Dataset is a single split
                split_name = str(split) if split else 'unspecified'
                print(f"Processing split '{split_name}'...")
                batch = {}
                for example in dataset:
                    doc_id = construct_doc_id(example)
                    example_with_split = dict(example)
                    example_with_split['split'] = split_name
                    batch[doc_id] = example_with_split
                    total_records += 1

                    # Batch insert every 1000 documents
                    if len(batch) >= 1000:
                        self.insert_multi(batch)
                        batch.clear()
                        print(f"{total_records} records migrated...")
                # Insert remaining documents in batch
                if batch:
                    self.insert_multi(batch)
                    print(f"{total_records} records migrated...")
                    batch.clear()

            print(f"Total records migrated: {total_records}")
            return True
        except Exception as e:
            logger.error(f"An error occurred during migration: {e}")
            return False
        finally:
            self.close()

    def insert_multi(self, batch):
        """
        Performs a batch insert operation using insert_multi.

        :param batch: A dictionary where keys are document IDs and values are documents
        """
        try:
            result: MultiMutationResult = self.collection.insert_multi(batch)
        except Exception as e:
            logger.error(f"Write error: {e}")
            msg = f"Failed to write documents to Couchbase. Error: {e}"
            raise Exception(msg) from e

        if not result.all_ok and result.exceptions:
            duplicate_ids = []
            other_errors = []
            for doc_id, ex in result.exceptions.items():
                if isinstance(ex, DocumentExistsException):
                    duplicate_ids.append(doc_id)
                else:
                    other_errors.append({"id": doc_id, "exception": ex})
            if duplicate_ids:
                msg = f"IDs '{', '.join(duplicate_ids)}' already exist in the document store."
                raise Exception(msg)
            if other_errors:
                msg = f"Failed to write documents to Couchbase. Errors:\n{other_errors}"
                raise Exception(msg)