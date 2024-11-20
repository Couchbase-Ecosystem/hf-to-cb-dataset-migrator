import logging
import uuid
import time
from datetime import timedelta
from typing import Any, Dict, List, Union, Optional, Sequence, Mapping

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.collection import Collection
from couchbase.management.collections import CollectionManager, CollectionSpec
from couchbase.exceptions import CouchbaseException, DocumentExistsException, ScopeAlreadyExistsException, CollectionAlreadyExistsException
from couchbase.options import ClusterOptions, KnownConfigProfiles
from couchbase.result import MultiMutationResult
from datasets import DatasetDict, IterableDatasetDict, Split
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from datasets.download import DownloadConfig, DownloadMode
from datasets.features import Features
from datasets.utils import Version
from datasets.utils.info_utils import VerificationMode
from datasets.utils.logging import set_verbosity_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetMigrator:
    def __init__(self, token: Optional[str] = None):
        """
        Initializes the DatasetMigrator with an optional Hugging Face token.

        :param token: Hugging Face Token for accessing private datasets (optional)
        """
        self.token = token
        self.cluster: Optional[Cluster] = None
        self.collection: Optional[Collection] = None

    def connect(
        self,
        cb_url: str,
        cb_username: str,
        cb_password: str,
        couchbase_bucket: str,
        cb_scope: Optional[str] = None,
        cb_collection: Optional[str] = None,
    ) -> None:
        """
        Establishes a connection to the Couchbase cluster and gets the collection.
        Creates the scope and collection if they do not exist.
        """
        try:
            cluster_opts = ClusterOptions(PasswordAuthenticator(cb_username, cb_password))
            cluster_opts.apply_profile(KnownConfigProfiles.WanDevelopment)
            self.cluster = Cluster(cb_url, cluster_opts)
            self.cluster.wait_until_ready(timedelta(seconds=60))  # Wait until cluster is ready
            bucket = self.cluster.bucket(couchbase_bucket)

            # Get the collection manager
            collection_manager = bucket.collections()

            if cb_scope and cb_collection:
                # Check if scope exists
                scopes = collection_manager.get_all_scopes()
                scope_names = [scope.name for scope in scopes]
                if cb_scope not in scope_names:
                    logger.info(f"Scope '{cb_scope}' does not exist. Creating scope.")
                    try:
                        collection_manager.create_scope(cb_scope)
                    except ScopeAlreadyExistsException:
                        logger.info(f"Scope '{cb_scope}' already exists.")

                    # Need to retrieve scopes again after creation
                    scopes = collection_manager.get_all_scopes()

                # Now check if collection exists in the scope
                # Find the scope
                for scope in scopes:
                    if scope.name == cb_scope:
                        break
                else:
                    raise Exception(f"Scope '{cb_scope}' not found after creation.")

                collection_names = [collection.name for collection in scope.collections]
                if cb_collection not in collection_names:
                    logger.info(f"Collection '{cb_collection}' does not exist in scope '{cb_scope}'. Creating collection.")
                    collection_spec = CollectionSpec(cb_collection, scope_name=cb_scope)
                    try:
                        collection_manager.create_collection(collection_spec)
                    except CollectionAlreadyExistsException:
                        logger.info(f"Collection '{cb_collection}' already exists in scope '{cb_scope}'.")

                    # Wait until the collection is ready
                    timeout = 30  # seconds
                    start_time = time.time()
                    while True:
                        scopes = collection_manager.get_all_scopes()
                        for scope in scopes:
                            if scope.name == cb_scope:
                                collection_names = [collection.name for collection in scope.collections]
                                if cb_collection in collection_names:
                                    break
                        else:
                            time.sleep(1)
                            if time.time() - start_time > timeout:
                                raise Exception(f"Collection '{cb_collection}' not available after creation.")
                            continue
                        break

                # Get the scope and collection
                scope = bucket.scope(cb_scope)
                self.collection = scope.collection(cb_collection)
            else:
                self.collection = bucket.default_collection()
        except CouchbaseException as e:
            logger.error(f"Failed to connect to Couchbase cluster: {e}")
            raise

    def close(self) -> None:
        """Closes the connection to the Couchbase cluster."""
        if self.cluster:
            self.cluster.close()
            self.cluster = None
            self.collection = None

    def list_configs(
        self,
        path: str,
        revision: Union[str, Version, None] = None,
        download_config: Optional[Dict] = None,
        download_mode: Union[DownloadMode, str, None] = None,
        dynamic_modules_path: Optional[str] = None,
        data_files: Union[Dict, List, str, None] = None,
        **config_kwargs: Any,
    ) -> Optional[List[str]]:
        """
        Lists all configuration names for a specified dataset.

        :param path: The path or name of the dataset.
        :param revision: The version or revision of the dataset script to load.
        :param download_config: Dictionary of download configuration parameters.
        :param download_mode: Specifies the download mode.
        :param dynamic_modules_path: Path to dynamic modules for custom processing.
        :param data_files: Paths to source data files.
        :param config_kwargs: Additional keyword arguments for dataset configuration.
        :return: A list of configuration names if successful; None if an error occurs.
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

    def list_splits(
        self,
        path: str,
        config_name: Optional[str] = None,
        data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None] = None,
        download_config: Optional[Dict] = None,
        download_mode: Union[DownloadMode, str, None] = None,
        revision: Union[str, Version, None] = None,
        **config_kwargs: Any,
    ) -> Optional[List[str]]:
        """
        Lists all available splits for a given dataset and configuration.

        :param path: Path or name of the dataset.
        :param config_name: Configuration name of the dataset.
        :param data_files: Path(s) to source data file(s).
        :param download_config: Specific download configuration parameters.
        :param download_mode: Specifies the download mode.
        :param revision: Version of the dataset script to load.
        :param config_kwargs: Additional keyword arguments for configuration.
        :return: A list of split names if successful; None if an error occurs.
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
                **config_kwargs,
            )
            return splits
        except Exception as e:
            logger.error(f"An error occurred while fetching splits: {e}")
            return None

    def list_fields(
        self,
        path: str,
        name: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[str] = None,
        **load_dataset_kwargs
    ) -> List[str]:
        """
        List the fields (columns) of a dataset.

        :param path: Path or name of the dataset.
        :param name: Name of the dataset configuration (optional).
        :param data_files: Paths to source data files (optional).
        :param revision: Version of the dataset script to load (optional).
        :param token: Hugging Face token for private datasets (optional).
        :param load_dataset_kwargs: Additional arguments to pass to load_dataset.
        :return: List of field names.
        """
        try:
            dataset = load_dataset(
                path=path,
                name=name,
                data_files=data_files,
                revision=revision,
                use_auth_token=token,
                **load_dataset_kwargs
            )
            # If the dataset is a DatasetDict (multiple splits), pick one split
            if isinstance(dataset, DatasetDict):
                dataset_split = next(iter(dataset.values()))
            else:
                dataset_split = dataset

            # Get the features (fields) of the dataset
            fields = list(dataset_split.column_names)
            return fields
        except Exception as e:
            logger.error(f"Failed to list fields for dataset '{path}': {e}")
            raise
    
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
        features: Optional[Features] = None,
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
        batch_size: int = 1000,
        **config_kwargs: Any,
    ) -> bool:
        """
        Migrates a Hugging Face dataset to Couchbase using batch insertion.

        :param path: Path or name of the dataset.
        :param cb_url: Couchbase cluster URL (e.g., couchbase://localhost).
        :param cb_username: Username for Couchbase authentication.
        :param cb_password: Password for Couchbase authentication.
        :param couchbase_bucket: Couchbase bucket to store data.
        :param cb_scope: Couchbase scope name.
        :param cb_collection: Couchbase collection name.
        :param id_fields: Comma-separated list of field names to use as document ID.
        :param name: Configuration name of the dataset.
        :param data_dir: Directory with the data files.
        :param data_files: Path(s) to source data file(s).
        :param split: Which split(s) of the data to load.
        :param cache_dir: Cache directory to store the datasets.
        :param features: Set of features to use.
        :param download_config: Specific download configuration parameters.
        :param download_mode: Specifies the download mode.
        :param verification_mode: Verification mode.
        :param keep_in_memory: Whether to keep the dataset in memory.
        :param save_infos: Whether to save dataset information.
        :param revision: Version of the dataset script to load.
        :param streaming: Whether to load the dataset in streaming mode.
        :param num_proc: Number of processes to use.
        :param storage_options: Storage options for remote filesystems.
        :param trust_remote_code: Allow loading arbitrary code from the dataset repository.
        :param batch_size: Number of documents to insert per batch.
        :param config_kwargs: Additional keyword arguments for dataset configuration.
        :return: True if migration is successful, False otherwise.
        """
        try:
            # Include token if provided
            if self.token:
                config_kwargs['token'] = self.token

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
                **config_kwargs,
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

            # Validate id_fields before processing
            if id_fields_list:
                missing_fields = [field for field in id_fields_list if field not in dataset.column_names]
                if missing_fields:
                    raise ValueError(f"The following id_fields are not present in the dataset: {missing_fields}")

            # If dataset is a dict (multiple splits), iterate over each split
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                for split_name, split_dataset in dataset.items():
                    logger.info(f"Processing split '{split_name}'...")
                    self._process_and_insert_split(
                        split_dataset, split_name, construct_doc_id, batch_size, total_records
                    )
            else:
                # Dataset is a single split
                split_name = str(split) if split else None
                logger.info(f"Processing split '{split_name}'...")
                self._process_and_insert_split(
                    dataset, split_name, construct_doc_id, batch_size, total_records
                )

            logger.info(f"Total records migrated: {total_records}")
            return True
        except Exception as e:
            logger.error(f"An error occurred during migration: {e}")
            return False
        finally:
            self.close()

    def _process_and_insert_split(
        self,
        split_dataset,
        split_name: Union[str, None],
        construct_doc_id,
        batch_size: int,
        total_records: int,
    ) -> None:
        """
        Processes and inserts a dataset split into Couchbase.

        :param split_dataset: The dataset split to process.
        :param split_name: Name of the split.
        :param construct_doc_id: Function to construct document IDs.
        :param batch_size: Number of documents to insert per batch.
        :param total_records: Total records processed so far.
        """
        batch = {}
        for example in split_dataset:
            doc_id = construct_doc_id(example)
            # Include split name in the document
            example_with_split = dict(example)
            example_with_split['split'] = split_name
            batch[doc_id] = example_with_split
            total_records += 1

            # Batch insert when batch size is reached
            if len(batch) >= batch_size:
                self.insert_multi(batch)
                batch.clear()
                logger.info(f"{total_records} records migrated...")

        # Insert remaining documents in batch
        if batch:
            self.insert_multi(batch)
            logger.info(f"{total_records} records migrated...")
            batch.clear()

    def insert_multi(self, batch: Dict[str, Any]) -> None:
        """
        Performs a batch insert operation using insert_multi.

        :param batch: A dictionary where keys are document IDs and values are documents.
        """
        try:
            result: MultiMutationResult = self.collection.insert_multi(batch)
        except CouchbaseException as e:
            logger.error(f"Couchbase write error: {e}")
            raise

        if not result.all_ok and result.exceptions:
            duplicate_ids = []
            other_errors = []
            for doc_id, ex in result.exceptions.items():
                if isinstance(ex, DocumentExistsException):
                    duplicate_ids.append(doc_id)
                else:
                    other_errors.append({"id": doc_id, "exception": str(ex)})
            if duplicate_ids:
                logger.warning(f"Documents with IDs already exist: {', '.join(duplicate_ids)}")
            if other_errors:
                logger.error(f"Errors occurred during batch insert: {other_errors}")
                raise Exception(f"Failed to write some documents to Couchbase: {other_errors}") 