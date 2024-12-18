import logging
import uuid
import time
from datetime import timedelta, datetime
from typing import Any, Dict, List, Union, Optional, Sequence, Mapping, Callable

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
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset

logger = logging.getLogger(__name__)


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
        cb_bucket: str,
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
            bucket = self.cluster.bucket(cb_bucket)

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
            raise Exception(f"Failed to connect to Couchbase cluster: {e}")

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
        trust_remote_code: Optional[bool] = None,
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
        :param trust_remote_code: Allow loading arbitrary code from the dataset repository.
        :param config_kwargs: Additional keyword arguments for dataset configuration.
        :return: A list of configuration names if successful; None if an error occurs.
        """
        try:
            set_verbosity_error()  # Suppress warnings
            # Include API key if provided
            if self.token:
                config_kwargs['token'] = self.token

            if trust_remote_code is not None:
                config_kwargs['trust_remote_code'] = trust_remote_code

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
            raise Exception(f"Error listing configurations: {e}")

    def list_splits(
        self,
        path: str,
        config_name: Optional[str] = None,
        data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None] = None,
        download_config: Optional[Dict] = None,
        download_mode: Union[DownloadMode, str, None] = None,
        revision: Union[str, Version, None] = None,
        trust_remote_code: Optional[bool] = None,
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
        :param trust_remote_code: Allow loading arbitrary code from the dataset repository.
        :param config_kwargs: Additional keyword arguments for configuration.
        :return: A list of split names if successful; None if an error occurs.
        """
        try:
            set_verbosity_error()  # Suppress warnings
            # Include token if provided
            if self.token:
                config_kwargs['token'] = self.token

            if trust_remote_code is not None:
                config_kwargs['trust_remote_code'] = trust_remote_code

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
            raise Exception(f"Error listing splits: {e}")

    def list_fields(
        self,
        path: str,
        name: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        download_config: Optional[Dict] = None,
        revision: Optional[Union[str, Version]] = None,
        split: Optional[str] = None,
        **load_dataset_kwargs
    ) -> List[str]:
        """
        List the fields (columns) of a dataset.

        :param path: Path or name of the dataset.
        :param name: Name of the dataset configuration (optional).
        :param data_files: Paths to source data files (optional).
        :param download_config: Specific download configuration parameters
        :param revision: Version of the dataset script to load (optional).
        :param split: Which split of the data to load (optional).
        :param load_dataset_kwargs: Additional arguments to pass to load_dataset.
        :return: List of field names.
        """
        try:
            dataset = load_dataset(
                path=path,
                name=name,
                data_files=data_files,
                download_config=DownloadConfig(**download_config) if download_config else None,
                revision=revision,
                token=self.token if self.token else None,
                split=split,
                streaming=True,
                **load_dataset_kwargs
            )
            
            # If the dataset is an IterableDatasetDict (multiple splits)
            if isinstance(dataset, IterableDatasetDict):
                dataset_split = next(iter(dataset.values()))
            else:
                dataset_split = dataset

            # Get the first example to extract fields
            try:
                first_example = next(iter(dataset_split))
                fields = list(first_example.keys())
                return fields
            except StopIteration:
                raise Exception("Dataset is empty")
                
        except Exception as e:
            raise Exception(f"Error listing fields: {e}")
    
    def migrate_dataset(
        self,
        path: str,
        cb_url: str,
        cb_username: str,
        cb_password: str,
        cb_bucket: str,
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

        Args:
            path: Path or name of the dataset
            cb_url: Couchbase cluster URL (e.g., couchbase://localhost)
            cb_username: Username for Couchbase authentication
            cb_password: Password for Couchbase authentication
            cb_bucket: Couchbase bucket to store data
            cb_scope: Couchbase scope name
            cb_collection: Couchbase collection name
            id_fields: Comma-separated list of field names to use as document ID
            name: Configuration name of the dataset
            data_dir: Directory with the data files
            data_files: Path(s) to source data file(s)
            split: Which split(s) of the data to load
            cache_dir: Cache directory to store the datasets
            features: Set of features to use
            download_config: Specific download configuration parameters
            download_mode: Specifies the download mode
            verification_mode: Verification mode
            keep_in_memory: Whether to keep the dataset in memory
            save_infos: Whether to save dataset information
            revision: Version of the dataset script to load
            streaming: Whether to load the dataset in streaming mode
            num_proc: Number of processes to use
            storage_options: Storage options for remote filesystems
            trust_remote_code: Allow loading arbitrary code from the dataset repository
            batch_size: Number of documents to insert per batch
            **config_kwargs: Additional keyword arguments for dataset configuration

        Returns:
            bool: True if migration is successful, False otherwise

        Raises:
            ValueError: If id_fields are specified but not found in the dataset
            Exception: For other errors during migration
        """
        try:
            # Include token if provided
            if self.token:
                config_kwargs['token'] = self.token

            # Build load_dataset options
            load_kwargs = {
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
            }

            load_kwargs.update(config_kwargs)
            # Remove None values
            load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

            # Parse id_fields into a list
            id_fields_list = [field.strip() for field in id_fields.split(',')] if id_fields else None

            # Load the dataset from Hugging Face
            dataset = load_dataset(path, **load_kwargs)

            # Establish Couchbase connection
            self.connect(cb_url, cb_username, cb_password, cb_bucket, cb_scope, cb_collection)

            total_records = 0

            # Function to construct document ID
            def construct_doc_id(example: Dict[str, Any]) -> str:
                if id_fields_list:
                    try:
                        id_values = [str(example[field]) for field in id_fields_list]
                        return '_'.join(id_values)
                    except KeyError as e:
                        raise ValueError(f"Field '{e.args[0]}' not found in the dataset examples.")
                return str(uuid.uuid4())

            # Validate id_fields before processing
            if id_fields_list:
                if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                    first_dataset = next(iter(dataset.values()))
                else:
                    first_dataset = dataset
                
                missing_fields = [field for field in id_fields_list 
                                if field not in first_dataset.column_names]
                if missing_fields:
                    raise ValueError(f"The following id_fields are not present in the dataset: {missing_fields}")

            # Process the dataset
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                for split_name, split_dataset in dataset.items():
                    logger.info(f"Processing split '{split_name}'...")
                    total_records = self._process_and_insert_split(
                        split_dataset, split_name, construct_doc_id, batch_size, total_records
                    )
            else:
                split_name = None # No split name for single dataset
                logger.info(f"Processing Dataset '{path}'...")
                total_records = self._process_and_insert_split(
                    dataset, split_name, construct_doc_id, batch_size, total_records
                )

            logger.info(f"Migration completed successfully. Total records migrated: {total_records}")

        except Exception as e:
            import traceback
            print(f"Error: {e}\nTraceback:\n{traceback.format_exc()}")
            raise
        finally:
            self.close()

    def _convert_datetime(self, obj: Any) -> Any:
        """Recursively convert datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_datetime(item) for item in obj]
        return obj

    def _process_and_insert_split(
        self,
        split_dataset: Union[Dataset, IterableDataset],
        split_name: Optional[str],
        construct_doc_id: Callable[[Dict[str, Any]], str],
        batch_size: int,
        total_records: int,
    ) -> int:
        """
        Processes and inserts a dataset split into Couchbase.

        :param split_dataset: The dataset split to process (can be Dataset or IterableDataset)
        :param split_name: Name of the split.
        :param construct_doc_id: Function to construct document IDs.
        :param batch_size: Number of documents to insert per batch.
        :param total_records: Total records processed so far.
        :return: Updated total records count
        """
        try:
            batch: Dict[str, Any] = {}
            processed_count = 0
            
            # Handle both Dataset and IterableDataset
            # IterableDataset is used when streaming=True
            for example in split_dataset:
                # Convert example to dict if it's not already
                if not isinstance(example, dict):
                    example = dict(example)
                    
                # Convert datetime objects recursively
                example = self._convert_datetime(example)
                
                doc_id = construct_doc_id(example)
                example_with_split = dict(example)
                if split_name is not None:
                    example_with_split['split'] = split_name
                batch[doc_id] = example_with_split
                processed_count += 1

                # Batch insert when batch size is reached
                if len(batch) >= batch_size:
                    self.upsert_multi(batch)
                    batch = {}
                    logger.info(f"Processed {total_records + processed_count} records...")

            # Insert remaining documents in batch
            if batch:
                self.upsert_multi(batch)
                logger.info(f"Processed {total_records + processed_count} records...")

            return total_records + processed_count
        except Exception as e:
            raise Exception(f"Error processing split '{split_name}': {e}")

    def upsert_multi(self, batch: Dict[str, Any]) -> None:
        """
        Performs a batch insert operation using upsert_multi.

        :param batch: A dictionary where keys are document IDs and values are documents.
        """
        try:
            result: MultiMutationResult = self.collection.upsert_multi(batch)
        except CouchbaseException as e:
            raise Exception(f"Couchbase write error: {e}")

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
                raise Exception(f"Failed to write some documents to Couchbase: {other_errors}")