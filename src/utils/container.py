from dependency_injector import containers, providers
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis
import mlflow
import boto3
import logging

from src.config.app_config import ConfigurationManager
from src.database.crud import DatabaseManager
from src.services.training.distributed_trainer import DistributedTrainer
from src.services.registry.model_registry import ModelRegistry
from src.services.serving.pipeline_orchestrator import PipelineOrchestrator
from src.services.streaming.event_stream import EventStream
from src.services.features.feature_store import EnhancedFeatureStore
from src.services.scheduling.job_scheduler import BackgroundScheduler
from src.services.validation.data_validator import DataValidator
from src.services.error_handling.error_manager import ErrorManager

logger = logging.getLogger(__name__)

class Container(containers.DeclarativeContainer):
    # Configuration
    config = providers.Singleton(
        ConfigurationManager,
        config_path="config"
    )
    
    # Database
    database_engine = providers.Singleton(
        create_engine,
        config.provided.database.url,
        pool_size=config.provided.database.pool_size,
        max_overflow=config.provided.database.max_overflow
    )
    
    Session = providers.Singleton(
        sessionmaker,
        bind=database_engine
    )
    
    database = providers.Factory(
        DatabaseManager,
        session=Session
    )
    
    # Redis
    redis_client = providers.Singleton(
        redis.Redis,
        host=config.provided.redis.host,
        port=config.provided.redis.port,
        password=config.provided.redis.password,
        db=config.provided.redis.db
    )
    
    # AWS
    s3_client = providers.Singleton(
        boto3.client,
        's3',
        region_name=config.provided.s3.region,
        aws_access_key_id=config.provided.s3.access_key,
        aws_secret_access_key=config.provided.s3.secret_key
    )
    
    # MLflow
    mlflow_client = providers.Singleton(
        mlflow.tracking.MlflowClient,
        config.provided.mlflow.tracking_uri
    )
    
    # Services
    error_manager = providers.Singleton(
        ErrorManager,
        config=config.provided.error
    )
    
    data_validator = providers.Singleton(
        DataValidator,
        config=config.provided.validation
    )
    
    feature_store = providers.Singleton(
        EnhancedFeatureStore,
        config=config.provided.feature_store,
        redis_client=redis_client
    )
    
    model_registry = providers.Singleton(
        ModelRegistry,
        config=config.provided.registry,
        s3_client=s3_client,
        database=database
    )
    
    event_stream = providers.Singleton(
        EventStream,
        config=config.provided.stream
    )
    
    scheduler = providers.Singleton(
        BackgroundScheduler,
        config=config.provided.scheduler
    )
    
    trainer = providers.Singleton(
        DistributedTrainer,
        config=config.provided.training,
        mlflow_client=mlflow_client,
        model_registry=model_registry
    )
    
    pipeline = providers.Singleton(
        PipelineOrchestrator,
        config=config.provided.pipeline,
        redis_client=redis_client,
        model_registry=model_registry
    )