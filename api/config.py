"""
Configuration Module
-------------------
This module manages the application configuration through environment variables.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field

# Load environment variables from .env file if it exists
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """
    Application settings class that loads variables from environment.
    
    Attributes:
        app_name: Name of the application
        debug: Debug mode flag
        api_prefix: Prefix for all API endpoints
        secret_key: Secret key for JWT token generation
        algorithm: Algorithm used for JWT token generation
        access_token_expire_minutes: Expiration time for access tokens in minutes
        model_path: Path to the trained model file
        allow_origins: List of allowed origins for CORS
    """
    app_name: str = Field("Complaint Detection API", env="APP_NAME")
    debug: bool = Field(False, env="DEBUG")
    api_prefix: str = Field("/api", env="API_PREFIX")
    
    # Security
    secret_key: str = Field(
        "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7", 
        env="SECRET_KEY"
    )
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Model
    model_path: str = Field(
        "output/models_20250320_124018/lstm_cnn_full.pt", 
        env="MODEL_PATH"
    )
    
    # CORS
    allow_origins: list = Field(["*"], env="ALLOW_ORIGINS")
    
    class Config:
        """Configuration for the Settings class."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings() 