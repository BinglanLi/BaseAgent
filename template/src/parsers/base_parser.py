"""
BaseParser: Abstract base class for all data parsers.

This class provides common functionality for downloading, caching,
and parsing data from external sources.
"""

import os
import logging
import requests
import gzip
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    Abstract base class for data parsers.
    
    All data source parsers should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            data_dir: Directory to store downloaded/cached data files.
                     If None, uses default location.
        """
        if data_dir is None:
            # Use default data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "data")
        
        self.data_dir = Path(data_dir)
        self.source_name = self.__class__.__name__.replace('Parser', '').lower()
        self.source_dir = self.data_dir / self.source_name
        
        # Create directories if they don't exist
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self.force = False

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Data directory: {self.source_dir}")
    
    @abstractmethod
    def download_data(self) -> bool:
        """
        Download data from the source.
        
        Returns:
            True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse the downloaded data.
        
        Returns:
            Dictionary mapping entity types to DataFrames containing parsed data.
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """
        Get the schema for parsed data.
        
        Returns:
            Dictionary mapping entity types to their column schemas.
        """
        pass
    
    def download_file(self, url: str, filename: str) -> Optional[str]:
        """
        Download a file from a URL.

        Args:
            url: URL to download from
            filename: Name to save the file as

        Returns:
            Path to downloaded file, or None if failed
        """
        filepath = self.source_dir / filename

        if filepath.exists() and not self.force:
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
        
        try:
            logger.info(f"Downloading from {url} to {filepath}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✓ Downloaded to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def extract_gzip(self, gz_path: str) -> Optional[str]:
        """
        Extract a gzipped file.

        Args:
            gz_path: Path to the .gz file

        Returns:
            Path to extracted file, or None if failed
        """
        if not gz_path.endswith('.gz'):
            logger.warning(f"File does not appear to be gzipped: {gz_path}")
            return gz_path

        output_path = Path(gz_path).with_suffix('')  # Remove .gz extension

        if output_path.exists() and not self.force:
            logger.info(f"Extracted file already exists: {output_path}")
            return str(output_path)

        try:
            logger.info(f"Extracting {gz_path}")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            logger.info(f"✓ Extracted to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to extract {gz_path}: {e}")
            return None
    
    def read_tsv(self, filepath: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Read a TSV file into a DataFrame.
        
        Args:
            filepath: Path to the TSV file
            **kwargs: Additional arguments for pd.read_csv
        
        Returns:
            DataFrame, or None if failed
        """
        try:
            df = pd.read_csv(filepath, sep='\t', **kwargs)
            logger.info(f"✓ Read {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return None
    
    def read_csv(self, filepath: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Read a CSV file into a DataFrame.
        
        Args:
            filepath: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv
        
        Returns:
            DataFrame, or None if failed
        """
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"✓ Read {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that a DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            True if valid, False otherwise
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True
    
    def get_file_path(self, filename: str) -> str:
        """Get full path for a file in the source directory."""
        return str(self.source_dir / filename)
