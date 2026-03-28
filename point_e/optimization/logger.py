"""
Structured logging system for Point-E operations.
Provides detailed logging with performance metrics, error tracking, and audit trails.
"""

import logging
import logging.handlers
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Formats log records as structured JSON for better parsing and analysis.
    Falls back to human-readable format for console output.
    """
    
    def __init__(self, json_format: bool = False):
        super().__init__()
        self.json_format = json_format
    
    def format(self, record: logging.LogRecord) -> str:
        if self.json_format:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            
            # Add extra fields if present
            if hasattr(record, "metrics"):
                log_data["metrics"] = record.metrics
            if hasattr(record, "context"):
                log_data["context"] = record.context
            
            return json.dumps(log_data)
        else:
            # Human-readable format
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            return (
                f"[{timestamp}] {record.levelname:8s} "
                f"[{record.name}:{record.funcName}:{record.lineno}] "
                f"{record.getMessage()}"
            )


class PointELogger:
    """
    Centralized logging for Point-E operations.
    Supports file and console output with structured formatting.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Optional[Path] = None
    
    @classmethod
    def initialize(
        cls,
        log_dir: Optional[Path] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        json_format: bool = False,
    ) -> None:
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory for log files. None = logs to console only.
            console_level: Logging level for console output
            file_level: Logging level for file output
            json_format: Use JSON format for file logs
        """
        cls._log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (typically __name__)
        
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = StructuredFormatter(json_format=False)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log directory specified)
        if cls._log_dir:
            log_file = cls._log_dir / f"{name.replace('.', '_')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = StructuredFormatter(json_format=True)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


class MetricsLogger:
    """
    Logs performance metrics and benchmarks.
    Tracks timing, memory usage, and throughput.
    """
    
    def __init__(self, name: str = "point_e.metrics"):
        self.logger = PointELogger.get_logger(name)
    
    def log_timing(
        self,
        stage: str,
        duration: float,
        items_processed: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log timing for a processing stage.
        
        Args:
            stage: Name of the processing stage
            duration: Time taken in seconds
            items_processed: Number of items processed
            context: Additional context information
        """
        throughput = items_processed / duration if duration > 0 else 0
        
        metrics = {
            "stage": stage,
            "duration_seconds": round(duration, 3),
            "items_processed": items_processed,
            "throughput_items_per_sec": round(throughput, 2),
        }
        
        extra = {"metrics": metrics}
        if context:
            extra["context"] = context
        
        self.logger.info(
            f"{stage}: {duration:.3f}s ({throughput:.2f} items/s)",
            extra=extra,
        )
    
    def log_memory_usage(
        self,
        stage: str,
        memory_mb: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log memory usage for a processing stage."""
        metrics = {
            "stage": stage,
            "memory_mb": round(memory_mb, 2),
        }
        
        extra = {"metrics": metrics}
        if context:
            extra["context"] = context
        
        self.logger.info(f"{stage} memory: {memory_mb:.2f} MB", extra=extra)
    
    def log_batch_processing(
        self,
        batch_size: int,
        batch_index: int,
        total_batches: int,
        duration: float,
        items_in_batch: Optional[int] = None,
    ) -> None:
        """Log batch processing progress."""
        if items_in_batch is None:
            items_in_batch = batch_size
        
        throughput = items_in_batch / duration if duration > 0 else 0
        
        self.logger.info(
            f"Batch {batch_index}/{total_batches}: "
            f"{items_in_batch} items in {duration:.3f}s "
            f"({throughput:.2f} items/s)"
        )


class PerformanceContext:
    """
    Context manager for timing code blocks.
    Automatically logs timing information.
    """
    
    def __init__(
        self,
        stage_name: str,
        metrics_logger: Optional[MetricsLogger] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.stage_name = stage_name
        self.metrics_logger = metrics_logger or MetricsLogger()
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.metrics_logger.logger.debug(f"Starting: {self.stage_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.metrics_logger.logger.error(
                f"Failed: {self.stage_name} - {exc_val}"
            )
        else:
            self.metrics_logger.log_timing(
                self.stage_name,
                duration,
                context=self.context,
            )


# Global logger instance
_global_logger = PointELogger.get_logger(__name__)


def initialize_logging(
    log_dir: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """
    Initialize the global logging system.
    
    Args:
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
    """
    PointELogger.initialize(
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return PointELogger.get_logger(name)


def get_metrics_logger(name: str = "point_e.metrics") -> MetricsLogger:
    """Get a metrics logger instance."""
    return MetricsLogger(name)


def log_performance(
    stage_name: str,
    context: Optional[Dict[str, Any]] = None,
) -> PerformanceContext:
    """Create a context manager for timing a code block."""
    return PerformanceContext(stage_name, context=context)
