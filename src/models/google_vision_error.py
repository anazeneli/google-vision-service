# google_vision_error.py
"""
Simplified error handling module for Google Vision Service.
No external Google Cloud dependencies - handles errors generically.
"""

import asyncio
import functools
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec
from viam.logging import getLogger

P = ParamSpec('P')
T = TypeVar('T')

LOGGER = getLogger(__name__)


class GoogleVisionError(Exception):
    """Enhanced base exception for Google Vision service with context."""
    
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.error_code = error_code
        self.original_error = original_error
        
        # Build comprehensive error message
        full_message = f"GoogleVision: {message}"
        if error_code:
            full_message += f" (Code: {error_code})"
        if original_error:
            full_message += f" - Original: {str(original_error)}"
            
        super().__init__(full_message)
        
    def __repr__(self):
        return f"GoogleVisionError(message='{self.args[0]}', code='{self.error_code}')"


def handle_vision_errors(error_code: str = None, fallback_result: Any = None, reraise: bool = True):
    """Decorator to handle Google Vision API errors consistently."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if we're in proper asyncio context before doing async logging
                try:
                    # This will raise RuntimeError if not in event loop
                    loop = asyncio.get_running_loop()
                    current_task = asyncio.current_task(loop)
                    if current_task:
                        LOGGER.error(f"Error in {func.__name__}: {e}")
                    else:
                        # We're in event loop but not in a task - use non-async logging
                        print(f"Error in {func.__name__} (no task context): {e}")
                except RuntimeError:
                    # Not in any event loop - definitely use non-async logging
                    print(f"Error in {func.__name__} (no loop): {e}")
                
                # Re-raise the error
                if reraise:
                    raise GoogleVisionError(f"Error in {func.__name__}", error_code, e)
                return fallback_result        
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except GoogleVisionError:
                raise  # Re-raise our custom errors
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["google", "vision", "api", "cloud"]):
                    error = GoogleVisionError(
                        f"Google Vision API error in {func.__name__}: {str(e)}",
                        error_code or "API_ERROR",
                        e
                    )
                else:
                    error = GoogleVisionError(
                        f"Unexpected error in {func.__name__}",
                        error_code or "UNEXPECTED_ERROR",
                        e
                    )
                if reraise:
                    raise error
                LOGGER.error(f"Error in {func.__name__}: {e}")
                return fallback_result
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_camera_operation(timeout_seconds: float = 30.0):
    """Decorator specifically for camera operations with timeout handling."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timeout = kwargs.get('timeout', timeout_seconds)
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                raise GoogleVisionError(
                    f"Camera operation timed out after {timeout}s",
                    "CAMERA_TIMEOUT"
                )
            except Exception as e:
                raise GoogleVisionError(
                    f"Camera operation failed in {func.__name__}",
                    "CAMERA_OPERATION_FAILED",
                    e
                )
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Thread-safe decorator to validate inputs."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Cache the signature inspection at decoration time, not runtime
        import inspect
        try:
            sig = inspect.signature(func)
        except Exception:
            # Fallback for functions that can't be inspected
            sig = None
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if sig:
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    for param_name, validator in validators.items():
                        if param_name in bound_args.arguments:
                            value = bound_args.arguments[param_name]
                            if not validator(value):
                                raise GoogleVisionError(
                                    f"Invalid {param_name}: {value}",
                                    f"INVALID_{param_name.upper()}"
                                )
                except Exception as e:
                    # If inspection fails, skip validation rather than crash
                    LOGGER.warning(f"Validation skipped for {func.__name__}: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Error code constants
class ErrorCodes:
    CREDENTIALS_FAILED = "CREDENTIALS_FAILED"
    CREDENTIALS_NOT_FOUND = "CREDENTIALS_NOT_FOUND"
    CAMERA_NOT_CONFIGURED = "CAMERA_NOT_CONFIGURED"
    CAMERA_NOT_FOUND = "CAMERA_NOT_FOUND"
    CAMERA_TIMEOUT = "CAMERA_TIMEOUT"
    CAMERA_CAPTURE_FAILED = "CAMERA_CAPTURE_FAILED"
    CLIENT_NOT_INITIALIZED = "CLIENT_NOT_INITIALIZED"
    API_CALL_FAILED = "API_CALL_FAILED"
    VISION_API_ERROR = "VISION_API_ERROR"
    EMPTY_IMAGE_DATA = "EMPTY_IMAGE_DATA"
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    OCR_FAILED = "OCR_FAILED"
    DETECTION_FAILED = "DETECTION_FAILED"
    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
    INVALID_SERVICE_MODE = "INVALID_SERVICE_MODE"
    MISSING_CONFIGURATION = "MISSING_CONFIGURATION"


# Utility functions
def raise_camera_error(message: str, error_code: str = None):
    """Raise a camera-related error."""
    raise GoogleVisionError(message, error_code or ErrorCodes.CAMERA_NOT_CONFIGURED)


def raise_api_error(message: str, original_error: Exception = None):
    """Raise an API-related error."""
    raise GoogleVisionError(message, ErrorCodes.API_CALL_FAILED, original_error)


def raise_validation_error(param_name: str, value: Any):
    """Raise a validation error for invalid parameters."""
    raise GoogleVisionError(
        f"Invalid {param_name}: {value}",
        f"INVALID_{param_name.upper()}"
    )