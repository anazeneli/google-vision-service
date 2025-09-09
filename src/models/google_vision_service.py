import os, sys
from typing import ClassVar, Tuple, List, Mapping, Optional, Sequence, Dict, Any
import hashlib

from typing_extensions import Self
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import (Classification, Detection,
                                       GetPropertiesResponse)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.components.camera import Camera

from viam.utils import ValueTypes
from viam.utils import struct_to_dict
from viam.logging import getLogger

# Google Vision imports
from google.cloud import vision


# Import error handling module
from .google_vision_error import (
    GoogleVisionError, 
    ErrorCodes,
    handle_vision_errors,
    safe_camera_operation,
    validate_inputs,
    raise_camera_error,
    raise_api_error,
    raise_validation_error
)

LOGGER = getLogger(__name__)


class GoogleVisionService(Vision, EasyResource):
    """
    Google Vision AI service for Viam robotics platform.
    
    Provides OCR (Optical Character Recognition) capabilities using Google Cloud Vision API.
    Extracts text from images with bounding boxes and confidence scores.
    
    Supports multiple authentication methods:
    - Service account JSON file
    - GOOGLE_APPLICATION_CREDENTIALS environment variable  
    - Default credentials (gcloud CLI)
    """
    
    MODEL: ClassVar[Model] = Model(
        ModelFamily("azeneli", "google-vision"), "google-vision-service"
    )
    # Service mode definitions
    SUPPORTED_SERVICE_MODES = {
        "ocr_basic": {
            "api_method": "document_text_detection",
            "config_params": ["ocr_languages", "max_results"],
            "description": "Basic OCR - extract text with word-level bounding boxes"
        }
        # Add more modes here later
    }

    def __init__(self, name: str):
        super().__init__(name)
        self.logger = LOGGER 
        
        # Camera
        self.camera = None
        
        # Google vision client
        self.client = None
        self.credential_method = None
        
        # General configuration    
        self.service_mode = "ocr_basic"
        
        # Limit API calls if same image rendered 
        self.last_image_hash = None
        self.last_detections = []


    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def _validate_camera(cls, attrs: Dict[str, Any], required_dependencies: list):
        """
        Validate camera component configuration.
        
        Args:
            attrs: Configuration attributes
            required_dependencies: List to append camera dependency to
            
        Raises:
            ValueError: If camera_name is missing or invalid
        """
        camera_name = attrs.get("camera_name")
        if not camera_name or not isinstance(camera_name, str):
            raise ValueError("'camera_name' is required and must be a string.")
        required_dependencies.append(camera_name)
        LOGGER.info(f"Validated camera dependency: {camera_name}")

    @classmethod 
    def _validate_credentials(cls, attrs: Dict[str, Any]):
        """
        Validate Google Cloud credentials configuration.
        
        Checks for credentials in this order:
        1. Service account JSON file path
        2. GOOGLE_APPLICATION_CREDENTIALS environment variable
        3. Default credentials (gcloud CLI)
        
        Args:
            attrs: Configuration attributes
            
        Raises:
            ValueError: If credentials_path is not a string
        """
        credentials_path = attrs.get("credentials_path")
        
        # Check available credential methods
        has_credentials_file = False
        has_env_credentials = bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        
        if credentials_path:
            if not isinstance(credentials_path, str):
                raise ValueError("'credentials_path' must be a string")
            
            has_credentials_file = os.path.exists(credentials_path)
            if not has_credentials_file:
                LOGGER.warning(f"Credentials file not found: {credentials_path}")
        
        if not (has_credentials_file or has_env_credentials):
            # Still allow validation to pass if gcloud default credentials might be available
            LOGGER.info("No explicit credentials found. Will attempt to use default credentials (gcloud CLI)")
        
        if has_credentials_file:
            LOGGER.info(f"Found credentials file: {credentials_path}")
        elif has_env_credentials:
            LOGGER.info("Found GOOGLE_APPLICATION_CREDENTIALS environment variable")
            
    
    # Single validation method that routes based on service mode:
    @classmethod
    def _validate_service_mode_and_config(cls, attrs: Dict[str, Any]):
        """
        Validate service mode and mode-specific configuration.
        
        Args:
            attrs: Configuration attributes
            
        Raises:
            ValueError: If service_mode is missing or invalid
        """
        service_mode = attrs.get("service_mode") # Intentionally has no default
        
        if not service_mode:
            raise ValueError(f"'service_mode' is required. Must be one of: {list(cls.SUPPORTED_SERVICE_MODES.keys())}")
    
        # Validate service mode exists
        if service_mode not in cls.SUPPORTED_SERVICE_MODES:  
            valid_modes = list(cls.SUPPORTED_SERVICE_MODES.keys())
            raise ValueError(f"'service_mode' must be one of: {valid_modes}")
        
        # Validate mode-specific configuration
        if service_mode == "ocr_basic":
            cls._validate_ocr_basic_config(attrs)
        
        LOGGER.info(f"Service mode and configuration validated: {service_mode}")
    
    @classmethod
    def _validate_ocr_basic_config(cls, attrs: Dict[str, Any]):
        """
        Validate OCR-specific configuration parameters.
        
        Args:
            attrs: Configuration attributes
            
        Raises:
            ValueError: If OCR parameters are invalid
        """
        ocr_languages = attrs.get("ocr_languages")
        if ocr_languages is not None:
            if not isinstance(ocr_languages, list):
                raise ValueError("'ocr_languages' must be a list of language codes")
            # Optional: validate language codes are valid ISO codes
            # if not all(len(lang) == 2 for lang in ocr_languages):
            #     raise ValueError("Language codes must be 2-character ISO codes")
        
        max_results = attrs.get("max_results")
        if max_results is not None:
            if not isinstance(max_results, int) or max_results < 1 or max_results > 1000:
                raise ValueError("'max_results' must be an integer between 1 and 1000")

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        optional_dependencies, required_dependencies = [], []
        attrs = struct_to_dict(config.attributes)
        
        cls._validate_camera(attrs, required_dependencies)
        cls._validate_credentials(attrs)
        cls._validate_service_mode_and_config(attrs) 

        return required_dependencies, optional_dependencies

    def _reconfigure_camera(self, attrs: Dict[str, Any], dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        """
        Configure camera component during service reconfiguration.
        
        Args:
            attrs: Configuration attributes
            dependencies: Available component dependencies
            
        Raises:
            GoogleVisionError: If camera configuration fails
        """
        camera_name = attrs.get("camera_name")
        
        if not camera_name:
            raise_camera_error("Camera name is required but not provided", ErrorCodes.MISSING_CONFIGURATION)
        
        camera_resource_name = Camera.get_resource_name(camera_name)
        camera = dependencies.get(camera_resource_name)
        
        if not camera:
            raise_camera_error(f"Camera '{camera_name}' not found in dependencies", ErrorCodes.CAMERA_NOT_FOUND)
        
        self.camera_name = camera_name
        self.camera = camera
        self.logger.info(f"Camera configured: {camera_name}")


    @handle_vision_errors(error_code=ErrorCodes.CREDENTIALS_FAILED)
    def _reconfigure_credentials(self, attrs: Dict[str, Any]) -> None:
        """
        Configure Google Cloud Vision API credentials.
        
        Attempts authentication in this order:
        1. Service account JSON file
        2. GOOGLE_APPLICATION_CREDENTIALS environment variable
        3. Default credentials (gcloud CLI)
        
        Args:
            attrs: Configuration attributes
            
        Raises:
            GoogleVisionError: If all credential methods fail
        """
        credentials_path = attrs.get("credentials_path")
        
        # Method 1: Service account file
        if credentials_path:
            if not os.path.exists(credentials_path):
                raise GoogleVisionError(
                    f"Credentials file not found: {credentials_path}",
                    ErrorCodes.CREDENTIALS_NOT_FOUND
                )
            
            self.client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
            self.credential_method = "service_account_file"
            self.logger.info(f"Using service account file: {credentials_path}")
            return
        
        # Method 2: Environment variable
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            self.client = vision.ImageAnnotatorClient()
            self.credential_method = "environment_variable"
            self.logger.info("Using GOOGLE_APPLICATION_CREDENTIALS")
            return
        
        # Method 3: Default credentials
        self.client = vision.ImageAnnotatorClient()
        self.credential_method = "default_credentials"
        self.logger.info("Using default credentials (gcloud CLI)")
            
    def _reconfigure_service_settings(self, attrs: Dict[str, Any]) -> None:
        """
        Configure service-specific settings.
        
        Args:
            attrs: Configuration attributes
        """
        self.service_mode = attrs.get("service_mode", "ocr_basic")
        self.max_results = attrs.get("max_results")  # Add this
        self.ocr_languages = attrs.get("ocr_languages")  # Add this too
        self.logger.info(f"Service configured: {self.service_mode}")
        
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        
        """Initialize Google Vision API client with credentials."""
        self.logger.info("Configuring Google Vision API client...")
        attrs = struct_to_dict(config.attributes)
        # Camera
        self._reconfigure_camera(attrs, dependencies)
        
        # Reconfigure Vision API client  
        self._reconfigure_credentials(attrs)
        
        if self.client:
            self.logger.info("Google Vision API client initialized successfully")
            
        # Service settings
        self._reconfigure_service_settings(attrs)
        
        return super().reconfigure(config, dependencies)

    @validate_inputs(
        image=lambda x: x is not None
    )
    def _viam_image_to_bytes(self, image: ViamImage) -> bytes:
        """Convert ViamImage to bytes with error handling."""
        try:
            if hasattr(image, 'data') and image.data:
                return image.data
            elif hasattr(image, 'bytes_jpeg'):
                return image.bytes_jpeg()
            elif hasattr(image, 'bytes_png'):
                return image.bytes_png()
            else:
                return bytes(image)
        except Exception as e:
            raise GoogleVisionError(
                "Failed to convert image to bytes", 
                ErrorCodes.INVALID_IMAGE_FORMAT, 
                e
            )
        
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Generate hash of image to detect changes."""
        return hashlib.md5(image_bytes).hexdigest()

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        """
        Get detections from the configured camera.
        
        Args:
            camera_name: Name of camera (uses configured camera)
            extra: Additional parameters
            timeout: Operation timeout in seconds
            
        Returns:
            List of detected objects/text with bounding boxes
            
        Raises:
            GoogleVisionError: If camera not configured or operation fails
        """
        if not self.camera:
            raise_camera_error("Camera not configured")
        
        image = await self.camera.get_image()
        return await self.get_detections(image, extra=extra, timeout=timeout)

    @handle_vision_errors(error_code=ErrorCodes.DETECTION_FAILED)
    @validate_inputs(
        image=lambda x: x is not None
    )
    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        """
        Get detections from an image using Google Vision API.
        
        Args:
            image: Image to analyze
            extra: Additional parameters
            timeout: Operation timeout in seconds
            
        Returns:
            List of detected objects/text with bounding boxes and confidence scores
            
        Raises:
            GoogleVisionError: If analysis fails
        """
        if not self.client:
            raise GoogleVisionError("Vision client not initialized", ErrorCodes.CLIENT_NOT_INITIALIZED)
        
        image_bytes = self._viam_image_to_bytes(image)

        # Check if image has changed
        current_hash = self._get_image_hash(image_bytes)
        if current_hash == self.last_image_hash:
            self.logger.info("Image unchanged, returning cached detections")
            return self.last_detections

        # Image changed, process it based on service mode
        if self.service_mode == "ocr_basic":
            detections = await self._get_ocr_detections(image_bytes)
        else:
            self.logger.warning(f"Service mode '{self.service_mode}' doesn't support detections")
            detections = []

        # Cache the results regardless of service mode
        self.last_image_hash = current_hash
        self.last_detections = detections

        return detections

    async def get_classifications(self, image: ViamImage, count: int, *, extra=None, timeout=None) -> List[Classification]:
        """OCR service does not support classifications."""
        self.logger.warning("OCR service doesn't support classifications")
        return []

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra=None, timeout=None) -> List[Classification]:
        """OCR service does not support classifications."""
        self.logger.warning("OCR service doesn't support classifications")
        return []
 
    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> CaptureAllResult:
        """
        Capture image and perform all requested vision analyses.
        
        Args:
            camera_name: Name of camera (uses configured camera)
            return_image: Whether to include captured image in results
            return_classifications: Whether to perform classification analysis
            return_detections: Whether to perform detection analysis
            return_object_point_clouds: Whether to include point clouds (not supported)
            extra: Additional parameters (e.g., classification_count)
            timeout: Operation timeout in seconds
            
        Returns:
            CaptureAllResult containing requested analysis results
            
        Raises:
            GoogleVisionError: If camera not configured or capture fails
            
        Note:
            If individual analyses fail, they return empty lists rather than
            failing the entire operation (graceful degradation).
        """
        if not self.camera:
            raise_camera_error("Camera not configured")
        
        result = CaptureAllResult()
        
        image = await self.camera.get_image(mime_type="image/jpeg")
        
        if return_image:
            result.image = image
        
        if return_detections:
            try:
                result.detections = await self.get_detections(image, extra=extra, timeout=timeout)
                self.logger.info(f"Captured {len(result.detections)} detections")
            except GoogleVisionError as e:
                self.logger.error(f"Failed to get detections: {e}")
                result.detections = []
        
        if return_classifications:
            try:
                classification_count = 10
                if extra and "classification_count" in extra:
                    classification_count = int(extra["classification_count"])
                
                result.classifications = await self.get_classifications(
                    image, classification_count, extra=extra, timeout=timeout
                )
                self.logger.info(f"Captured {len(result.classifications)} classifications")
            except GoogleVisionError as e:
                self.logger.error(f"Failed to get classifications: {e}")
                result.classifications = []
        
        if return_object_point_clouds:
            self.logger.warning("Point clouds not supported by Google Vision API")
            result.objects = []
        
        return result

    @handle_vision_errors(error_code=ErrorCodes.OCR_FAILED)
    @validate_inputs(
        image_bytes=lambda x: x is not None and len(x) > 0
    )
    async def _get_ocr_detections(self, image_bytes: bytes) -> List[Detection]:
        vision_image = vision.Image(content=image_bytes)
        response = self.client.document_text_detection(image=vision_image)
        
        if response.error.message:
            raise_api_error(f"Vision API error: {response.error.message}")
        
        if not response.text_annotations:
            self.logger.info("No text detected in image")
            return []
        
        self.logger.info(f"Total annotations: {len(response.text_annotations)}")
        
        detections = []
        overall_confidence = 0.5
        
        # Process individual text blocks (skip first annotation which is full text)
        for i, annotation in enumerate(response.text_annotations[1:]):
            text_content = annotation.description
            self.logger.debug(f"Processing annotation {i+1}: '{text_content}'")
            
            try:
                if not (annotation.bounding_poly and annotation.bounding_poly.vertices):
                    self.logger.warning(f"Annotation {i+1} missing bounding box")
                    continue
                
                vertices = annotation.bounding_poly.vertices
                x_coords = [v.x for v in vertices if hasattr(v, 'x')]
                y_coords = [v.y for v in vertices if hasattr(v, 'y')]
                
                if not x_coords or not y_coords:
                    self.logger.warning(f"Annotation {i+1} has invalid coordinates")
                    continue
                
                detection = Detection(
                    x_min=min(x_coords),
                    y_min=min(y_coords),
                    x_max=max(x_coords),
                    y_max=max(y_coords),
                    confidence=overall_confidence,
                    class_name=text_content  # This should contain the actual text
                )
                
                self.logger.debug(f"Created Detection with class_name: '{detection.class_name}'")
                detections.append(detection)
                
            except Exception as e:
                self.logger.warning(f"Failed to process annotation {i+1}: {e}")
                continue
        
        # Apply max_results limit if configured
        if self.max_results and len(detections) > self.max_results:
            detections = detections[:self.max_results]
            self.logger.debug(f"Limited results to {self.max_results} detections")

        self.logger.info(f"Returning {len(detections)} detections")
        return detections
            
        
    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        """
        Get 3D point clouds from objects in the camera view.
        
        Args:
            camera_name: Name of camera
            extra: Additional parameters
            timeout: Operation timeout in seconds
            
        Returns:
            List of point cloud objects
            
        Raises:
            NotImplementedError: Google Vision API doesn't support point clouds
            
        Note:
            Google Vision API is a 2D image analysis service and does not
            provide 3D point cloud data. This method always raises
            NotImplementedError.
        """
        self.logger.error("get_object_point_clouds is not implemented - Google Vision API doesn't support point clouds")
        raise NotImplementedError("Point clouds not supported by Google Vision API")

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:
        """
        Get the capabilities of this vision service.
        
        Args:
            extra: Additional parameters
            timeout: Operation timeout in seconds
            
        Returns:
            Vision.Properties describing supported capabilities
            
        Note:
            Capabilities depend on the configured service_mode:
            - ocr_basic: supports detections only
            - Other modes may support different combinations
        """
        if self.service_mode == "ocr_basic":
            return Vision.Properties(
                classifications_supported=False,
                detections_supported=True,
                object_point_clouds_supported=False
            )
        else:
            return Vision.Properties(
                classifications_supported=False,
                detections_supported=False,
                object_point_clouds_supported=False
            )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        """
        Execute custom commands for service management and diagnostics.
        
        Args:
            command: Command dictionary with 'command' key specifying action
            timeout: Operation timeout in seconds
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing command results
            
        Raises:
            GoogleVisionError: If command execution fails
            
        Supported commands:
            - get_config: Returns current service configuration
            - health_check: Returns service health status
            - test_credentials: Tests Google Cloud credentials validity
        """
        if not self.client:
            raise GoogleVisionError("Vision client not initialized", ErrorCodes.CLIENT_NOT_INITIALIZED)
        
        cmd = command.get("command")
        if not cmd:
            raise_validation_error("command", None)
        
        try:
            if cmd == "get_config":
                return {
                    "service_mode": self.service_mode,
                    "credential_method": self.credential_method,
                    "camera_name": getattr(self, 'camera_name', None)
                }
            elif cmd == "health_check":
                return {
                    "status": "healthy",
                    "client_initialized": self.client is not None,
                    "camera_configured": self.camera is not None,
                    "service_mode": self.service_mode
                }
            elif cmd == "test_credentials":
                test_image = vision.Image(content=b'\x89PNG\r\n\x1a\n')
                try:
                    self.client.label_detection(image=test_image)
                    return {"credentials_valid": True}
                except Exception as e:
                    return {"credentials_valid": False, "error": str(e)}
            else:
                raise GoogleVisionError(f"Command '{cmd}' not supported", ErrorCodes.INVALID_SERVICE_MODE)
                
        except GoogleVisionError:
            raise
        except Exception as e:
            raise GoogleVisionError(f"Command execution failed", "COMMAND_FAILED", e)