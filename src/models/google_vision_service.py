import os, sys
from typing import ClassVar, Tuple, List, Mapping, Optional, Sequence, Dict, Any

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

LOGGER = getLogger(__name__)


class GoogleVisionService(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("azeneli", "google-vision"), "google-vision-service"
    )
    
    def __init__(self, name: str):
        super().__init__(name)
        
        # Google vision client
        self.client = None

        self.camera = None

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
        """Validate camera component is available."""
        camera_name = attrs.get("camera_name")
        if not camera_name or not isinstance(camera_name, str):
            raise ValueError("'camera_name' is required and must be a string.")
        required_dependencies.append(camera_name)
        LOGGER.info(f"Validated camera dependency: {camera_name}")

    @classmethod 
    def _validate_credentials(cls, attrs: Dict[str, Any]):
        """Validate Google Cloud credentials are available."""
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

        return required_dependencies, optional_dependencies

    def _reconfigure_camera(self, attrs: Dict[str, Any], dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        """Get and store camera component during reconfigure."""
        camera_name = attrs.get("camera_name")
        
        if not camera_name:
            raise RuntimeError("Camera name is required but not provided")
        
        # Get the camera component from dependencies
        camera_resource_name = Camera.get_resource_name(camera_name)
        camera = dependencies.get(camera_resource_name)
        
        if not camera:
            raise RuntimeError(f"Camera '{camera_name}' not found in dependencies")
        
        # Store both name and camera reference
        self.camera_name = camera_name
        self.camera = camera
        self.logger.info(f"Camera configured: {camera_name}")
    
    def _reconfigure_credentials(self, attrs: Dict[str, Any]) -> None:
        """
        Reconfigures the Google Vision API client with credentials.
        Tries multiple credential methods with fallback.
        """
        credentials_path = attrs.get("credentials_path")
        
        # Method 1: Try service account file if provided
        if credentials_path:
            try:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
                
                self.client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
                self.logger.info(f"Using service account credentials file: {credentials_path}")
                self.credential_method = "service_account_file"
                return
                
            except Exception as e:
                self.logger.warning(f"Failed to load service account file: {e}")
                self.logger.info("Falling back to default credentials...")
        
        # Method 2: Try environment variable
        env_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if env_creds:
            try:
                self.client = vision.ImageAnnotatorClient()
                self.logger.info(f"Using GOOGLE_APPLICATION_CREDENTIALS: {env_creds}")
                self.credential_method = "environment_variable"
                return
                
            except Exception as e:
                self.logger.warning(f"Failed to use environment credentials: {e}")
                self.logger.info("Trying gcloud default credentials...")
        
        # Method 3: Try default credentials (gcloud auth)
        try:
            self.client = vision.ImageAnnotatorClient()
            self.logger.info("Using default credentials (gcloud CLI)")
            self.credential_method = "default_credentials"
            return
            
        except Exception as e:
            self.logger.error(f"All credential methods failed: {e}")
            raise RuntimeError(
                "Could not initialize Google Vision client. Please ensure you have:\n"
                "1. Valid service account JSON file, or\n"
                "2. GOOGLE_APPLICATION_CREDENTIALS environment variable, or\n" 
                "3. Run 'gcloud auth application-default login'"
            )
            
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
            
            
            
        return super().reconfigure(config, dependencies)

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
        self.logger.error("`capture_all_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        self.logger.error("`get_detections_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        self.logger.error("`get_detections` is not implemented")
        raise NotImplementedError()

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications` is not implemented")
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:
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
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()

