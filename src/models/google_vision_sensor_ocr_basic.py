from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, cast
import time

from viam.components.sensor import Sensor
from viam.components.camera import Camera
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, Detection
from viam.utils import SensorReading, struct_to_dict
from viam.logging import getLogger

LOGGER = getLogger(__name__)

class GoogleVisionSensor_OCRBasic(Sensor, EasyResource):
    """
    Sensor for OCR receipts using GoogleVisionService in 'ocr_basic' mode.
    Minimal: validates config, binds camera + vision service, and returns
    detections from capture_all_from_camera().
    """

    MODEL: ClassVar[Model] = Model(
        ModelFamily("azeneli", "vision"), "google-vision-sensor-ocr-basic"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.camera: Optional[Camera] = None
        self.vision_service: Optional[Vision] = None
        self.camera_name: Optional[str] = None

    # ---------- Config ----------
    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        required: List[str] = []

        cam = attrs.get("camera_name")
        if not isinstance(cam, str) or not cam:
            raise ValueError("'camera_name' is required and must be a non-empty string")
        required.append(cam)

        vs = attrs.get("vision_service")
        if not isinstance(vs, str) or not vs:
            raise ValueError("'vision_service' is required and must be a non-empty string")
        required.append(vs)

        return required, []

    def reconfigure(
        self,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)

        # Camera dependency
        self.camera_name = attrs["camera_name"]
        cam_res = dependencies.get(Camera.get_resource_name(self.camera_name))
        if not isinstance(cam_res, Camera):
            raise ValueError(f"Dependency '{self.camera_name}' did not resolve to a Camera")
        self.camera = cam_res

        # Vision service dependency
        vs_name = attrs["vision_service"]
        vs_res = dependencies.get(Vision.get_resource_name(vs_name))
        if not isinstance(vs_res, Vision):
            raise ValueError(f"Dependency '{vs_name}' did not resolve to a Vision service")
        self.vision_service = vs_res

        LOGGER.info(f"{self.name}: configured with camera={self.camera_name}, vision={vs_name}")
        return super().reconfigure(config, dependencies)

    # ---------- Sensor API ----------
    async def get_readings(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, SensorReading]:
        """
        Calls the Vision service's capture_all_from_camera() to get detections.
        (Later you’ll transform these into vendor/items/subtotal/tax/total/etc.)
        """
        if not self.vision_service or not self.camera_name:
            return {"error": "sensor not configured"}

        # Ask the Vision service to capture and run OCR/detections on the camera
        result = await self.vision_service.capture_all_from_camera(  # type: ignore
            self.camera_name,
            return_image=False,
            return_classifications=False,
            return_detections=True,
            return_object_point_clouds=False,
            extra=extra,
            timeout=timeout,
        )

        detections: List[Detection] = getattr(result, "detections", []) or []

        return {
            "timestamp": time.time(),
            "detections_count": len(detections),
            "detections_raw": detections,
        }

    async def get_geometries(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> List[Geometry]:
        return []
