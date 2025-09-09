import asyncio
from viam.module.module import Module
try:
    # Services
    from models.google_vision_service import GoogleVisionService

    # Components
    from models.google_vision_sensor import GoogleVisionSensor_OCRBasic
except ModuleNotFoundError:
    # when running as local module with run.sh
    # Services
    from .models.google_vision_service import GoogleVisionService

    # Components
    from .models.google_vision_sensor import GoogleVisionSensor_OCRBasic

if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
