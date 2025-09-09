import asyncio
from viam.module.module import Module
try:
    from models.google_vision_service import GoogleVisionService
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.google_vision_service import GoogleVisionService


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
