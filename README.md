# Google Vision Service Module
A Viam `vision` service for Google Cloud Vision API providing OCR (Optical Character Recognition) capabilities.

## Model `azeneli:google-vision:google-vision-service`
This model implements the `rdk:service:vision` API with Google Cloud Vision integration for text extraction from images.

### Features
- **OCR Basic Mode**: Extract text from images with word-level bounding boxes
- **Multi-language Support**: Auto-detection or specify language codes
- **Flexible Authentication**: Service accounts, environment variables, or Google Cloud CLI
- **Standard Vision API**: Uses Viam's standard `get_detections()` interface

### Configuration
```json
{
  "camera_name": "<string>",
  "credentials_path": "<string>", 
  "service_mode": "ocr_basic",
  "ocr_languages": ["<language_code>"],
  "max_results": <integer>
}
```

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `camera_name` | string | Name of the camera component (required) |
| `credentials_path` | string | Path to Google Cloud service account JSON file (optional) |
| `service_mode` | string | Must be "ocr_basic" |
| `ocr_languages` | array | Language codes for OCR (optional, empty = auto-detect) |
| `max_results` | integer | Maximum text blocks to return (optional, 1-1000) |

### Authentication Options

**Option 1: Google Cloud CLI**
```bash
gcloud auth application-default login
```

**Option 2: Service Account**
```json
{
  "credentials_path": "/path/to/service-account.json"
}
```

**Option 3: Environment Variable**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### Example Configurations

**Basic OCR with Camera**
```json
{
  "camera_name": "my_camera",
  "service_mode": "ocr_basic"
}
```

**Multi-language OCR**
```json
{
  "camera_name": "my_camera",
  "service_mode": "ocr_basic",
  "ocr_languages": ["en", "es"],
  "max_results": 50
}
```

### Usage

**Text Detection from Camera**
```python
from viam.services.vision import Vision

vision = Vision.from_robot(robot, "google_vision")
detections = await vision.get_detections_from_camera("my_camera")

for detection in detections:
    print(f"Text: {detection.class_name}")
    print(f"Confidence: {detection.confidence:.2f}")
    print(f"Box: ({detection.x_min}, {detection.y_min}) to ({detection.x_max}, {detection.y_max})")
```

**Text Detection from Image**
```python
image = await camera.get_image()
detections = await vision.get_detections(image)

for detection in detections:
    print(f"Found text: '{detection.class_name}'")
```

**Capture All (Image + Text)**
```python
result = await vision.capture_all_from_camera(
    "my_camera", 
    return_image=True, 
    return_detections=True
)

print(f"Found {len(result.detections)} text blocks")
if result.image:
    print("Image captured successfully")
```

### Output Format
Returns `Detection` objects where:
- `class_name`: The detected text content
- `confidence`: Detection confidence score (typically 0.5)
- `x_min, y_min, x_max, y_max`: Text bounding box coordinates

### DoCommand

**Health Check**
```python
status = await vision.do_command({"command": "health_check"})
print(f"Status: {status['status']}")
print(f"Camera configured: {status['camera_configured']}")
```

**Get Configuration**
```python
config = await vision.do_command({"command": "get_config"})
print(f"Service mode: {config['service_mode']}")
print(f"Camera: {config['camera_name']}")
```

### Error Handling
- **Authentication errors**: Check credentials configuration and Google Cloud permissions
- **Camera errors**: Verify camera_name matches configured camera component
- **API errors**: Check Google Cloud Vision API quota and billing
- **Network issues**: Ensure internet connectivity for API calls