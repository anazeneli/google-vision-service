# Google Vision Service Module
A Viam `vision` service for Google Cloud Vision API providing OCR, object detection, and image classification capabilities.

## Model `azeneli:google-vision:google-vision-service`
This model implements the `rdk:service:vision` API with Google Cloud Vision integration. It supports multiple service modes optimized for different computer vision tasks and provides OCR capabilities through do_command interface.

### Features

- **OCR (Optical Character Recognition)**: Extract text from images with basic or document analysis modes
- **Object Detection**: Detect and locate objects using Google Vision API via standard Viam interface
- **Image Classification**: Label and classify image content via standard Viam interface
- **Multiple Service Modes**: Optimized configurations for specific use cases
- **Flexible Authentication**: Supports service accounts, environment variables, or Google Cloud CLI
- **Multi-language OCR**: Support for multiple languages with auto-detection
- **Confidence Filtering**: Configurable confidence thresholds for all operations
- **Sensor Integration**: OCR results return structured dictionaries perfect for sensor consumption

### Configuration
The following attribute template can be used to configure this model:

```json
{
  "camera_name": "<string>",
  "credentials_path": "<string>",
  "service_mode": "<string>",
  "ocr_mode": "<string>",
  "ocr_languages": ["<language_code>"],
  "min_confidence": <float>,
  "max_results": <integer>
}
```

Note: All attributes except credentials_path are optional (if using Google Cloud CLI authentication).

### Attributes

| Name | Type | Inclusion | Description |
|------|------|-----------|-------------|
| `camera_name` | string | Optional | Name of the camera component (adds camera dependency) |
| `credentials_path` | string | Optional | Path to Google Cloud service account JSON file |
| `service_mode` | string | Optional | Service optimization mode: "general", "ocr", "detection", "classification" (defaults to "general") |
| `ocr_mode` | string | Optional | OCR detection mode: "basic" or "document" (defaults to "basic") |
| `ocr_languages` | array | Optional | Language codes for OCR (empty array = auto-detect) |
| `min_confidence` | float | Optional | Minimum confidence threshold 0.0-1.0 (defaults vary by service mode) |
| `max_results` | integer | Optional | Maximum results per API call (defaults vary by service mode) |

### Service Modes

**General Mode (default)**
- All standard Vision API methods available
- Balanced settings for mixed usage
- OCR available via do_command

**OCR Mode**  
- Optimized for text extraction
- Standard Vision methods return empty with warnings
- OCR via do_command with document-optimized defaults
- Higher confidence threshold (0.7), more results (100)

**Detection Mode**
- Only object detection via get_detections()
- Classification methods return empty with warnings
- Medium confidence threshold (0.6), moderate results (20)

**Classification Mode**
- Only image classification via get_classifications()
- Detection methods return empty with warnings  
- Lower confidence threshold (0.5) for more labels (15)

### Authentication Options

**Option 1: Google Cloud CLI (Recommended for development)**
```bash
gcloud auth application-default login
```
Configuration: Omit credentials_path attribute

**Option 2: Service Account (Recommended for production)**
```json
{
  "credentials_path": "/path/to/service-account.json"
}
```

**Option 3: Environment Variable**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```
Configuration: Omit credentials_path attribute

### Example Configurations

**Basic OCR Configuration**
```json
{
  "service_mode": "ocr",
  "ocr_mode": "document",
  "min_confidence": 0.8,
  "ocr_languages": ["en"]
}
```

**General Purpose with Camera**
```json
{
  "camera_name": "my_camera",
  "credentials_path": "/path/to/credentials.json",
  "service_mode": "general",
  "min_confidence": 0.6
}
```

**Object Detection Focused**
```json
{
  "service_mode": "detection",
  "min_confidence": 0.7,
  "max_results": 25
}
```

**Multi-language OCR**
```json
{
  "service_mode": "ocr",
  "ocr_mode": "document", 
  "ocr_languages": ["en", "es", "fr"],
  "min_confidence": 0.8
}
```

### Standard Vision Interface Usage

**Object Detection**
```python
from viam.services.vision import Vision

vision = Vision.from_robot(robot, "google_vision")
detections = await vision.get_detections(image)

for detection in detections:
    print(f"Object: {detection.class_name}")
    print(f"Confidence: {detection.confidence:.2f}")
    print(f"Box: ({detection.x_min}, {detection.y_min}) to ({detection.x_max}, {detection.y_max})")
```

**Image Classification**
```python
classifications = await vision.get_classifications(image, count=5)

for classification in classifications:
    print(f"Label: {classification.class_name} ({classification.confidence:.2f})")
```

### OCR via do_command

**Basic Text Extraction**
```python
import base64

# Load and encode image
with open("receipt.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Extract text
result = await vision.do_command({
    "command": "text_detection",
    "image": image_b64
})

print("Full text:", result['full_text'])
print("Text blocks:", result['block_count'])
print("Confidence:", result['confidence'])
```

**Document Analysis**
```python
result = await vision.do_command({
    "command": "document_text_detection",
    "image": image_b64
})

print(f"Pages: {result['pages']}")
print(f"Words: {result['words']}")
print(f"Text: {result['full_text']}")
```

**OCR Response Format**
```python
{
  "full_text": "SILKY DESSERTS PUYO EST 2015\nSilky Green Tea 1x 12,500",
  "blocks": [
    {
      "text": "SILKY",
      "confidence": 0.95,
      "vertices": [(100, 50), (150, 50), (150, 70), (100, 70)]
    }
  ],
  "confidence": 0.93,
  "block_count": 15,
  "detection_mode": "document",
  "languages_used": ["en"]
}
```

**Sensor Integration**
OCR results return structured dictionaries perfect for sensor consumption:
```python
class OCRSensor(Sensor):
    async def get_readings(self) -> Dict[str, Any]:
        ocr_result = await self.vision.do_command({
            "command": "text_detection",
            "image": image_b64
        })
        
        return {
            "extracted_text": ocr_result['full_text'],
            "confidence": ocr_result['confidence'],
            "block_count": ocr_result['block_count']
        }
```

### DoCommand

**get_mode_info**
Get current service configuration:
```python
info = await vision.do_command({"command": "get_mode_info"})
print(f"Service mode: {info['service_mode']}")
print(f"OCR mode: {info['ocr_mode']}")
print(f"Languages: {info['ocr_languages']}")
```

**text_detection**
Extract text from base64-encoded images. Returns structured dictionary with full text, individual blocks, confidence scores, and metadata.

**document_text_detection**
Structured document analysis for forms, receipts, and invoices. Returns page count, word count, and organized text structure.

### Output Examples

**Standard Vision Interface**
- Object Detection: Detection objects with class_name, confidence, and bounding box coordinates
- Image Classification: Classification objects with class_name and confidence scores

**OCR via do_command**
- Returns dictionaries with full_text, individual text blocks, confidence scores, and positioning data
- Perfect for sensor integration and structured text processing

### Error Handling

The service includes comprehensive error handling:
- **Authentication errors**: Clear messages about credential configuration
- **API quota errors**: Guidance on rate limits and quota management  
- **Service mode conflicts**: Warnings when methods aren't available in current mode
- **Invalid configurations**: Validation errors with specific requirements
- **Network issues**: Graceful handling of connectivity problems

### Available Commands

| Command | Description | Returns |
|---------|-------------|---------|
| `text_detection` | Extract text from image | Structured text data with blocks and confidence |
| `document_text_detection` | Structured document analysis | Page/word counts with organized text |
| `get_mode_info` | Get current service configuration | Service settings and capabilities |