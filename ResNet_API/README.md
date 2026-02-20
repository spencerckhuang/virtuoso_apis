(README and `virtuoso_resnet_api.py` backbone generated using AI, and fine-tuned afterwards by a human)

# Flask Image Classification API

A simple Flask API that:

- Accepts an image
- Accepts a task number (1–3)
- Uses a task-specific ResNet-101 model
- Returns `"COMPLETE"` or `"INCOMPLETE"`

---

## Requirements

- Python 3.8+
- Model weight files:
  ```
  task_1_weights.pth
  task_2_weights.pth
  task_3_weights.pth
  ```

---

## Installation

Install dependencies:

```bash
pip install flask torch torchvision pillow
```

---

## Project Structure

```
project/
│
├── virtuoso_resnet_api.py
├── resnet_weights.zip (UNZIP THIS FIRST)
├── task_1_weights.pth (AFTER UNZIP)
├── task_2_weights.pth (AFTER UNZIP)
├── task_3_weights.pth (AFTER UNZIP)
└── README.md
```

---

## Run the Server

Please make sure to unzip `resnet_weights.zip` before running the API. Unzip contents into the `ResNet_API` directory.

```bash
python virtuoso_resnet_api.py
```

The API will start at:

```
http://localhost:5000
```

Models are loaded once at startup.

---

## API Endpoint

### `POST /predict`

### Form Data

| Key      | Type  | Required | Description                  |
|----------|-------|----------|------------------------------|
| image    | file  | Yes      | Image file (jpg/png/etc.)    |
| task_id  | int   | Yes      | Task id (1, 2, or 3)         |

---

## Example Request using Python
```python
import requests

url = "http://localhost:5000/predict"

image_path = "test.jpg"
task_number = 2

with open(image_path, "rb") as img:
    files = {
        "image": img
    }
    data = {
        "task": str(task_number)
    }

    response = requests.post(url, files=files, data=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
```
```bash
Status Code: 200
Response JSON: {'task': 2, 'result': 'COMPLETE'}
```


## Example Request Using `curl`

```bash
curl -X POST \
  -F "image=@test.jpg" \
  -F "task_id=2" \
  http://localhost:5000/predict
```

---

## Example Response

```json
{
  "task_id": 2,
  "result": "COMPLETE"
}
```

---

## Behavior

- `task_id` must be 1, 2, or 3  
- Each task loads a separate ResNet-101 model  
- Image is resized and normalized automatically  
- Output threshold is 0.5 (sigmoid)

---
