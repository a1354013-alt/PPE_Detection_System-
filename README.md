# PPE Detection System Pro

This project provides a Tkinter-based PPE detection demo application with:

- video / camera detection flow
- event list and screenshot capture
- CSV / Excel / PDF report export
- heatmap generation for violation hotspots
- tracking-based cooldown to reduce duplicate events
- Demo Mode for portfolio presentation when no PPE model is available

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The GUI

Default launch:

```bash
python main_gui.py
```

Launch in Demo Mode:

```bash
python main_gui.py --demo
```

Launch with a custom model:

```bash
python main_gui.py --model path/to/model.pt
```

Launch with both a custom model path and Demo Mode:

```bash
python main_gui.py --demo --model path/to/model.pt
```

## Model Notes

- The default model path is `yolov8n.pt`.
- `yolov8n.pt` is a COCO model, not a PPE-specific model.
- If the loaded model only supports `person`, the GUI will clearly warn:
  `目前模型不是 PPE 專用模型，僅能偵測 person；若要偵測 helmet / vest / mask / goggles，請載入自訂 PPE 模型或使用 Demo Mode。`
- To detect `helmet`, `vest`, `mask`, and `goggles`, load a PPE-specific custom model.

## Demo Mode

Demo Mode is for product demonstration and workflow verification only.

- It can be used when you do not have a PPE model available.
- It simulates event generation so you can demonstrate the event list, report export, and heatmap flow.
- Demo Mode does not represent real inference results.

## Generated Outputs

- violation screenshots are written under `violations/`
- exported reports are written under `reports/`
- heatmap output is written to `reports/violation_heatmap.jpg`
- violation CSV log is written to `violations/violations.csv`

These generated artifacts should stay out of version control. The repository `.gitignore` ignores:

- `reports/`
- `violations/`
- video files: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`
- model files: `*.pt`, `*.pth`, `*.onnx`, `*.engine`, `*.weights`

## Tests

Run unit tests:

```bash
python -m unittest discover -v
```

Run delivery verification:

```bash
python scripts/verify_delivery.py
```

## Delivery Check

The delivery script verifies:

- the project can be compiled with `python -m compileall -q .`
- unit tests pass with `python -m unittest discover -v`
- `.gitignore` contains the required delivery-safe rules
- README commands are current and executable
- required packages exist in `requirements.txt`
- generated artifacts, model weights, and video files are not included in the repo
