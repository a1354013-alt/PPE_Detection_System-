# PPE Detection System Pro

This project provides a Tkinter-based PPE detection application focused on:

- video / camera PPE inspection
- violation event logging and screenshots
- CSV / Excel / PDF export
- heatmap generation
- Demo Mode for portfolio walkthroughs when no PPE model is available

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

## Real Mode Model Contract

Real Mode supports three PPE model formats:

1. `presence-based`
- Classes include `person` plus PPE presence classes such as `helmet`, `hardhat`, `vest`, `mask`, `goggles`.
- The system uses the `person` box as the anchor and checks PPE coverage inside person regions.

2. `violation-class-based`
- Classes directly represent violations such as `no_helmet`, `no hardhat`, `bare_head`, `no_vest`, `no_mask`, `no_goggles`, or `no_ppe`.
- Real Mode can work without a `person` class in this mode.

3. `mixed`
- The model exposes both presence classes and direct violation classes.
- The system merges both sources and avoids double-counting the same missing PPE item for the same tracked subject.

Class normalization includes mappings such as:

- `hardhat`, `hard_hat`, `safety helmet` -> `helmet`
- `safety vest`, `reflective vest` -> `vest`
- `no_helmet`, `no-hardhat`, `no hardhat`, `bare_head` -> `missing_helmet`
- `no_vest`, `no-safety-vest` -> `missing_vest`
- `no_mask`, `no mask` -> `missing_mask`
- `no_goggles` -> `missing_goggles`

If the model is unsupported, Real Mode is blocked with a clear error:

```text
Real Mode requires either:
1. person + selected PPE classes, or
2. direct violation classes such as no_helmet / no_vest.

The current model does not match the supported PPE contract. Please load a PPE-specific model or use Demo Mode.
```

## Default `yolov8n.pt` Limitation

- The default path is `yolov8n.pt`.
- `yolov8n.pt` is not a PPE-specific model.
- It is only suitable as an installation / GUI / pipeline smoke test.
- Real PPE detection requires a PPE-trained model that matches the Real Mode contract above.

## Demo Mode

Demo Mode is a fixed showcase flow for UI and reporting.

- It is suitable for demonstrations in environments without a PPE model.
- It simulates detections, event flow, screenshots, exports, and heatmap generation.
- It does not represent real model accuracy.
- Demo Mode is not a fallback for Real Mode.
- If Real Mode is unsupported, the app blocks startup instead of silently switching modes.

## Generated Outputs

Each detection run now writes into its own run folder:

```text
outputs/
  run_YYYYMMDD_HHMMSS/
    screenshots/
    reports/
    violations.csv
    heatmap.png
```

Notes:

- `screenshots/` stores captured violation frames.
- `reports/` stores manual CSV / XLSX / PDF exports from the GUI.
- `violations.csv` is the automatic per-run violation log.
- `heatmap.png` is written when the run contains violation coordinates.
- Older runs are preserved. Starting a new run does not delete previous run folders.

Legacy `reports/` and `violations/` folders remain ignored for compatibility, but formal deliverables should come from `outputs/run_xxx/`.

## Tests And Delivery Verification

Run the full acceptance flow with:

```bash
python -m compileall -q .
python -m unittest discover -v
pytest -q
python scripts/verify_delivery.py
```

## CI

GitHub Actions runs the same acceptance commands:

- `python -m compileall -q .`
- `python -m unittest discover -v`
- `pytest -q`
- `python scripts/verify_delivery.py`

## Known Limitations

- This repository does not include a training dataset.
- This repository does not include PPE-specific trained weights.
- Real-world accuracy depends on the chosen model, camera angle, lighting, occlusion, and jobsite conditions.
