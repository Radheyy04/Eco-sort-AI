# EcoSort AI – AI-Powered Waste Classification

Flask + HTML/CSS/JS app. Image → one of 6 dataset classes → Biodegradable / Non-Biodegradable.
It does not determine material composition, exact recyclability or legal disposal rules.

## Structure
```
ecosort-ai/
├── app.py                 Flask routes
├── model/
│   ├── class_names.py     ← EDIT: classes, mapping, model settings
│   ├── predictor.py       model loading + inference
│   └── model.pth          ← PUT YOUR MODEL HERE
├── templates/index.html
├── static/style.css, script.js
├── uploads/               (unused; images are processed in memory only)
└── requirements.txt
```

## Setup
1. Install Python 3.9+ and open a terminal in this folder.
2. Create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
   (TensorFlow users: edit requirements.txt first, see comments inside.)
4. Copy your trained model into `model/` (default name `model.pth`).
5. Open `model/class_names.py` and set:
   - `CLASS_NAMES` – your six classes **in training-label order** (wrong order = wrong labels)
   - `BIODEGRADABLE_CLASSES` / `NON_BIODEGRADABLE_CLASSES`
   - `MODEL_CONFIG` – framework, file path, image size, normalisation mean/std,
     and (PyTorch state_dict files) the architecture, e.g. `resnet18`.
6. Run: `python app.py` and open http://127.0.0.1:5000

## Matching your training setup
- **PyTorch full model** (`torch.save(model)`): loads directly.
- **PyTorch state_dict** (`torch.save(model.state_dict())`): set `architecture` to the torchvision
  model you trained. For a custom architecture, build it in `load_model()` in `predictor.py`.
- **TensorFlow/Keras**: set `framework` to `"tensorflow"` and path to `.h5`/`.keras`.
- Image size, mean and std must match training, otherwise confidence will be poor.

## Troubleshooting
- *"Model file not found"* – check the file is in `model/` and matches `MODEL_CONFIG["path"]`.
- *State-dict size mismatch* – `architecture` doesn't match the trained network.
- The page loads even without a model; classification then shows a clear error message.
