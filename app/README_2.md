# Web app

Shiny for Python app for classifying satellite images with the trained model.

## Run

From the **repository root**:

```bash
pip install -r requirements.txt
shiny run --reload app/app.py
```

Open http://127.0.0.1:8000 in your browser.

## Files

| Path | Purpose |
|---|---|
| `app.py` | UI and inference code. Contains its own copy of the model class. |
| `assets/weights/best_model.pth` | Trained weights (PyTorch `state_dict`). |

## Notes

- The weights file only loads into the `ResNetSmall` architecture defined in `app.py`. If you retrain with a different architecture, update the class here as well.
- `CLASS_NAMES` and the normalisation `mean`/`std` must match what the model was trained with (both come from the notebook).
- The app runs on GPU if CUDA is available, otherwise on CPU. CPU is fast enough for single images.
- Uploaded images are resized to 64 × 64 before prediction.
