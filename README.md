# Satellite Image Classifier

A small convolutional neural network (a compact ResNet) that classifies satellite images into 10 land-use / land-cover categories, plus a web app where you can upload an image and see the prediction.

Built as a university project for **Programming for AI II** at Johannes Kepler University Linz (JKU).

![Demo screenshot](docs/screenshot.png)
<!-- TODO: add a screenshot of the web app at docs/screenshot.png -->

## Results

| Metric | Value |
|---|---|
| Validation accuracy (best epoch) | **96.4 %** |
| Validation accuracy (final epoch, 40) | 96.3 % |
| Training accuracy (final epoch) | 96.3 % |
| Input size | 64 × 64 RGB |
| Parameters | ~1.2 M |
| Training time | 40 epochs on CPU |

The validation set is a stratified 15 % hold-out of the 10,000 labelled images. The training and validation curves stay close together, so there is no strong sign of overfitting. The model was also used to predict a separate unlabelled test set of 1,980 images for the course challenge server; those test labels are not public, so no test score is reported here.

## Classes

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

## Project structure

```
.
├── k62405902.ipynb          # Full pipeline: data prep, EDA, training, evaluation, test predictions
├── app/
│   ├── app.py               # Shiny for Python web app (inference only)
│   └── assets/weights/
│       └── best_model.pth   # Trained model weights (~4.9 MB)
├── requirements.txt
├── LICENSE
└── README.md
```

The dataset itself is **not** included in the repository.

## Quick start: run the web app

```bash
git clone https://github.com/vrhvnsk/sattelite-image-classifier-.git
cd sattelite-image-classifier-

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

shiny run --reload app/app.py
```

Then open the address printed in the terminal (usually http://127.0.0.1:8000), upload a satellite image (JPG or PNG) and the app shows:

- the uploaded image,
- the predicted class with its confidence,
- a table with the probability of every class.

Images are resized to 64 × 64 before prediction. Works best with images similar to the training data (small top-down land-use patches); a full-size photo of an entire city will not give meaningful results.

## Reproducing the training

1. Obtain the dataset: 10 class folders with 10,000 labelled 64 × 64 images in total (the class names match the EuroSAT land-use categories). Arrange it like this:

   ```
   data sets/
   └── data/
       ├── AnnualCrop/    *.jpg
       ├── Forest/        *.jpg
       └── ...            (one folder per class)
   ```

2. Install the extra notebook dependencies (already in `requirements.txt`) and start Jupyter:

   ```bash
   jupyter notebook k62405902.ipynb
   ```

3. Run the notebook top to bottom. The best weights (by validation accuracy) are saved to `assets/weights/best_model.pth`; copy that file to `app/assets/weights/` to use it in the app.

Random seeds are fixed (`42`) for the split and for PyTorch, so results should be close to, but not necessarily bit-identical with, the numbers above, especially across different hardware.

## How it works

### 1. Data handling
- `preprocess()` walks the dataset folders and builds a DataFrame with the folder, file name and a numeric label for every image, plus a class-name to label mapping.
- The data is split **85 % train / 15 % validation**, stratified by label so every class keeps the same proportion in both sets.

### 2. Exploratory data analysis
- Random sample grid, distribution of average R/G/B values, and a brightness boxplot per class.
- Per-channel mean and standard deviation of the training set are computed and reused for normalisation. The same values are hard-coded in `app/app.py`, so training and inference preprocessing match.

### 3. Model
A small ResNet-style network (`ResNetSmall`):

```
Input 3×64×64
 → BasicBlock(3 → 32)
 → BasicBlock(32 → 64,  stride 2)
 → BasicBlock(64 → 128, stride 2)
 → BasicBlock(128 → 256, stride 2)
 → Global average pooling
 → Linear(256 → 10)
```

Each `BasicBlock` has two 3×3 convolutions with batch normalisation and a skip connection (a 1×1 convolution on the shortcut when the shape changes).

### 4. Training setup

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 5e-4 |
| LR schedule | StepLR, halved every 5 epochs |
| Loss | Cross-entropy with label smoothing 0.1 |
| Batch size | 64 |
| Epochs | 40 (early stopping with patience 10, not triggered) |
| Augmentation | random horizontal flip, rotation ±15°, brightness/contrast jitter |
| Model selection | weights from the epoch with the best validation accuracy |

### 5. Evaluation
Training curves, a confusion matrix on the validation set, and a plot of misclassified examples, followed by predictions on the unlabelled test set written to `submission.csv`.

## Limitations

- Trained on small 64 × 64 patches of a single dataset; it will not generalise well to other image sources, resolutions or regions.
- Validation data was also used to pick the best epoch, so the reported validation accuracy is slightly optimistic.
- Only 10 classes: anything else (for example, a desert or a glacier) will still be forced into one of them.
- The model definition is duplicated in the notebook and in `app/app.py`. If you change one, change the other.

## Possible improvements

- Per-class precision/recall and a proper held-out test split.
- Transfer learning from a pretrained backbone.
- Shared `model.py` module instead of duplicated code.
- Deploy the app (for example on shinyapps.io or Hugging Face Spaces).

## Tech stack

Python · PyTorch · torchvision · pandas · NumPy · scikit-learn · Matplotlib · Shiny for Python

## Author

**Ivan Novikov**, BSc Artificial Intelligence, Johannes Kepler University Linz

## License

BSD 2-Clause, see [LICENSE](LICENSE).
