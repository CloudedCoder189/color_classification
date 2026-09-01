# Color Classification Web App

A Flask-based machine-learning web app that classifies the dominant color of an uploaded image.

## Overview

The application extracts the average RGB value from each training image, standardizes those features, and trains a multiclass logistic-regression model to predict one of nine color categories.

Supported classes:

- Black
- Blue
- Brown
- Green
- Orange
- Red
- Violet
- White
- Yellow

Users can upload an image through the web interface and receive the predicted class along with a swatch showing the image's average RGB color.

## How it works

1. Training images are loaded from category folders in `dataset/`.
2. Each image is converted from BGR to RGB.
3. Its average RGB value is calculated and normalized to `[0, 1]`.
4. Features are standardized with `StandardScaler`.
5. A balanced multiclass `LogisticRegression` model is trained once when the application starts.
6. Uploaded images go through the same preprocessing pipeline before prediction.

## Tech stack

- Python
- Flask
- OpenCV
- NumPy
- scikit-learn
- Gunicorn
- HTML/CSS/JavaScript

## Run locally

```bash
git clone https://github.com/CloudedCoder189/color_classification.git
cd color_classification
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

The development server uses port `5000` by default.

## Project structure

```text
.
├── app.py
├── dataset/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

## Deployment

The included `Procfile` starts the app with Gunicorn and binds to the platform-provided `PORT` environment variable.

## Current limitations

The model uses only average RGB values, so it describes the overall color of an image rather than understanding individual objects or regions. The current repository also trains the classifier during application startup rather than loading a separately versioned model artifact.

## Future improvements

- Separate training from web-server startup
- Save and version a trained model artifact
- Add a validation/test split and report classification metrics
- Compare average RGB features with richer color representations such as HSV histograms
- Add automated tests
