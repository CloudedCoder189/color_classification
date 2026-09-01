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

Users can upload an image through the web interface and receive the predicted color along with a visualization of the image's average RGB color.

## How It Works

1. Training images are loaded from category folders in `dataset/`.
2. Each image is converted from BGR to RGB.
3. The average RGB value is calculated and normalized.
4. Features are standardized with `StandardScaler`.
5. A balanced multiclass `LogisticRegression` model is trained.
6. Uploaded images go through the same preprocessing pipeline before prediction.

## Tech Stack

- Python
- Flask
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- HTML/CSS

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```text
SECRET_KEY=replace-with-a-random-secret
```

Then run:

```bash
python app.py
```

The app will start on port `5000` by default.

## Project Structure

```text
color_classification/
├── app.py
├── dataset/
├── templates/
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

## Security

Secrets are loaded through environment variables. Do not commit `.env` files or production credentials to the repository.

## Future Improvements

- Separate model training from application startup
- Save and load a trained model instead of retraining on each launch
- Add a validation/test split and report model accuracy
- Improve classification beyond average RGB features
- Add automated tests and clearer deployment configuration
