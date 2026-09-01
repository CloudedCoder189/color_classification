import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
VALID_COLORS = {
    "Black",
    "Blue",
    "Brown",
    "Green",
    "Orange",
    "Red",
    "Violet",
    "White",
    "Yellow",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def average_rgb(image: np.ndarray) -> np.ndarray:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.mean(rgb_image, axis=(0, 1))


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    if not DATASET_DIR.is_dir():
        raise RuntimeError(f"Dataset directory not found: {DATASET_DIR}")

    features: list[np.ndarray] = []
    labels: list[str] = []

    for color_dir in sorted(DATASET_DIR.iterdir()):
        if not color_dir.is_dir():
            continue

        label = color_dir.name.strip().capitalize()
        if label not in VALID_COLORS:
            continue

        for image_path in sorted(color_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            features.append(average_rgb(image) / 255.0)
            labels.append(label)

    if not features:
        raise RuntimeError("No valid training images were found in the dataset.")

    return np.asarray(features), np.asarray(labels)


def train_model() -> tuple[LogisticRegression, StandardScaler, LabelEncoder]:
    features, labels = load_training_data()

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    classifier = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(scaled_features, encoded_labels)

    print(f"Loaded {len(features)} training images across {len(encoder.classes_)} classes.")
    return classifier, scaler, encoder


model, scaler, encoder = train_model()


def predict_color(image: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    rgb = average_rgb(image)
    scaled_features = scaler.transform((rgb / 255.0).reshape(1, -1))
    encoded_prediction = model.predict(scaled_features)
    predicted_color = encoder.inverse_transform(encoded_prediction)[0]
    preview_rgb = tuple(int(round(channel)) for channel in rgb)
    return predicted_color, preview_rgb


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return render_template("index.html", error="Please choose an image to upload."), 400

    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return render_template("index.html", error="The uploaded file is not a valid image."), 400

    predicted_color, preview_rgb = predict_color(image)
    preview_css = f"rgb({preview_rgb[0]}, {preview_rgb[1]}, {preview_rgb[2]})"

    return render_template(
        "index.html",
        predicted_color=predicted_color,
        preview_color=preview_css,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="Image is too large. Maximum upload size is 8 MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
