import os
import cv2
import io
import zipfile
import base64
import requests
import warnings
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
app.secret_key = 'BASICALLYICOOCKIE'

# ------------------------------
# Download & Extract Dataset if Needed
# ------------------------------
DATASET_URL = "https://github.com/CloudedCoder189/color_classification/archive/refs/heads/main.zip"
DATASET_ZIP = "dataset.zip"
DATASET_DIR = "dataset"


def download_and_extract_dataset():
    # If the folder already exists, skip re-downloading
    if os.path.isdir(DATASET_DIR):
        print("\U0001F4C2 Dataset already exists; skipping download.")
        return

    print("\U0001F4E5 Downloading dataset from GitHub...")
    response = requests.get(DATASET_URL)
    if response.status_code == 200:
        with open(DATASET_ZIP, "wb") as f:
            f.write(response.content)
        print("\U0001F4E6 Extracting dataset zip...")
        try:
            with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
                zip_ref.extractall()
            os.rename("color_classification-main/dataset", DATASET_DIR)
            print("\u2705 Dataset extracted!")
        except zipfile.BadZipFile:
            print("\u274C Error: The downloaded file is not a valid zip.")
            raise
        finally:
            os.remove(DATASET_ZIP)
    else:
        print("\u274C Failed to download dataset. Status code:", response.status_code)
        raise Exception("Dataset download failed")


# ------------------------------
# Load/Train Model
# ------------------------------
download_and_extract_dataset()
dataset_path = DATASET_DIR

valid_colors = {"Black", "Blue", "Brown", "Green", "Violet", "White", "Orange", "Red", "Yellow"}
X, y = [], []

try:
    for color_folder in os.listdir(dataset_path):
        folder_name = color_folder.lower().capitalize()
        if folder_name not in valid_colors:
            continue

        folder_path = os.path.join(dataset_path, color_folder)
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            avg_color = np.mean(img, axis=(0, 1)) / 255.0

            X.append(avg_color)
            y.append(folder_name)

    X = np.array(X)
    y = np.array(y)
    print(f"\u2705 Total images processed: {len(X)}")

    if len(X) == 0:
        raise ValueError("❌ No images loaded! Check dataset.")
except Exception as e:
    print(f"Error processing dataset: {e}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced")
model.fit(X, y_encoded)
print("✅ Model trained on 100% of the dataset!")


# ------------------------------
# Prediction Function
# ------------------------------
def predict_color_from_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(img, axis=(0, 1)).reshape(1, -1) / 255.0
    avg_color_scaled = scaler.transform(avg_color)
    predicted_label = model.predict(avg_color_scaled)
    predicted_color = encoder.inverse_transform(predicted_label)[0]
    return predicted_color, avg_color


# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            flash("Invalid image file. Please upload a valid image.")
            return redirect(request.url)

        predicted_color, avg_color = predict_color_from_image(image)

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow([[avg_color[0]]])
        ax.axis("off")

        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        pngImageB64String = (
            "data:image/png;base64,"
            + base64.b64encode(pngImage.getvalue()).decode('utf8')
        )

        return render_template(
            'index.html',
            predicted_color=predicted_color,
            plot_url=pngImageB64String
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
