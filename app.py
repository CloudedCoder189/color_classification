import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template, redirect, url_for, flash
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
app.secret_key = 'BASICALLYICOOCKIE'  # Replace with your secret key

# ------------------------------
# Load/Train your Model
# ------------------------------

# 📂 Dataset Path (adjust the path if needed)
dataset_path = r"C:\Users\nirmi\Downloads\sdf\ColorClassification"

# 🚀 Valid Color Labels
valid_colors = {"Black", "Blue", "Brown", "Green", "Violet", "White", "Orange", "Red", "Yellow"}

X, y = [], []

# 🏷️ Process Images
for color_folder in os.listdir(dataset_path):
    # Capitalize the folder name to match valid_colors
    folder_name = color_folder.lower().capitalize()
    if folder_name not in valid_colors:
        continue  # Ignore non-color folders

    folder_path = os.path.join(dataset_path, color_folder)
    print(f"Processing {folder_name}...")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(file_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_color = np.mean(img, axis=(0, 1)) / 255.0  # Normalize

        X.append(avg_color)
        y.append(folder_name)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print(f"✅ Total images processed: {len(X)}")

if len(X) == 0:
    raise ValueError("❌ No images loaded! Check dataset.")

# 🔢 Encode Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ⚖ Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🎯 Train Logistic Regression on 100% of the data
model = LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced")
model.fit(X, y_encoded)

print("✅ Model trained on 100% of the dataset!")


# ------------------------------
# Prediction Function
# ------------------------------
def predict_color_from_image(image):
    # Convert image to RGB if needed
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
        print("Received POST request.")
        # Debug: list keys in request.files
        print("Request files keys:", list(request.files.keys()))
        if 'image' not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        file = request.files['image']
        print("Filename:", file.filename)
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        # Read the image file from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            flash("Invalid image file. Please upload a valid image.")
            return redirect(request.url)

        # Get prediction and average color
        predicted_color, avg_color = predict_color_from_image(image)

        # Create a plot for the average color
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow([[avg_color[0]]])
        ax.axis("off")

        # Save plot to PNG in memory
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        pngImageB64String = "data:image/png;base64," + base64.b64encode(pngImage.getvalue()).decode('utf8')

        return render_template('index.html', predicted_color=predicted_color, plot_url=pngImageB64String)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
