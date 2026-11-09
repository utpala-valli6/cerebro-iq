
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import os
import sqlite3
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# ---------------- MODEL ----------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "stroke_model_v4.keras"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "model",
    "threshold.txt"
)

model = tf.keras.models.load_model(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = float(file.read().strip())

print("Loaded model: stroke_model_v4.keras")
print("Loaded threshold:", THRESHOLD)

IMAGE_SIZE = (224, 224)

# ---------------- UPLOADS ----------------

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ---------------- DATABASE ----------------

DATABASE = os.path.join(BASE_DIR, "cerebroiq.db")


def init_database():
    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            result TEXT,
            confidence REAL,
            date TEXT
        )
    """)

    connection.commit()
    connection.close()


init_database()


# ---------------- IMAGE PROCESSING ----------------

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ---------------- HOME / DASHBOARD ----------------

@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------------- ANALYZE SCAN ----------------

@app.route("/analyze")
def analyze():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select a brain CT image."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:
        # Open uploaded image
        image = Image.open(file.stream)

        # Standardize image orientation and format
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        # Generate unique filename
        filename = str(uuid.uuid4()) + ".png"

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Save uploaded image
        image.save(filepath, format="PNG")

        # Preprocess image
        processed_image = preprocess_image(image)

        # Model prediction
        prediction = model(
        processed_image,
        training=False
    )

        probability = float(prediction.numpy()[0][0])

        print("Stroke probability:", round(probability, 4))
        print("Decision threshold:", THRESHOLD)

        # Classification
        if probability >= THRESHOLD:
            result = "STROKE DETECTED"
            result_class = "stroke"
            confidence = probability * 100

        else:
            result = "NO STROKE DETECTED"
            result_class = "normal"
            confidence = (1 - probability) * 100

        confidence = round(confidence, 2)

        # Save result to database
        connection = sqlite3.connect(DATABASE)

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO history
            (filename, result, confidence, date)
            VALUES (?, ?, ?, ?)
        """, (
            filename,
            result,
            confidence,
            datetime.now().strftime("%d-%m-%Y %H:%M")
        ))

        connection.commit()
        connection.close()

        return render_template(
            "index.html",
            result=result,
            result_class=result_class,
            confidence=confidence,
            image_path="uploads/" + filename
        )

    except (UnidentifiedImageError, OSError, ValueError):
        return render_template(
            "index.html",
            error="Invalid image. Please upload a supported image."
        )

    except Exception as e:
        print("Prediction error:", str(e))

        return render_template(
            "index.html",
            error="Unable to process the image. Please try again."
        )


# ---------------- PATIENT HISTORY ----------------

@app.route("/history")
def history():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, filename, result, confidence, date
        FROM history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    connection.close()

    return render_template(
        "history.html",
        records=records
    )


# ---------------- REPORTS ----------------

@app.route("/reports")
def reports():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT filename, result, confidence, date
        FROM history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    connection.close()

    return render_template(
        "reports.html",
        records=records
    )



# ---------------- SETTINGS ----------------

@app.route("/settings")
def settings():

    return render_template(
        "settings.html",
        threshold=THRESHOLD
    )


# ---------------- HELP ----------------

@app.route("/help")
def help_page():
    return render_template("help.html")


# ---------------- FILE SIZE ERROR ----------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error="Image is too large. Please upload an image under 10 MB."
    ), 413


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app.run(debug=True)
