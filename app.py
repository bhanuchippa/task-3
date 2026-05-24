"""
app.py — Flask Web Application for the AI Image Classifier
===========================================================
Serves a web UI that lets users upload an image and receive
CIFAR-10 class predictions from the trained MobileNetV2 model.

Endpoints
---------
GET  /                     – Main upload page
POST /predict              – Classify an uploaded image (JSON response)
GET  /outputs/<filename>   – Serve generated training artefacts
GET  /api/training-history – Return the saved training history JSON
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import os
import uuid
import json
import re

import numpy as np
import tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
)
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# ──────────────────────────────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER      = os.path.join(app.root_path, "static", "uploads")
MODEL_PATH         = os.path.join(app.root_path, "model", "mobilenet_model.h5")
LABELS_PATH        = os.path.join(app.root_path, "model", "labels.txt")
OUTPUTS_DIR        = os.path.join(app.root_path, "outputs")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}
IMG_SIZE           = 96

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Return True if *filename* has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_labels() -> list[str]:
    """Read class labels from labels.txt (one label per line)."""
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as fh:
            labels = [line.strip() for line in fh if line.strip()]
        print(f"[INFO] Loaded {len(labels)} class labels.")
        return labels
    except FileNotFoundError:
        print(f"[WARNING] Labels file not found at {LABELS_PATH}")
        return []


def load_model_from_disk():
    """Load the trained Keras .h5 model.  Returns None on failure."""
    if not os.path.isfile(MODEL_PATH):
        print(
            f"[WARNING] Model file not found at {MODEL_PATH}. "
            "Run model/train_model.py first."
        )
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return None


def preprocess_image(image_path: str) -> np.ndarray:
    """Open an image, resize, and preprocess it for MobileNetV2.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Preprocessed array with shape ``(1, IMG_SIZE, IMG_SIZE, 3)``.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    img_array = np.array(img, dtype=np.float32)                        # (96, 96, 3) in [0, 255]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # → [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)                      # (1, 96, 96, 3)
    return img_array


def preprocess_image_imagenet(image_path: str) -> np.ndarray:
    """Open an image, resize, and preprocess it for standard ImageNet MobileNetV2 (224x224)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ──────────────────────────────────────────────────────────────────────
# Global model / label loading (runs once at startup)
# ──────────────────────────────────────────────────────────────────────
print("[INFO] Initialising AI Image Classifier…")
model  = load_model_from_disk()
labels = load_labels()

# Load pretrained ImageNet MobileNetV2 for high-accuracy fallback/mapping
print("[INFO] Loading pretrained ImageNet MobileNetV2 model...")
try:
    imagenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    print("[INFO] ImageNet MobileNetV2 loaded successfully.")
except Exception as e:
    print(f"[WARNING] Could not load ImageNet MobileNetV2 (likely offline): {e}")
    imagenet_model = None


# ──────────────────────────────────────────────────────────────────────
# ImageNet-to-CIFAR-10 Class Mapping Rules
# ──────────────────────────────────────────────────────────────────────
IMAGENET_TO_CIFAR10_MAP = {
    "Dog": [
        'dog', 'puppy', 'hound', 'terrier', 'retriever', 'spaniel', 'collie', 'shepherd', 
        'mastiff', 'husky', 'malamute', 'corgi', 'poodle', 'pug', 'dalmatian', 'boxer', 
        'bulldog', 'schnauzer', 'rottweiler', 'pinscher', 'chihuahua', 'beagle', 'setter', 
        'pointer', 'vizsla', 'weimaraner', 'pekingese', 'maltese', 'pomeranian', 'dhole', 
        'dingo', 'samoyed', 'basenji', 'saluki', 'borzoi', 'greyhound', 'whippet', 
        'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'bouvier', 
        'appenzeller', 'entlebucher', 'leonberg', 'great_dane', 'saint_bernard', 
        'brabancon', 'dandie_dinmont', 'kuvasz', 'pekinese', 'shihtzu', 'redbone', 
        'bluetick', 'coonhound', 'afghan', 'elkhound', 'ibizan', 'walker_hound', 
        'black-and-tan_coonhound', 'lhasa', 'tibetan_terrier', 'coton_de_tulear', 
        'toy_poodle', 'miniature_poodle', 'standard_poodle', 'mexican_hairless', 
        'cardigan', 'pembroke', 'hunting_dog',
        'bloodhound', 'foxhound', 'wolfhound', 'otterhound', 'deerhound', 'bullterrier', 'sheepdog'
    ],
    "Cat": [
        'cat', 'kitty', 'kitten', 'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 
        'egyptian_cat', 'angora', 'cougar', 'puma', 'caracal', 'lynx', 'leopard', 
        'jaguar', 'cheetah', 'lion', 'tiger', 'snow_leopard', 'panther', 'wildcat', 
        'bobcat', 'ocelot', 'margay', 'serval'
    ],
    "Bird": [
        'bird', 'chick', 'finch', 'brambling', 'goldfinch', 'house_finch', 'junco', 
        'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 
        'kite', 'bald_eagle', 'vulture', 'caracara', 'peregrine', 'falcon', 'screech_owl', 
        'great_grey_owl', 'owl', 'black_grouse', 'ptarmigan', 'ruffed_grouse', 
        'prairie_chicken', 'peacock', 'quail', 'partridge', 'macaw', 'cockatoo', 
        'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 
        'toucan', 'drake', 'merganser', 'goose', 'swan', 'flamingo', 'spoonbill', 
        'ibis', 'stork', 'bittern', 'crane', 'limpkin', 'gallinule', 'coot', 'bustard', 
        'sandpiper', 'dowitcher', 'oystercatcher', 'pelican', 'penguin', 'albatross', 
        'ostrich', 'cassowary', 'emu', 'cock', 'hen', 'chicken', 'duck', 'mallard', 
        'puffin', 'kingfisher', 'tailorbird', 'weaver'
    ],
    "Deer": [
        'deer', 'elk', 'moose', 'hartebeest', 'impala', 'gazelle', 'antelope', 'caribou', 'reindeer'
    ],
    "Frog": [
        'frog', 'toad', 'bullfrog', 'tree_frog', 'tailed_frog'
    ],
    "Horse": [
        'horse', 'colt', 'foal', 'stallion', 'mare', 'zebra', 'pony', 'donkey', 'mule', 'wild_ass', 'sorrel'
    ],
    "Airplane": [
        'airplane', 'aeroplane', 'airliner', 'space_shuttle', 'wing', 'biplane', 'warplane', 'stealth',
        'jet', 'fighter_jet', 'hang_glider', 'glider'
    ],
    "Automobile": [
        'car', 'automobile', 'sports_car', 'limousine', 'cab', 'taxicab', 'convertible', 
        'coupe', 'station_wagon', 'minivan', 'jeep', 'racer', 'race_car', 'model_t', 'beach_wagon',
        'ambulance', 'police_car', 'taxi'
    ],
    "Truck": [
        'truck', 'trailer', 'pickup', 'lorry', 'fire_engine', 'garbage_truck', 'tow_truck', 
        'moving_van', 'recreational_vehicle', 'rv', 'tractor', 'police_van'
    ],
    "Ship": [
        'ship', 'boat', 'liner', 'ocean_liner', 'cruise', 'submarine', 'trimaran', 
        'catamaran', 'yacht', 'lifeboat', 'canoe', 'kayak', 'gondola', 'schooner', 
        'pirate', 'container_ship', 'freighter', 'warship', 'barge', 'houseboat',
        'sailboat', 'sailing_vessel', 'rowboat', 'speedboat', 'fireboat', 'aircraft_carrier'
    ]
}

EXCLUSIONS = {
    "tiger_shark", "tiger_beetle", "sea_lion", "freight_car", "streetcar", 
    "cable_car", "horse_cart", "mail_bag", "paper_towel", "carbonara", 
    "carton", "carousel", "carpenter's_kit", "potter's_wheel", "pinwheel",
    "paddlewheel", "hen-of-the-woods", "n03126707"
}

def match_keyword(label: str, keyword: str) -> bool:
    label_norm = label.lower().replace('_', ' ').replace('-', ' ')
    kw_norm = keyword.lower().replace('_', ' ').replace('-', ' ')
    pattern = r'\b' + re.escape(kw_norm) + r'\b'
    return bool(re.search(pattern, label_norm))

def get_mapped_cifar10_class(synset: str, class_label: str) -> str:
    """Return the CIFAR-10 category name if it maps, else None."""
    if class_label in EXCLUSIONS or synset in EXCLUSIONS:
        return None
    for cifar10_class, keywords in IMAGENET_TO_CIFAR10_MAP.items():
        for kw in keywords:
            if match_keyword(class_label, kw):
                return cifar10_class
    return None


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main upload / prediction page."""
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Serve the model performance dashboard page."""
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an uploaded image, classify it, and return JSON results.

    Response (success)::

        {
            "success": true,
            "predictions": [
                {"class": "Dog",   "confidence": 92.4},
                {"class": "Cat",   "confidence":  5.1},
                {"class": "Horse", "confidence":  1.8}
            ],
            "image_url": "/static/uploads/<uuid>_filename.jpg"
        }

    Response (error)::

        {"success": false, "error": "description"}
    """
    try:

        # ── Guard: a file must be present in the request ────────────
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": (
                    f"File type not allowed. Accepted: "
                    f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
            }), 400

        # ── Save the uploaded file with a unique name ───────────────
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(save_path)

        # ── Preprocess & predict ────────────────────────────────────
        mapped_results = None
        if imagenet_model is not None:
            try:
                processed_imagenet = preprocess_image_imagenet(save_path)
                imagenet_preds = imagenet_model.predict(processed_imagenet, verbose=0)
                # Decode top 1000 predictions from ImageNet model
                decoded = decode_predictions(imagenet_preds, top=1000)[0]

                # Accumulate prediction probabilities for matched classes
                cifar10_probs = {lbl: 0.0 for lbl in labels}
                total_mapped = 0.0

                for synset, class_label, prob in decoded:
                    mapped_class = get_mapped_cifar10_class(synset, class_label)
                    if mapped_class:
                        cifar10_probs[mapped_class] += float(prob)
                        total_mapped += float(prob)

                # Extract the top 2 unmapped ImageNet predictions to label the "Other" category descriptively
                unmapped_labels = []
                for synset, desc, prob in decoded[:5]:  # Check top 5 raw predictions
                    mapped_class = get_mapped_cifar10_class(synset, desc)
                    if not mapped_class:
                        clean_desc = desc.replace("_", " ").title()
                        unmapped_labels.append(clean_desc)
                        if len(unmapped_labels) == 2:
                            break

                if unmapped_labels:
                    other_label = f"Other ({' / '.join(unmapped_labels)})"
                else:
                    other_label = "Other"

                # Include "Other" as a prediction class
                all_preds = cifar10_probs.copy()
                all_preds[other_label] = max(0.0, 1.0 - total_mapped)

                # Format top 3 predictions
                results = []
                sorted_classes = sorted(all_preds.items(), key=lambda item: item[1], reverse=True)
                top3_results = sorted_classes[:3]
                for name, score in top3_results:
                    results.append({
                        "class": name,
                        "confidence": round(score * 100, 2)
                    })
                mapped_results = results
                print(f"[INFO] Classification succeeded via ImageNet mapping: {results}")
            except Exception as mapping_err:
                print(f"[WARNING] ImageNet mapping prediction failed: {mapping_err}")

        # Fallback to local CIFAR-10 model if ImageNet mapping is offline/failed to load
        if mapped_results is None:
            if model is None:
                return jsonify({
                    "success": False,
                    "error": "Model prediction failed: local model is not trained and ImageNet model mapping is unavailable.",
                }), 503
            print("[INFO] Using local model for prediction.")
            processed = preprocess_image(save_path)
            predictions = model.predict(processed, verbose=0)[0]          # shape (10,)

            # Top-3 predictions (descending confidence)
            top3_indices = predictions.argsort()[::-1][:3]
            results = []
            for idx in top3_indices:
                results.append({
                    "class":      labels[idx] if idx < len(labels) else f"Class {idx}",
                    "confidence": round(float(predictions[idx]) * 100, 2),
                })
        else:
            results = mapped_results

        return jsonify({
            "success":     True,
            "predictions": results,
            "image_url":   f"/static/uploads/{unique_filename}",
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve files from the outputs directory (graphs, history, etc.)."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/training-history")
def training_history():
    """Return the saved training history JSON, if available."""
    history_file = os.path.join(OUTPUTS_DIR, "training_history.json")
    if os.path.isfile(history_file):
        with open(history_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["available"] = True
        return jsonify(data)
    return jsonify({"available": False})


# ──────────────────────────────────────────────────────────────────────
# Error handlers (all return JSON so the frontend can parse them)
# ──────────────────────────────────────────────────────────────────────
@app.errorhandler(413)
def file_too_large(error):
    """Triggered when an upload exceeds MAX_CONTENT_LENGTH."""
    return jsonify({
        "success": False,
        "error":   "File too large. Maximum upload size is 16 MB.",
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Triggered for any unrecognised route."""
    return jsonify({
        "success": False,
        "error":   "The requested resource was not found.",
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Catch-all for unhandled server exceptions."""
    return jsonify({
        "success": False,
        "error":   "An internal server error occurred.",
    }), 500


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUTS_DIR,   exist_ok=True)

    print("[INFO] Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
