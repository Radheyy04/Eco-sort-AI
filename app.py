"""EcoSort AI - Flask routes only. ML logic lives in model/predictor.py."""
from flask import Flask, render_template, request, jsonify
from model import predictor
from model.class_names import CLASS_NAMES, BIODEGRADABLE_CLASSES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024       # 8 MB upload limit
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

RECOMMENDATIONS = {
    "Biodegradable": "Place biodegradable waste in the appropriate organic-waste/composting stream according to local waste-management guidelines.",
    "Non-Biodegradable": "Place the item in the appropriate dry/recyclable or residual-waste stream according to local waste-management guidelines.",
}

# Load the trained model ONCE at startup. The app still runs without it
# so the page can be previewed; /predict then returns a clear error.
model_error = None
try:
    predictor.load_model()
except Exception as e:
    model_error = str(e)
    print(f"[EcoSort AI] Model not loaded: {model_error}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")          # Page 1: Introduction


@app.route("/classify")
def classify():
    return render_template("classify.html")       # Page 2: Main classifier


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")   # Page 3: How it works


@app.route("/classes")
def classes():
    """Six dataset classes + category, so the frontend never hard-codes them."""
    return jsonify([
        {"name": c, "category": "Biodegradable" if c in BIODEGRADABLE_CLASSES else "Non-Biodegradable"}
        for c in CLASS_NAMES
    ])


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image selected. Please choose an image first."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Please upload a JPG, JPEG or PNG image."), 400
    if model_error:
        return jsonify(error=f"The model is not available: {model_error}"), 503

    try:
        # Image is read in memory only - nothing is saved to disk.
        result = predictor.predict(file.read())
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify(error="Could not process this image. Please try a different JPG or PNG file."), 422

    result["recommendation"] = RECOMMENDATIONS[result["category"]]
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large. Maximum size is 8 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
