# ⚡ Quick Start Guide

Get the AI Image Classifier running in under 5 minutes.

---

## Prerequisites

- **Python 3.8+** — [Download from python.org](https://www.python.org/downloads/)
- **pip** — Comes bundled with Python 3.4+

Verify your installation:

```bash
python --version    # Must be 3.8 or higher
pip --version
```

---

## 🚀 Automated Setup (Recommended)

The easiest way to start the project is using the automated runner, which handles virtual environments, dependencies, model checks, and Flask startup in one step.

### Windows

```powershell
# Run the batch script
run\run.bat

# Or run the Python launcher directly
python run/run.py
```

### macOS / Linux

```bash
# Run the shell script
bash run/run.sh

# Or run the Python launcher directly
python3 run/run.py
```

---

## 🛠️ Manual Setup Alternative

If you prefer to run the setup steps manually:

### Windows

```powershell
# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (first time only, takes 10-30 minutes)
python model/train_model.py

# Start the web app
python app.py
```

### macOS / Linux

```bash
# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Train the model (first time only, takes 10-30 minutes)
python3 model/train_model.py

# Start the web app
python3 app.py
```

---

## Open in Browser

Once the app is running, open your browser and go to:

```
http://localhost:5000
```

---

## How to Use

1. **Upload** — Click the upload zone or drag and drop an image into it.
2. **Classify** — Click the **"Classify Image"** button.
3. **Results** — View the **top-3 predictions** with confidence scores and animated bars.

Supported formats: **JPEG, PNG, BMP, GIF, WEBP**

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named '...'`

Reinstall all dependencies:

```bash
pip install -r requirements.txt
```

If using a virtual environment, make sure it is activated first:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

### ❌ `Model file not found` warning

The model must be trained before running the app. Run training first:

```bash
python model/train_model.py
```

Training will save the model to `model/mobilenet_model.h5`.

---

### ❌ TensorFlow installation issues

Upgrade pip and reinstall TensorFlow:

```bash
pip install --upgrade pip
pip install tensorflow
```

On **Apple Silicon (M1/M2/M3)** Macs, use:

```bash
pip install tensorflow-macos
```

---

### ❌ Port 5000 already in use

Kill the process occupying the port:

**Windows:**

```powershell
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**macOS / Linux:**

```bash
lsof -i :5000
kill -9 <PID>
```

Alternatively, change the port in `app.py` by modifying the `app.run()` call:

```python
app.run(debug=True, port=8080)
```

---

### ❌ Low memory / slow training

- Close other memory-intensive applications.
- Reduce `BATCH_SIZE` in `model/train_model.py` (try `32` instead of `64`).
- Training uses **CPU** if no GPU is available — it works, but takes longer.
- Expect **10–30 minutes on CPU**, **3–10 minutes on GPU**.

---

### ❌ Python version too old

```bash
python --version
```

If the version is below 3.8, download the latest Python from [python.org](https://www.python.org/downloads/).

---

## What's Next?

- Read the full [README.md](README.md) for detailed documentation.
- Check the `outputs/` folder for training accuracy/loss graphs and confusion matrix after training.
- Try classifying different images — the model recognizes: **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**.
