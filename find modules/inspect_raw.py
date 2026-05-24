import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

model = tf.keras.applications.MobileNetV2(weights='imagenet')

uploads_dir = r"c:\Users\penda\OneDrive\Documents\My Stuff\ML Task 3\AI_Image_Classifier\uploads"
img_path = os.path.join(uploads_dir, "cat.jpg")

img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224), Image.LANCZOS)
img_array = np.array(img, dtype=np.float32)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array, verbose=0)
decoded = decode_predictions(preds, top=20)[0]

print("Top 20 raw predictions for cat image:")
for i, (id_val, label, prob) in enumerate(decoded):
    print(f"{i+1}. {label}: {prob * 100:.2f}%")
