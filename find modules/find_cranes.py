from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import numpy as np

diag = np.eye(1000)
decoded_all = decode_predictions(diag, top=1)

for idx in range(1000):
    synset, class_label, _ = decoded_all[idx][0]
    if class_label == 'crane':
        print(f"Index {idx}: synset={synset}, label={class_label}")
