import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

diag = np.eye(1000)
decoded_all = decode_predictions(diag, top=1)

keywords = ['horse', 'colt', 'foal', 'stallion', 'mare', 'zebra', 'pony', 'donkey', 'mule', 'ass', 'sorrel']

print("Fuzzy matches for Horse keywords in ImageNet:")
for idx in range(1000):
    _, class_label, _ = decoded_all[idx][0]
    class_lower = class_label.lower()
    for kw in keywords:
        if kw in class_lower:
            print(f" - Index {idx}: {class_label}")
            break
