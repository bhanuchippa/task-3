import os
import json
from app import app

client = app.test_client()

uploads_dir = os.path.join(app.root_path, "uploads")
images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(images)} images in uploads folder.")

for img in images:
    img_path = os.path.join(uploads_dir, img)
    print(f"\nTesting image: {img} (Size: {os.path.getsize(img_path)} bytes)")
    
    with open(img_path, 'rb') as f:
        response = client.post('/predict', data={'file': (f, img)})
        
    print(f"Status Code: {response.status_code}")
    try:
        data = response.get_json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print("Failed to get JSON response:", e)
        print("Response text:", response.data)
