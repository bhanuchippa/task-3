import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Define the mapping rules from app.py
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
        'cardigan', 'pembroke'
    ],
    "Cat": [
        'cat', 'kitty', 'kitten', 'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 
        'egyptian_cat', 'angora', 'cougar', 'puma', 'caracal', 'lynx', 'leopard', 
        'jaguar', 'cheetah', 'lion', 'tiger', 'snow_leopard'
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
        'ostrich', 'cassowary', 'emu', 'cock', 'hen', 'chicken'
    ],
    "Deer": [
        'deer', 'elk', 'moose', 'hartebeest', 'impala', 'gazelle', 'antelope', 'caribou', 'reindeer'
    ],
    "Frog": [
        'frog', 'toad', 'bullfrog', 'tree_frog', 'tailed_frog'
    ],
    "Horse": [
        'horse', 'colt', 'foal', 'stallion', 'mare', 'zebra', 'pony'
    ],
    "Airplane": [
        'airplane', 'aeroplane', 'airliner', 'space_shuttle', 'wing', 'biplane', 'warplane', 'stealth'
    ],
    "Automobile": [
        'car', 'automobile', 'sports_car', 'limousine', 'cab', 'taxicab', 'convertible', 
        'coupe', 'station_wagon', 'minivan', 'jeep', 'racer', 'race_car', 'model_t', 'beach_wagon'
    ],
    "Truck": [
        'truck', 'trailer', 'pickup', 'lorry', 'fire_engine', 'garbage_truck', 'tow_truck', 
        'moving_van', 'recreational_vehicle', 'rv'
    ],
    "Ship": [
        'ship', 'boat', 'liner', 'ocean_liner', 'cruise', 'submarine', 'trimaran', 
        'catamaran', 'yacht', 'lifeboat', 'canoe', 'kayak', 'gondola', 'schooner', 
        'pirate', 'container_ship', 'freighter', 'carrier', 'warship', 'barge', 'houseboat'
    ]
}

model = tf.keras.applications.MobileNetV2(weights='imagenet')

uploads_dir = r"c:\Users\penda\OneDrive\Documents\My Stuff\ML Task 3\AI_Image_Classifier\static\uploads"
img_path = os.path.join(uploads_dir, "3060f1daa9bf42bca475add4070436e0.jpg")

img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224), Image.LANCZOS)
img_array = np.array(img, dtype=np.float32)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array, verbose=0)
decoded = decode_predictions(preds, top=100)[0]

print("Predictions that map to nothing (contributing to 'Other'):")
unmapped_sum = 0.0
for i, (id_val, label, prob) in enumerate(decoded):
    label_lower = label.lower()
    matched = False
    for cifar10_class, keywords in IMAGENET_TO_CIFAR10_MAP.items():
        if any(kw in label_lower for kw in keywords):
            matched = True
            break
    if not matched:
        print(f" - {label}: {prob * 100:.4f}%")
        unmapped_sum += float(prob)

print(f"\nTotal unmapped probability in top 100: {unmapped_sum * 100:.2f}%")
