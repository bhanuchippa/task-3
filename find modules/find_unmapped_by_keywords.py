import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

diag = np.eye(1000)
decoded_all = decode_predictions(diag, top=1)

# List of all unmapped labels from the previous run
IMAGENET_TO_CIFAR10_MAP_KEYWORDS = {
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
        'cardigan', 'pembroke', 'hunting_dog'
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
        'moving_van', 'recreational_vehicle', 'rv', 'tractor'
    ],
    "Ship": [
        'ship', 'boat', 'liner', 'ocean_liner', 'cruise', 'submarine', 'trimaran', 
        'catamaran', 'yacht', 'lifeboat', 'canoe', 'kayak', 'gondola', 'schooner', 
        'pirate', 'container_ship', 'freighter', 'carrier', 'warship', 'barge', 'houseboat',
        'sailboat', 'sailing_vessel', 'rowboat', 'speedboat'
    ]
}

import re
def match_keyword(label: str, keyword: str) -> bool:
    label_norm = label.lower().replace('_', ' ').replace('-', ' ')
    kw_norm = keyword.lower().replace('_', ' ').replace('-', ' ')
    pattern = r'\b' + re.escape(kw_norm) + r'\b'
    return bool(re.search(pattern, label_norm))

unmapped = []
for idx in range(1000):
    _, class_label, _ = decoded_all[idx][0]
    matched = False
    for cifar10_class, keywords in IMAGENET_TO_CIFAR10_MAP_KEYWORDS.items():
        for kw in keywords:
            if match_keyword(class_label, kw):
                matched = True
                break
        if matched:
            break
    if not matched:
        unmapped.append(class_label)

# Fuzzy match the unmapped list with generic terms
broad_terms = {
    "Dog": ['dog', 'pup', 'canine', 'terrier', 'spaniel', 'hound', 'retriever', 'shepherd', 'corgi', 'poodle'],
    "Cat": ['cat', 'feline', 'kitty', 'kitten', 'leopard', 'tiger', 'lion', 'jaguar', 'cheetah', 'puma', 'cougar'],
    "Bird": ['bird', 'fowl', 'avian', 'eagle', 'hawk', 'falcon', 'owl', 'duck', 'goose', 'swan', 'penguin', 'finch'],
    "Deer": ['deer', 'elk', 'moose', 'antelope', 'gazelle'],
    "Frog": ['frog', 'toad', 'amphibian'],
    "Horse": ['horse', 'colt', 'foal', 'stallion', 'mare', 'pony', 'equine', 'zebra', 'donkey', 'mule'],
    "Airplane": ['plane', 'aircraft', 'jet', 'aviation'],
    "Automobile": ['car', 'auto', 'vehicle', 'wheel', 'motor'],
    "Truck": ['truck', 'van', 'wagon'],
    "Ship": ['ship', 'boat', 'vessel', 'watercraft', 'yacht', 'sail']
}

print("Fuzzy checks on unmapped ImageNet classes:")
for label in unmapped:
    label_lower = label.lower()
    for category, terms in broad_terms.items():
        for term in terms:
            if term in label_lower:
                print(f" - {category} fuzzy match: {label} (matched '{term}')")
                break
