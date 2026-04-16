from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("food_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

calorie_dict = {
    'apple_pie': 237, 'baby_back_ribs': 292, 'baklava': 334,
    'beef_carpaccio': 120, 'beef_tartare': 180, 'beet_salad': 150,
    'beignets': 290, 'bibimbap': 490, 'bread_pudding': 300,
    'breakfast_burrito': 450, 'bruschetta': 150, 'caesar_salad': 180,
    'cannoli': 250, 'caprese_salad': 220, 'carrot_cake': 350,
    'ceviche': 200, 'cheese_plate': 400, 'cheesecake': 321,
    'chicken_curry': 240, 'chicken_quesadilla': 300, 'chicken_wings': 430,
    'chocolate_cake': 371, 'chocolate_mousse': 280, 'churros': 300,
    'clam_chowder': 200, 'club_sandwich': 350, 'crab_cakes': 240,
    'creme_brulee': 300, 'croque_madame': 500, 'cup_cakes': 305,
    'deviled_eggs': 160, 'donuts': 260, 'dumplings': 220,
    'edamame': 120, 'eggs_benedict': 420, 'escargots': 180,
    'falafel': 333, 'filet_mignon': 250, 'fish_and_chips': 600,
    'foie_gras': 460, 'french_fries': 365, 'french_onion_soup': 200,
    'french_toast': 350, 'fried_calamari': 300, 'fried_rice': 330,
    'frozen_yogurt': 150, 'garlic_bread': 250, 'gnocchi': 250,
    'greek_salad': 200, 'grilled_cheese_sandwich': 400, 'grilled_salmon': 280,
    'guacamole': 160, 'gyoza': 250, 'hamburger': 354,
    'hot_and_sour_soup': 150, 'hot_dog': 290, 'huevos_rancheros': 300,
    'hummus': 166, 'ice_cream': 207, 'lasagna': 350,
    'lobster_bisque': 250, 'lobster_roll_sandwich': 400,
    'macaroni_and_cheese': 310, 'macarons': 100, 'miso_soup': 84,
    'mussels': 230, 'nachos': 346, 'omelette': 154,
    'onion_rings': 276, 'oysters': 50, 'pad_thai': 350,
    'paella': 400, 'pancakes': 227, 'panna_cotta': 300,
    'peking_duck': 337, 'pho': 350, 'pizza': 285,
    'pork_chop': 250, 'poutine': 500, 'prime_rib': 400,
    'pulled_pork_sandwich': 450, 'ramen': 436, 'ravioli': 250,
    'red_velvet_cake': 360, 'risotto': 350, 'samosa': 262,
    'sashimi': 200, 'scallops': 200, 'seaweed_salad': 90,
    'shrimp_and_grits': 400, 'spaghetti_bolognese': 350,
    'spaghetti_carbonara': 380, 'spring_rolls': 200,
    'steak': 271, 'strawberry_shortcake': 300, 'sushi': 200,
    'tacos': 300, 'takoyaki': 300, 'tiramisu': 300,
    'tuna_tartare': 180, 'waffles': 291
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = Image.open(filepath).resize((160,160))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        pred = class_names[np.argmax(score)]
        confidence = np.max(score) * 100
        calories = calorie_dict.get(pred, "Unknown")

        result = (pred, confidence, calories, filepath)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)