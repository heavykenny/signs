import pandas as pd

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
import keras

from werkzeug.utils import secure_filename

app = Flask(__name__)
base_path = os.path.dirname(os.path.abspath(__file__))

model_path = 'build/_traffic_sign_model.keras'
model = keras.models.load_model(model_path)

csv_path = os.path.join(base_path, f"{base_path + '/datasets/Meta.csv'}")  # Update with your actual CSV file path
df = pd.read_csv(csv_path)

# Construct full image paths for the HTML
df['image_url'] = df['Path'].apply(lambda x: os.path.join('/static/images', x))

# Extracting the numeric part of the path and converting it to integer
df['sort_key'] = df['Path'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))

df['index'] = df['sort_key']

# Sorting the DataFrame by the numeric key
df = df.sort_values(by='sort_key')

# Dropping the temporary sort_key if no longer needed
df.drop('sort_key', axis=1, inplace=True)

# Convert the DataFrame to a list of dictionaries for easier handling in the template
images_data = df[['image_url', 'ClassId', 'index']].to_dict(orient='records')

classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
           'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
           'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
           'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry', 'General caution',
           'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road',
           'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
           'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End speed + passing limits',
           'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
           'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons'
           ]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', images_data=images_data, classes=classes)


@app.route('/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image part'
    img_file = request.files['image']
    if img_file.filename == '':
        return 'No selected image'
    if img_file:
        file_path = os.path.join(
            base_path, 'static', 'images', secure_filename(img_file.filename))
        img_file.save(file_path)
        img = tf.keras.utils.load_img(file_path, target_size=(32, 32))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Debugging: Print the shape of the image array
        print(f"Debug - Image shape: {img_array.shape}")

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        confidence = np.max(prediction) * 100

        return render_template('index.html', images_data=images_data, class_id=predicted_class_index,
                               class_name=classes[predicted_class_index], confidence=f"{confidence:.2f}%",
                               classes=classes)
    return 'Error'


if __name__ == "__main__":
    app.run()
