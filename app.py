# import keras.utils as image
# from keras.preprocessing import image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.utils import send_from_directory
import cv2 as cv
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('ambatik-model.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/'

def finds(filename):
    
	vals = ['cendrawasih', 'geblek renteng', 'insang', 'kawung', 'mega mendung', 'parang', 'pring sedapur', 'simbut', 'sogan', 'truntum']
	path = "uploaded/" + filename

	img_width, img_height = 256, 256
	img = tf.keras.utils.load_img(path, target_size = (img_width, img_height))
	img = tf.keras.utils.img_to_array(img)
	img = np.expand_dims(img, axis = 0)

	pred = model.predict(img)
	
	pred_index = np.argmax(pred[0])
	result = {'types': vals[pred_index], 'accuracy': pred[0][pred_index] * 100}
	# result = (vals[pred_index] + ": " + str(pred[0][pred_index] * 100) + "%")

	# print(pred)
	# return str(vals[pred_index])
	return result 



# Predict Image in cloud storage
def predictCloudStorage(filename):
    vals = ['cendrawasih', 'geblek renteng', 'insang', 'kawung', 'mega mendung', 'parang', 'pring sedapur', 'simbut', 'sogan', 'truntum']

    gcs_url = f'https://storage.googleapis.com/user-photo-profile/batik_prediction_photo/{filename}'


	# Specify the destination directory within your project folder
    project_folder = os.path.dirname(__file__) 
    destination_dir = os.path.join(project_folder, 'batik_gcs')
    os.makedirs(destination_dir, exist_ok=True)
    
    # Download the image using tf.keras.utils.get_file
    local_path = tf.keras.utils.get_file(filename, gcs_url, cache_dir=destination_dir, cache_subdir='.')
    
    img_width, img_height = 256, 256

    img = tf.keras.utils.load_img(local_path, target_size = (img_width, img_height))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    pred = model.predict(img)
	
    pred_index = np.argmax(pred[0])
    result = {'types': vals[pred_index], 'accuracy': pred[0][pred_index] * 100}
	# result = (vals[pred_index] + ": " + str(pred[0][pred_index] * 100) + "%")
    os.remove(local_path)
	# print(pred)
	# return str(vals[pred_index])
    return result 


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds(f.filename)
		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		return jsonify(val)

@app.route('/process-image', methods=['POST'])
def process_image():
    # Get the file name from the request
    file_name = request.json.get('fileName')
    
    # Process the image and get the result
    result = predictCloudStorage(file_name)

    # Return the result
    print(result)
    return result

if __name__ == '__main__':
	app.run()
