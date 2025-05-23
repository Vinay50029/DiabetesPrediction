from flask import Flask, render_template, request,make_response, redirect, url_for, session, flash
import os
import numpy as np
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels = ['Mild DR', 'Moderate DR', 'No DR', 'Proliferative DR', 'Severe DR']

densenet_model = load_model('model/densenet_weights.hdf5')
# resnet_model = load_model('model/resnet_weights.hdf5')


def is_valid_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not loaded.")
        return False

    h, w, c = img.shape
    if h < 100 or w < 100 or c != 3:
        print("Image too small or not a color image.")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    print("Brightness:", brightness)
    if brightness < 24 or brightness > 145:
        print("Image too dark or too bright.")
        return False

    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=30, maxRadius=150
    )
    circle_count = len(circles[0]) if circles is not None else 0
    print("Circle Count:", circle_count)
    if circle_count > 5:
        print("Circular retina-like pattern detected.")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 30, 50), (10, 255, 255))   # red
    mask2 = cv2.inRange(hsv, (10, 50, 50), (25, 255, 255))  # orange
    mask = cv2.bitwise_or(mask1, mask2)
    coverage = np.sum(mask > 0) / (h * w)
    print("Coverage:", coverage)
    if coverage <= 0.99 and coverage >= 0.10:
        print(" Enough red/orange coverage — likely retina.")
        return True
    else:
        print(" Not enough red/orange coverage.")
        return False


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255.0
    return img

def get_best_model_prediction(image):
    
    den_pred = densenet_model.predict(image)
    den_label = labels[np.argmax(den_pred)]
    confidence = float(np.max(den_pred) * 100)
    model_used = 'DenseNet121'
    return den_label, confidence, model_used
# def get_resnet_prediction(image):
#     pred = resnet_model.predict(image)
#     label = labels[np.argmax(pred)]
#     confidence = float(np.max(pred) * 100)
#     model_used = 'ResNet101'
#     return label, confidence, model_used


@app.route('/')
def home():
    response = make_response(render_template('home.html'))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/about')
def about():
    return render_template('about.html')

app.secret_key = 'your_secret_key_here'  # Add this if not already present
@app.route('/route_decider', methods=['POST'])
def route_decider():
    session['Name'] = request.form.get('Name')
    session['Age'] = request.form.get('Age')
    session['Gender'] = request.form.get('Gender')
    session['Email'] = request.form.get('email')

    action = request.form.get('action_type')
    if action == 'pic':
        return redirect(url_for('predict'))
    elif action == 'text':
        return redirect(url_for('predict_text'))
    else:
        return "Invalid action type", 400


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    left_file = request.files.get('left_image')
    right_file = request.files.get('right_image')

    results = {}

    if left_file and left_file.filename:
        left_filename = secure_filename(left_file.filename)
        left_path = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
        left_file.save(left_path)
        
        # is_valid_image(left_path)
        if not is_valid_image(left_path):
            flash("Left image is not a valid retina scan.")
            return redirect(url_for('predict'))
        
        left_input = preprocess_image(left_path)
        left_pred, left_conf, left_model = get_best_model_prediction(left_input)

        results['left_result'] = left_pred
        results['left_confidence'] = left_conf
        results['left_model'] = left_model
        results['left_image'] = left_path

    if right_file and right_file.filename:
        right_filename = secure_filename(right_file.filename)
        right_path = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
        right_file.save(right_path)
        
        
        # is_valid_image(right_path)
        if not is_valid_image(right_path):
            flash("Right image is not a valid retina scan.")
            return redirect(url_for('predict'))

        right_input = preprocess_image(right_path)
        right_pred, right_conf, right_model = get_best_model_prediction(right_input)


        results['right_result'] = right_pred
        results['right_confidence'] = right_conf
        results['right_model'] = right_model
        results['right_image'] = right_path

    if not results:
        flash("please upload at least one image.")
        return redirect(url_for('predict'))


    return render_template('result.html',
                            # left_result=left_pred, left_model=left_model, left_image=left_path, right_result=right_pred, right_model=right_model, right_image=right_path,
                            name=session.get('Name'),
                            age=session.get('Age'),
                            gender=session.get('Gender'),
                            email=session.get('Email'),
                            **results)
    

import joblib
diabetes_model = joblib.load("model/lightgbm_real_diabetes_model.pkl")

@app.route('/predict_text', methods=['GET', 'POST'])
def predict_text():
    if request.method == 'GET':
        return render_template('predict_text.html')

    def yes_no_to_binary(value):
        return 1 if value.lower() == 'yes' else 0

    features = np.array([[  
        1 if request.form['Gender'] == 'Male' else 0,
        int(request.form['Age']),
        yes_no_to_binary(request.form['Polyuria']),
        yes_no_to_binary(request.form['Polydipsia']),
        yes_no_to_binary(request.form['sudden_weight_loss']),
        yes_no_to_binary(request.form['weakness']),
        yes_no_to_binary(request.form['Polyphagia']),
        yes_no_to_binary(request.form['Genital_thrush']),
        yes_no_to_binary(request.form['visual_blurring']),
        yes_no_to_binary(request.form['Itching']),
        yes_no_to_binary(request.form['Irritability']),
        yes_no_to_binary(request.form['delayed_healing']),
        yes_no_to_binary(request.form['partial_paresis']),
        yes_no_to_binary(request.form['muscle_stiffness']),
        yes_no_to_binary(request.form['Alopecia']),
        yes_no_to_binary(request.form['Obesity']),
    ]])

    prediction = diabetes_model.predict(features)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return render_template("result_text.html", 
                           result=result,
                           name=session.get('Name'),
                       age=session.get('Age'),
                       gender=session.get('Gender'),
                       email=session.get('Email'))

if __name__ == "__main__":
    app.run(debug=True)

 # if 'left_image' not in request.files or 'right_image' not in request.files:
    #     return "Please upload both eye images."

    # left_file = request.files['left_image']
    # right_file = request.files['right_image']
    # left_filename = secure_filename(left_file.filename)
    # right_filename = secure_filename(right_file.filename)

    # left_path = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
    # right_path = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
    # left_file.save(left_path)
    # right_file.save(right_path)

    # left_input = preprocess_image(left_path)
    # right_input = preprocess_image(right_path)

    # left_pred, left_conf, left_model = get_best_model_prediction(left_input)
    # right_pred, right_conf, right_model = get_best_model_prediction(right_input)
