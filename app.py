import os
import cv2
import utils
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define the allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

TARGET = os.path.join(APP_ROOT, 'static/images/')

if not os.path.isdir(TARGET):
    os.mkdir(TARGET)

MODEL = joblib.load('model.joblib')

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            dest = '/'.join([TARGET, file.filename])
            file.save(dest)

        _, input_keypoints = utils.extract_keypoints(dest)
        if len(input_keypoints) != 33*4:
            detection = False
            result = None
            accuracy = None
        else:
            result = MODEL.predict([input_keypoints])[0]
            detection = True
            reference_results, reference_keypoints = utils.extract_keypoints(f'static/reference/{result}.png')
            accuracy = utils.compare(reference_keypoints, input_keypoints)
            accuracy = f'{accuracy*100:.2f}'
            image = cv2.imread(dest)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ann_image = utils.draw_landmarks_on_image(image_rgb, reference_results)
            ann_image_bgr = cv2.cvtColor(ann_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"static/images/{result}.png", ann_image_bgr)
        print(dest)
        return render_template('complete.html', detection=detection, input_image=file.filename, result=result, accuracy=accuracy)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
