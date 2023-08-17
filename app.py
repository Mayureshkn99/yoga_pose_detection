import os
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

        keypoints = utils.extract_keypoints(dest)
        if len(keypoints) != 33*4:
            detection = False
            result = None
        else:
            result = MODEL.predict([keypoints])[0]
            detection = True

        return render_template('complete.html', reference_image=result+".png", input_image=file.filename, detection=detection, result=result)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
