import utils
import matplotlib.pyplot as plt
import numpy as np
import joblib
import random

if __name__ == "__main__":
    model = joblib.load('model.joblib')
    file_path = 'uploads/image.png'  # include filepath from Flask uploaded image
    keypoints = utils.extract_keypoints(file_path)
    result = model.predict([keypoints])[0]  # display this on the results page
    print(result)