import os
import cv2
import mediapipe as mp
import csv
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pose = mp_pose.Pose()
    results = pose.process(image_rgb)
    
    keypoints_data = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints_data.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    pose.close()
    return keypoints_data


if __name__ == '__main__':
    data_root = 'data'
    class_folders = [folder for folder in os.listdir(data_root)]
    with open('pose_keypoints.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for class_folder in class_folders[-4:]:
            print(class_folder)
            image_files = os.listdir(os.path.join(data_root, class_folder))
            for image_file in image_files:
                image_path = os.path.join(data_root, class_folder, image_file)
                # print(image_path)
                keypoints = extract_keypoints(image_path)
                if len(keypoints) == 33:
                    csv_writer.writerow([class_folder] + [item for sublist in keypoints for item in sublist])
                else:
                    print(image_path)
