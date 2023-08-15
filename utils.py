import cv2
import mediapipe as mp

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
            keypoints_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    pose.close()
    return keypoints_data