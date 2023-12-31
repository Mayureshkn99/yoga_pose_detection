from unittest import result
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def extract_keypoints_from_path(image_path):
    image = cv2.imread(image_path)
    return extract_keypoints(image)

def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pose = mp_pose.Pose()
    results = pose.process(image_rgb)
    
    keypoints_data = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    pose.close()
    return results, keypoints_data


def points_to_distances(points):
    points = np.array(points)
    points = np.reshape(points, (-1,4))
    # Reference point (the first point in the list)
    head = points[0]
    # Calculate Euclidean distances
    distances = np.array([euclidean(head, point) for point in points])
    return distances


def compare(reference_keypoints, input_keypoints):
    reference_distances = points_to_distances(reference_keypoints)
    input_distances = points_to_distances(input_keypoints)

    # Calculate the cosine similarity
    cosine_sim = cosine_similarity([reference_distances], [input_distances])[0][0]

    # Map the cosine similarity to a range of 0 to 100
    # similarity_measure = ((cosine_sim - 0.95) / (100 - 95)) * 100
    return cosine_sim


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        detection_result.pose_landmarks,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image