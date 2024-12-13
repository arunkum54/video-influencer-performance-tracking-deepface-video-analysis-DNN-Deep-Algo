import os
import pandas as pd
import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from collections import defaultdict

def extract_faces_from_video(video_path, output_folder="faces", net=None, conf_threshold=0.5):
    """
    Extracts faces from a video using a preloaded DNN model.
    Args:
        video_path (str): Path to the video.
        output_folder (str): Folder to save the extracted faces.
        net (cv2.dnn_Net): Preloaded DNN model.
        conf_threshold (float): Confidence threshold for detections.
    Returns:
        List of file paths to validated face images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []

    frame_count = 0
    face_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        try:
            net.setInput(blob)
            detections = net.forward()
        except cv2.error as e:
            print(f"Error during forward pass of the DNN model: {e}")
            break

        face_detected = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                face_img = frame[y1:y2, x1:x2]
                face_filename = os.path.join(output_folder, f"frame{frame_count}_face{i}.jpg")
                cv2.imwrite(face_filename, face_img)

                if validate_face(face_filename):
                    face_paths.append(face_filename)
                else:
                    os.remove(face_filename)

        if not face_detected:
            print(f"No face detected in frame {frame_count} of video {video_path}")

        frame_count += 1

    cap.release()
    return face_paths


def validate_face(image_path):
    """
    Validate if an image is a real face using DeepFace.
    Args:
        image_path (str): Path to the image to validate.
    Returns:
        bool: True if the image is a real face, False otherwise.
    """
    try:
        DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=True)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Load DNN model
    prototxt_path = "/Users/arunkumarrana/Desktop/NN/deploy.prototxt"
    model_path = "/Users/arunkumarrana/Desktop/NN/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}. Ensure the file exists.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Caffe model file not found at {model_path}. Ensure the file exists.")

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Load video data
    csv_file_path = "/Users/arunkumarrana/Desktop/NN/Data.csv"
    data = pd.read_csv(csv_file_path)

    # Process downloaded videos
    all_faces = []
    video_folder = "downloaded_videos"
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        video_filename = os.path.join(video_folder, os.path.basename(row['Video URL']))
        if os.path.exists(video_filename):
            faces = extract_faces_from_video(video_filename, output_folder="faces", net=net)
            if not faces:
                print(f"No faces detected in video: {video_filename}")
            all_faces.extend(faces)
