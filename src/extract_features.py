import os
import shutil

import cv2
from nsfw_detector import predict


# path = 'D:\BE Project\adult_content_detection\adult_content_detection\src\saved_model.h5'
# model = predict.load_model(path)


def extract_frames(video_path, output_dir, frame_interval):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        print(frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")


def get_file_names(directory_path):
    file_list = os.listdir(directory_path)
    return [file for file in file_list if os.path.isfile(os.path.join(directory_path, file))]


def split_safe_unsafe(directory_path):
    file_names = get_file_names(directory_path)
    file_names = [f'{directory_path}/' + file_name for file_name in file_names]
    safe_dir = 'dataset/safe/'
    unsafe_dir = 'dataset/unsafe/'
    for file_name in file_names:
        score = predict.classify(model, file_name)
        score = list(score.values())[0]
        if score['neutral'] >= 0.7:
            shutil.move(file_name, safe_dir)
        else:
            shutil.move(file_name, unsafe_dir)
