# knn_face_reg_cam.py
# 该程序可载入已训练好的knn模型并使用摄像头检测并识别人脸，检测使用face_locations
import numpy as np
import os
import face_recognition as frg
import re
import math
import pickle
import cv2
from PIL import Image, ImageDraw, ImageFont


def predict(X_frame, klf=None, model_path=None, distance_threshold=0.4):
    # Load a trained KNN model
    
    with open(model_path, 'rb') as f:
        klf = pickle.load(f)

    f_locations = frg.face_locations(X_frame,model='cnn')
    
    # If no faces are found in the image, return an empty result.
    if len(f_locations) == 0:
        return []
    # Find encodings for faces in the test image
    faces_encodings = frg.face_encodings(X_frame, known_face_locations=f_locations)
    
    # Use the KNN model to find the best matches for the test face
    closest_distances = klf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(f_locations))]
    
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("UNKNOWN", loc) for pred, loc, rec in zip(klf.predict(faces_encodings), f_locations, are_matches)]

def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.
    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame) #array转换成image
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":

    # process one frame in every 15 frames for speed 每15帧图片检测1次
    process_this_frame = 14
    print('准备摄像头中...')
    cap = cv2.VideoCapture(-1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('当前fps为：',fps)
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 15 == 0:
                predictions = predict(img, model_path="face_model_file_frg")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame) #第一个参数是窗口名称（对话框的名称），字符串类型。第二个参数是我们的图像。
            if ord('q') == cv2.waitKey(10):#按q关闭窗口
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
