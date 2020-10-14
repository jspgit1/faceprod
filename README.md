# faceprod
Springboard ML course repo
This repo is for my capstone step 12: Run Your Code End-to-End with Logging and testing.
There are two python program files that I moved out from jupyter notebook: face_model_train.py and face_recog_webcam.py
The face_model_train.py uses a set of images to train a KNN model. Just run command as below:
$ python face_model_train.py
a trained ML model file face_model_file_frg will be saved in current folder
The following command will launch a openCV window and detect face from webcam. It shows the face is a name of the training image groups or a unknown person.
$ python face_recog_webcam.py
