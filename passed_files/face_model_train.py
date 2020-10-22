# Use face_recognition to identify masked face
# 使用frg检测人脸训练knn模型
import numpy as np
import os
import face_recognition as frg
from sklearn.neighbors import KNeighborsClassifier
import re
import math
import matplotlib.pyplot as plt
import pickle 


# We define a train function

def kntrain(X, y, neighbors, kn_alg, weight):
    if neighbors is None:
        neighbors = int(math.sqrt(len(X)))
    klf1 = KNeighborsClassifier(algorithm=kn_alg, n_neighbors=neighbors, weights=weight)
    klf1.fit(X,y)
    return klf1, neighbors         

# Train KNN model
# Create training matrix X, y
from timeit import default_timer as timer
from datetime import timedelta
start = timer()

extension = ['jpg','png','bmp','jpeg']
X =[]
y =[]

tfiles = 0  #Total number of train files        #training sample number
dfiles = 0  #Number of files detected face

for (root,dirs,files) in os.walk('maskedface4'):
    pattern = '^\w+/train/\w+'
    if re.match(pattern, root):
        print('root:',root)
        #print('files:',files)
        label0 = root.split('/')[-1]
        for imgf in files:
            imgf = imgf.lower()
            if imgf.split('.')[1] in extension:
                imgpath = os.path.join(root, imgf)
                tfiles += 1
                #将图像文件（.jpg，.png等）加载到numpy数组中
                npimg = frg.load_image_file(imgpath, mode='RGB')
                # Use model='hog' for non-masked face. Use model='cnn' for masked face
                f_location = frg.face_locations(npimg, model='cnn')    #Set model as cnn or hog

                if len(f_location) == 1:
                    #print('fpath:',imgpath)
                    #print('f_location:',f_location)   
                    f_encord = frg.face_encodings(npimg,known_face_locations=f_location)[0]
                    X.append(f_encord)
                    y.append(label0)
                    dfiles += 1
                else:
                    print('Incorrect face image!')    
            else:
                print('File $s has wrong format' % imgf)

end = timer()
print('Processing images elapsed time:',timedelta(seconds=end-start))

#Adjust neighbors, kn_alg(Algorithm), weight
klf, neighbor = kntrain(X, y, neighbors=5, weight='distance', kn_alg='ball_tree')

#Save y list to a file in current folder
y_path = 'y_list.txt'
with open(y_path, 'w') as filehandle:
    filehandle.writelines("%s\n" % line for line in y)

#Save trained model
# Open a pickle channel to save trained model in current folder
model_path = 'face_model_file_frg'
model_saving = open(model_path, 'wb')
pickle.dump(klf, model_saving)

print('Saved model file:',model_path)
print('Number of neighbors:', neighbor)
print('Face detection rate of train samples:', (dfiles/tfiles))
print('Number of train sample files:', tfiles)
end = timer()
print('Train procedure elapsed time:',timedelta(seconds=end-start))

