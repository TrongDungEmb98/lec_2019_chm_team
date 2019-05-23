import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
TEST_DATA_PATH = "./ImageData/test"
TRAIN_DATA_PATH = "./ImageData/train"
CATEGORIES = ["hand","one","punch","right"]
IMAGE_SIZE = 50
TRAIN_DATA_PROCESS = 1
TEST_DATA_PROCESS = 0
def createTrainData(path_to_data):
    data = []
    for category in CATEGORIES:
        path = os.path.join(path_to_data,category)
        class_number = CATEGORIES.index(category)
        for dataset_image in os.listdir(path):
            try:
                im_gray = cv2.imread(os.path.join(path,dataset_image),cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.resize(im_gray,(IMAGE_SIZE,IMAGE_SIZE))
                data.append([im_gray,class_number])
            except Exception as e:
                pass
    return data

def randomTrainDataSet(training_data):
    feature_set = []
    label_set = []
    random.shuffle(training_data)
    for features,labels in training_data:
        feature_set.append(features)
        label_set.append(labels)
    feature_set = np.array(feature_set).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
    return feature_set,label_set

def saveDataFile(feature_set,label_set,isTrainOrTest):
	if(isTrainOrTest == TRAIN_DATA_PROCESS):
		suffixe = "train"
	else:
		suffixe = "test"
	pickle_out = open("X_"+suffixe+"1.pickle","wb")
	pickle.dump(feature_set,pickle_out)
	pickle_out.close
	pickle_out = open("y_"+suffixe+"1.pickle","wb")
	pickle.dump(label_set,pickle_out)
	pickle_out.close

def HandlingDataTrain(isTrainOrTest):
	feature_set= []
	label_set = []
	training_data = []
	if(isTrainOrTest == TRAIN_DATA_PROCESS):
		training_data = createTrainData(TRAIN_DATA_PATH)
	else:
		training_data = createTrainData(TEST_DATA_PATH)
	feature_set, label_set = randomTrainDataSet(training_data)
	saveDataFile(feature_set,label_set,isTrainOrTest)
	print("data is processed successfully!!")
if __name__ == '__main__':
    #HandlingTestData()
    HandlingDataTrain(TRAIN_DATA_PROCESS)
    HandlingDataTrain(TEST_DATA_PROCESS)
