import cv2
import numpy as np
import time
# from keras.models import load_model
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
cap = cv2.VideoCapture(1)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_1_2.mp4',fourcc, 20.0, (640,480))
IMAGE_SIZE = 50
is_bgr_taken = False
background = None
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480
MIN_COODINATE_X = 0
MAX_COORDINATE_X = 640
MIN_COODINATE_Y = 0
MAX_COORDINATE_Y = 480
MIN_SQUARE_WIDTH = 200
FONT = cv2.FONT_HERSHEY_SIMPLEX
handImageCount = 0
punchImageCount = 0
rightImageCount = 0
DATA_PATH_HAND = "./Hand/"
DATA_PATH_PUNCH = "./Punch/"
DATA_PATH_RIGHT = "./Right/"
DATA_PATH_ONE = "./One/"
label = ["hand","punch","right"]
listRoi = []
ImageCount = 0
N_RATIO_FRAME = 0
# Define the codec and create VideoWriter object
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
            raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def extract_foreground(background,frame):
    # Find the absolute difference between the background frame and current frame
    diff = cv2.absdiff(background, frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the mask
    th, thresh = cv2.threshold(mask,30, 255, cv2.THRESH_BINARY)

    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)
    # Get mask indexes
    imask = img_dilation > 0
    # Get foreground from mask
    foreground = mask_array(frame, imask)

    return foreground, mask, img_dilation,diff
def tracking(frame,img_dilation):
    global ImageCount,N_RATIO_FRAME
    contour_list, hierarchy = cv2.findContours(img_dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contourArea = []
    white_pixel = 0
    if(len(contour_list) > 0):
        indexOfBiggestContour = -1
        sizeOfBiggestContour = 0
        for i in range(len(contour_list)):
            if(cv2.contourArea(contour_list[i]) > sizeOfBiggestContour and cv2.contourArea(contour_list[i]) > 500 ):
                sizeOfBiggestContour = cv2.contourArea(contour_list[i])
                indexOfBiggestContour = i
        if(indexOfBiggestContour >= 0):
            N_RATIO_FRAME = N_RATIO_FRAME + 1
            x,y,w,h = cv2.boundingRect(contour_list[indexOfBiggestContour])
            centroid,SquareCentroid_X,SquareCentroid_Y,width = handContourExtract(x,y,w,h)
            drawSquareBounding(frame,SquareCentroid_X,SquareCentroid_Y,width)
            x1,y1,x2,y2 = squareCoordinate(SquareCentroid_X,SquareCentroid_Y,width)
            # Roi = img_dilation[x1-20:x2+20,y1-20:y2-20]
            Roi = img_dilation[y1:y2,x1:x2]
            cv2.imshow('roi',Roi)
            if(N_RATIO_FRAME % 10) == 0:    
                listRoi.append(Roi)
            if(len(listRoi) >= 100):
                saveData(listRoi)
                print(ImageCount)
                if(ImageCount == 10):
                    exit()
        else:
            print("no contours found")
def saveData(listRoiImages):
    global handImageCount,punchImageCount,rightImageCount,Roi,ImageCount
    for roi in listRoiImages:
        cv2.imwrite(DATA_PATH_RIGHT+str(handImageCount)+"_3.jpg",roi)
        handImageCount = handImageCount+1
    listRoiImages.clear()
    ImageCount = ImageCount+1   

def handContourExtract(x,y,width,height):
    centroid = (int(x+width/2),int(y+height/2))
    if(width < height and (MAX_COORDINATE_X - x) > height):
        width = height
    elif(width<height and (MAX_COORDINATE_X - x) < height):
        width = MAX_COORDINATE_X - x
    elif(width > height and (MAX_COORDINATE_Y - y) > width):
        height = width
    elif(width > height and (MAX_COORDINATE_Y - y) < width):
        height = (MAX_COORDINATE_Y - y)
    (SquareCentroid_X,SquareCentroid_Y) = (int(x+width/2),int(y+height/2))
    return  centroid,SquareCentroid_X,SquareCentroid_Y,width
def squareCoordinate(SquareCentroid_X,SquareCentroid_Y,width):
    x_new1 = SquareCentroid_X-int(width/2)
    y_new1 = SquareCentroid_Y-int(width/2)
    x_new2 = SquareCentroid_X+int(width/2)
    y_new2 = SquareCentroid_Y+int(width/2)
    return x_new1,y_new1,x_new2,y_new2
def drawSquareBounding(frame,SquareCentroid_X,SquareCentroid_Y,width):
    x_new1,y_new1,x_new2,y_new2 = squareCoordinate(SquareCentroid_X,SquareCentroid_Y,width)
    cv2.rectangle(frame,(x_new1,y_new1),(x_new2,y_new2),(255,255,0),2)
def processDataToPredict(Roi):
    isNotPredictable = False
    if(Roi.shape[0] == 0 or Roi.shape[1]==0):
        isNotPredictable = True
    else:
        Roi = cv2.resize(Roi,(IMAGE_SIZE,IMAGE_SIZE))
        Roi = np.array(Roi).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
        Roi = Roi/255
    return Roi, isNotPredictable
def predictGesture(model,Roi):
    Roi,isNotPredictable = processDataToPredict(Roi)
    if isNotPredictable == True:
        return "None",0
    predict = model.predict_classes(Roi)
    predict_pro = model.predict(Roi)
    print(predict_pro)
    return label[predict[0]],predict_pro[0][predict[0]]
def main():
    global is_bgr_taken,background
    #model = load_model("model_hand_gesture.h5")
    while True:
        if(is_bgr_taken == False):
            print("press z to take a background picture!!")
        while(is_bgr_taken == False):
            ret,frame = cap.read()
            frame = cv2.flip(frame,1)
            cv2.imshow('frame',frame)
            if(cv2.waitKey(1) & 0xFF == ord('z')):
                ret,background = cap.read()
                background = cv2.flip(background,1)
                is_bgr_taken = True
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        foreground,mask,img_dilation ,diff= extract_foreground(background,frame)
        tracking(frame,img_dilation)
        # out.write(frame)
        cv2.imshow('frame', frame)
        #cv2.imshow('foreground', foreground)
        #cv2.imshow('mask', mask)
        #cv2.imshow('diff', diff)
        cv2.imshow('img_dilation', img_dilation)   
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()