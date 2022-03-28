import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math
from keras.models import load_model

from classification import training, getLabel

### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    cv2.imshow('CONTRAST IMAGE', image)
    image = LaplacianOfGaussian(image)
    cv2.imshow('LAPLACIAN OF GAUSSIAN IMAGE', image)
    image = binarization(image)
    return image

# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign 
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def prepro(img,model1):
	#Preprocess image
	img = np.asarray(img)
	img = cv2.resize(img, (32, 32))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	img = img/255

	#Reshape reshape
	img = img.reshape(1, 32, 32, 1)
	resultid=model1.predict_classes(img)
	name=['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vechiles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles', 'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vechiles over 3.5 metric tons']

	print('Result :',name[int(resultid)])
	return name[int(resultid)]
	



def cropSign(image, coordinate, model1):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    image1=image[top:bottom,left:right]
    text = prepro(image1,model1)
    return text
    # return image[top:bottom,left:right]


def findLargestSign(image, contours, threshold, distance_theshold, model1):
    max_distance = 0
    text =""
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            text = cropSign(image,coordinate,model1)
    return text, coordinate


def findSigns(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type, model1):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))

    #binary_image = remove_line(binary_image)

    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    text, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15, model1)
    
    return coordinate, original_image, text

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask

def main(args):
	#Clean previous image    
    # clean_images()
    #Training phase
    model = training()
    print("ARGS SOURCE:",args.source)
    if args.source =="0":
    	vidcap = cv2.VideoCapture(0)
    else:
    	vidcap = cv2.VideoCapture(args.file_name)   #change to 0 for webcam  args.file_name
    width = vidcap.set(3,320)  # float
    height = vidcap.set(4,240) # float
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = 0.65   # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    
    modelfile='trafficmodel.h5'
    print('Model Selected : ',modelfile)
    model1 = load_model(modelfile)

    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        #frame = cv2.resize(frame, (640,int(height/(width/640))))
        frame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        coordinate, image, text = localization(frame, args.min_size_components, args.similitary_contour_with_circle, model, count, current_sign, model1)
        if coordinate is not None:
            # cv2.rectangle(image, coordinate[0],coordinate[1], (255, 255, 255), 1)
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
        

        cv2.imshow('Result', image)
        count = count + 1
        #Write to video
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.write("{}".format(sign_count))
    for pos in coordinates:
        file.write("\n{} {} {} {} {} {}".format(pos[0],pos[1],pos[2],pos[3],pos[4], pos[5]))
    print("Finish {} frames".format(count))
    file.close()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NLP Assignment Command Line")
    
    parser.add_argument(
      '--file_name',
      default= "./MVI_1049.avi",
      help= "Video to be analyzed"
      )

    parser.add_argument(
      '--source',
      help= "Live Video Access"
      )
    
    parser.add_argument(
      '--min_size_components',
      type = int,
      default= 300,
      help= "Min size component to be reserved"
      )

    
    parser.add_argument(
      '--similitary_contour_with_circle',
      type = float,
      default= 0.65,
      help= "Similitary to a circle"
      )
    
    args = parser.parse_args()
    main(args)
