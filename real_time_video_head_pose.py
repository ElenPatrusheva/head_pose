import numpy as np
import cv2
import imutils
import argparse
import dlib


def get_2d_coordinates(image, shape_predictor):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    # load the input image, resize it, and convert it to grayscale
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    faces = []
    appropriate = [30, 8, 36, 45, 48, 54]  
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        res = []
        for a in [33, 8, 36, 45, 48, 54]:
            res.append((shape.part(a).x, shape.part(a).y))	
        shape = res
        faces.append(res)
    return faces


def head_phose(im, shape_predictor):
    size = im.shape
    faces = get_2d_coordinates(im, shape_predictor)
    for f in faces:
        #reshape to init
        f = [(int(i*size[1]/500), int(j*size[1]/500)) for i, j in f] 
        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                    f[0],     # Nose tip
                                    f[1],     # Chin
                                    f[2],     # Left eye left corner
                                    f[3],     # Right eye right corne
                                    f[4],     # Left Mouth corner
                                    f[5]      # Right mouth corner
                                ], dtype="double")

        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                
                                ])
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )


        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(im, p1, p2, (255,0,0), 2)

        # Display image
    return im 


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())
shape_predictor = args['shape_predictor']

cap = cv2.VideoCapture(0)
ret,img=cap.read()
flag = 0
while(True):
    flag += 1
    ret, frame = cap.read()
    if flag != 15:
        continue
    # only 1/15 frames is processed, because slow (no gpu used)
    flag=0
    w = 500
    h = int(img.shape[0] * 500 / img.shape[1])
    frame = cv2.resize(frame,(w, h), fx = 500 / frame.shape[1])
    processed = head_phose(frame, shape_predictor)

    cv2.imshow('frame',processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()