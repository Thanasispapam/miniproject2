import cv2
import numpy as np
from math import sqrt
from kalman import kalman




def blob_detection(window_name, detect_image, display_image, connectivity=8):

    ret, thresh = cv2.threshold(detect_image, 0, 255, cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(detect_image, connectivity, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    display_image = detect_image
    # Draw circles where blobs where found
    for i, val in enumerate(centroids):
        if i == 0:
            continue
        cv2.circle(display_image, (int(centroids[i][0]), int(centroids[i][1])), 10, (50, 255, 0), 3, 8, 0)
    return centroids

def background_subtract(video):
    frameCounter = 0
    trackerArray = []
    #kalmanArray = []
    duplicate = False
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history = 200, varThreshold = 200) #Starting back ground subtractor object
    while(1):
        #Loading video, one frame at a time
        ret, frame = video.read()
        #transform the picture
        #frame = transform_perspective(frame)
        #Blurring and applying erosion and dilation to the element used for blob detection
        fgmask = fgbg.apply(frame)
        blur_size = 3
        cv2.blur(frame, (blur_size, blur_size), frame)
        elip_val = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        fgmask = cv2.erode(fgmask, elip_val)
        fgmask = cv2.dilate(fgmask, elip_val, iterations=5)

        #Finding new blobs each 30 frames
        if frameCounter % 200 == 0 or frameCounter == 2:
            print("200 frames passed, updating blobs")

            centroids = blob_detection('test', fgmask, fgmask)
           #Going through all the centroids and comparing them with the tracked blobs, not adding new ones whichs is too close to an existing tracked object
            for i in centroids:
                one_bbox = i
                bbox = (one_bbox[0]-20 ,one_bbox[1]-20, 40, 40 )
                duplicate = False
                for j in range(len(trackerArray)):
                    ok, t = trackerArray[j].update(frame)
                    if sqrt((bbox[0] - t[0])**2 + (bbox[1] - t[1])**2)<50 and ok:
                        duplicate = True
                if duplicate == False:
                    #Creating a new tracker and a new kalman filter for the blob
                    tracker = cv2.TrackerKCF_create()
                    ok = tracker.init(frame, bbox)
                    #kFilter = Kalman(bbox[0], bbox[1])
                    trackerArray.append(tracker)
                    #kalmanArray.append(kFilter)

        '''
        # Update tracker & Kalman filter
        cnt = 0
        for i in trackerArray:
            ok, bbox = i.update(frame)
            kMan = kalmanArray[cnt]
            #Calculating velocity from old positional values and time since last measurements
            deltaX =  bbox[0] - kMan.getX()
            deltaY =  bbox[1] - kMan.getY()
            xVel = deltaX / 0.04 # 1/25 which is the time in seconds from last frame
            yVel = deltaY / 0.04
            kMan.update(bbox[0], bbox[1], xVel, yVel)
            estimate = kMan.estimate()
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv2.rectangle(frame, (estimate[0], estimate[1]), (estimate[0]+10, estimate[1]+10), (0,255,0), 2, 1)
                distance = sqrt(estimate[2]**2+estimate[3]**2)
                #Going from pixels to km/h
                distance = (distance/10.1159156)*3.6
                cv2.putText(frame, str(float(("%.2f" % distance))) + "km/h", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            else:
                del trackerArray[cnt]
                del kalmanArray[cnt]
            cnt = cnt + 1
        '''
        frameCounter = frameCounter +1

        # Display result
        cv2.imshow("Tracking", frame)
        cv2.imshow("blobs", fgmask)
        # Exit if q pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

'''
def transform_perspective(frame):

    points = np.array([[251, 39],[1110, 62],[1141, 685], [178, 659]])

    lenght_in_n2w = (np.sqrt(1110**2 + 62**2) - np.sqrt(251**2 + 39**2))/84.789
    #print(lenght_in_n2w)
    #print(lenght_in_n2w) To get the constant to transform from meter to pixel

    north_to_west = 84.789*lenght_in_n2w # meter, x top direction
    south_to_east = 84.594*lenght_in_n2w # meter, x bottom direction
    north_to_east = 114.315*lenght_in_n2w # meter, y right
    south_to_west = 90.462*lenght_in_n2w # meter, y left
     #go from north_west clock dir end in south_west
    new_corner_positions = np.array([[0, 0], [north_to_west, 0], [south_to_east, north_to_east], [0, south_to_west]])
    h, status =cv2.findHomography(points, new_corner_positions)

    size = (int(max(north_to_west,south_to_east)), int(max(north_to_east, south_to_west)))

    im_out = cv2.warpPerspective(frame, h, size)

    return im_out
'''
#cap = cv2.VideoCapture("2017_06_23_1609 Fra Dyrskuepladsen mod krydset Søndre Boulevard-UPUQtdMUkoY.mp4")

if __name__ == '__main__' :
    video = cv2.VideoCapture \
        ("Relaxing highway traffic-nt3D26lrkho.mp4")

    if not video.isOpened():
        print("No video found")
        sys.exit()

## Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video")
        sys.exit()

    background_subtract(video)

'''
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1366, 768))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''