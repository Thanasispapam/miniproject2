#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('tracker_node')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CarTracker():
    #def __init__(self):
    bg_itteration = 0
    bg_alpha = 0
    background = []
    px = py = 32
    diff = 1
    p1 = np.array([[px, py]])
    first_iteration = True
    lk_params = dict( winSize  = (7,7),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    entry_point = np.array([[450, 480]], dtype=np.float32)

    def background_subtraction(self, transformed):
        ########## Background estimation
        if self.bg_itteration == 0:
            self.background = transformed
			
        if self.bg_itteration % 1 == 0:
            self.bg_alpha = self.bg_itteration/(30.0+self.bg_itteration)
            #print(self.bg_itteration)


            if self.bg_alpha < 0.95:
                #print(self.bg_alpha)		
                self.background = cv2.addWeighted(self.background, self.bg_alpha, transformed, 1-self.bg_alpha, 0)
            else:
                #print(0.95)
                self.background = cv2.addWeighted(self.background, 0.95, transformed, 1-0.95, 0)			


        self.bg_itteration = self.bg_itteration + 1


        ########## Background subtraction
        bgg = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        tfg = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

        bgfg_threshold = 15
        temp, threshold = cv2.threshold(cv2.subtract(bgg,tfg), bgfg_threshold, 255, cv2.THRESH_BINARY)
        foreground1 = threshold		

        temp, threshold = cv2.threshold(cv2.subtract(tfg,bgg), bgfg_threshold, 255, cv2.THRESH_BINARY)
        foreground2 = threshold		


        ########## Find green
        ExG = 2.*self.background[:,:,1] - 1.*self.background[:,:,0] - 1.*self.background[:,:,2]
        yikes, ExG_thresholded = cv2.threshold(ExG, 10, 255, cv2.THRESH_BINARY)
        ExG_thresholded = ExG_thresholded.astype(np.uint8)
        ExG_thresholded = 255 - ExG_thresholded


        output = cv2.bitwise_or(foreground1,foreground2)
        output = cv2.bitwise_and(output, ExG_thresholded)
        output = self.four_point_transform(output)


        output = cv2.resize(output, (0,0), fx=0.5, fy=0.5) 


        temp, output = cv2.threshold(output, 90, 255, cv2.THRESH_BINARY)

        # Perform opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output = cv2.erode(output,kernel, iterations = 1)		
        output = cv2.dilate(output,kernel, iterations = 2)
        output = cv2.erode(output,kernel, iterations = 1)

        #return cv2.merge((output,output,output))

        trans_background = self.four_point_transform(bgg)
        trans_background = cv2.resize(trans_background, (0,0), fx=0.5, fy=0.5)

        trans_foreground = self.four_point_transform(tfg)
        trans_foreground = cv2.resize(trans_foreground, (0,0), fx=0.5, fy=0.5)

        return_img = np.concatenate((cv2.merge((output,output,output)), cv2.merge((trans_background,trans_background,trans_background))), axis=1)

        return_img2 = np.concatenate((return_img, cv2.merge((trans_foreground,trans_foreground,trans_foreground))), axis=1)

        return return_img2	

    def four_point_transform(self, image):

        dst = np.array([[449,108],[878,377],[915,717],[384,886]], dtype = "float32")	
        rect2 = np.array([[178,166],[883,125],[1418,236],[837,934]], dtype = "float32")

        #dst = dst * 0.6666
        rect2 = rect2 * 0.6666


        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect2, dst)
        warped = cv2.warpPerspective(image, M, (2000, 2000))

        # return the warped and cropped image
        warped = warped[0:1000, 350:750]
        return warped

    def lucas_kanade(self, image):
        if (self.first_iteration == True):
            self.oldg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.first_iteration = False
            return self.oldg
        else:        
            # Do some Lukas Kanade :::::::::::::::::::::::::::::::::

            newg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Brug det binere billede til at aktivere en Lukas naar omkring 50% af pixels i entry point er hvide
            self.p1, _st, _err = cv2.calcOpticalFlowPyrLK(self.oldg, newg, self.entry_point, None, **self.lk_params)
            self.oldg = newg
            self.entry_point = self.p1

            windowSize = (self.lk_params["winSize"][0] + 1) / 2
            cv2.rectangle(newg, (int(self.entry_point[0][0]) + windowSize, int(self.entry_point[0][1]) + windowSize), 
                                (int(self.entry_point[0][0]) - windowSize, int(self.entry_point[0][1]) - windowSize), 
                                (255, 255, 255), 
                                1)            

            return newg

    def track(self, image): # This class' main function
        image = self.background_subtraction(image)

        image = self.lucas_kanade(image)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image


class sub_pub:
  def __init__(self):
    rospy.init_node('sub_pub', anonymous=True)
    self.image_pub = rospy.Publisher("tracked_image", Image, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("analyzed_image", Image, self.callback)

    self.tracker = CarTracker()

  def callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv_image = self.track_cars(cv_image)
    # self.showImage(cv_image)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)


  def track_cars(self, image):
    image = self.tracker.track(image)
    return image


  def showImage(self, image):
    cv2.imshow("Image window", image)
    cv2.waitKey(3)


def main(args):
  tr = sub_pub()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Launching the tracker")
    main(sys.argv)
