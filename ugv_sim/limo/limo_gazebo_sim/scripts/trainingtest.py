import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
#from grobot_utilities.msg import PatrolActionResult 
from std_srvs.srv import Empty

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

rospy.init_node("test", anonymous=True)
rospy.loginfo("Starting test node")

bridge = CvBridge()
cv_image = None
marker_locations = []
reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_world',Empty)
rospy.wait_for_service('/gazebo/reset_world')
reset_simulation_client()
count=0

def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

def detect_aruco_markers(img):
    detected=False
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    locations = []
    if ids is not None:
        detected=True
        for i in range(len(ids)):
            marker_id = ids[i][0]
            corners_i = corners[i][0]
            center_x = int(np.mean(corners_i[:, 0]))
            center_y = int(np.mean(corners_i[:, 1]))
            locations.append((marker_id, center_x, center_y))
    
    return locations,detected

twist_pub = rospy.Publisher("/cmd_vel",Twist, queue_size=10)


def publisher(x_linear, z_angular):
    twist_msg = Twist()
    twist_msg.linear.x = x_linear
    twist_msg.linear.y = 0
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = z_angular
    twist_pub.publish(twist_msg)


# Image callback function for rospy subscriber
def image_callback(img_msg):
    #detected=False
    global count
    global cv_image, marker_locations
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    
    #cv_image = cv2.resize(cv_image, (96, 96), interpolation=cv2.INTER_AREA)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    print(cv_image.shape)
    print(marker_locations)
    marker_locations,detected= detect_aruco_markers(cv_image)
    #print(detected)

    if detected:
        count=0
        print("Detected!")
        if(marker_locations[0][1]>240):
            val=-0.2
        else:
            val=0.2
        publisher(0.1,val)
        print(depth)
    else:
        count+=1
        if count>10:
            publisher(0.1,0.5)
        else:
            publisher(0.0,0.0)
        #publisher(0.0,0.0)
    print(marker_locations)
    print(count)
    for marker_loc in marker_locations:
        marker_id, center_x, center_y = marker_loc
        cv2.circle(cv_image, (center_x, center_y), 10, (0, 255, 0), -1)
        cv2.putText(cv_image, str(marker_id), (center_x-10, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    show_image(cv_image)

def calc_dist(img):
    global depth
    try:
        depth_image = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    if(len(marker_locations)>0):
        depth=depth_image[marker_locations[0][1]][marker_locations[0][2]]
    else:
        depth=0
    #return img

def get_marker_locations():
    print(marker_locations)
    return marker_locations

def twist_callback(msg):
    rospy.loginfo("Received a /cmd_vel message!")
    rospy.loginfo("Linear Components: [%f, %f, %f]" % (msg.linear.x, msg.linear.y, msg.linear.z))
    rospy.loginfo("Angular Components: [%f, %f, %f]" % (msg.angular.x, msg.angular.y, msg.angular.z))

def patrol_result_callback(result_msg):
    rospy.loginfo(result_msg)

image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)
image_sub1= rospy.Subscriber("/limo/depth/image_raw", Image, calc_dist)
cv2.namedWindow("Image Window",1)
#patrol_result_sub = rospy.Subscriber("/karl/patrol_server/result", PatrolActionResult, patrol_result_callback)
#global cv_image
#cv2.imshow("Image Window 2", cv_image)
#cv2.namedWindow("Image Window 2", 1)

# # Subscribe to cmd_vel stream (to move robot)
# twist_sub = rospy.Subscri#ber("/cmd_vel", Twist, twist_callback)
#param_name = rospy.get_param('/karl')
#print(param_name)
#robot_x = rospy.get_param('/karl/x')
#robot_y = rospy.get_param('/karl/y')
##print("Robot Coordinates:")
#print("x:", robot_x)
#print("y:", robot_y)


def shutdown_hook():
    publisher(0.0,0.0)

rospy.on_shutdown(shutdown_hook)

#sent_msg=False
while not rospy.is_shutdown():
    #if not sent_msg:
    #    publisher(0.2,1.5)
    rospy.spin()
