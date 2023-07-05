import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node("test",anonymous=True)
rospy.loginfo("Starting test node")

bridge = CvBridge()
cv_image=None
#global cv_image

def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

# Image callback function for rospy subscriber
def image_callback(img_msg):
    # rospy.loginfo(img_msg.header)
    # cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    global cv_image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    cv_image=cv2.resize(cv_image,(96,96), interpolation = cv2.INTER_AREA)
    cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
    #return cv_image
    show_image(cv_image)

def twist_callback(msg):
    rospy.loginfo("Received a /cmd_vel message!")
    rospy.loginfo("Linear Components: [%f, %f, %f]"%(msg.linear.x, msg.linear.y, msg.linear.z))
    rospy.loginfo("Angular Components: [%f, %f, %f]"%(msg.angular.x, msg.angular.y, msg.angular.z))

# Subscribe to image stream
image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)
cv2.namedWindow("Image Window",1)
#global cv_image
#cv2.imshow("Image Window 2", cv_image)
#cv2.namedWindow("Image Window 2", 1)

# # Subscribe to cmd_vel stream (to move robot)
# twist_sub = rospy.Subscri#ber("/cmd_vel", Twist, twist_callback)
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

def shutdown_hook():
    publisher(0.0,0.0)

rospy.on_shutdown(shutdown_hook)

sent_msg=False
while not rospy.is_shutdown():
    if not sent_msg:
        publisher(0.2,1.5)
    rospy.spin()
