import rospy
import tf

rospy.init_node('get_robot_coordinates')  # Initialize the ROS node
listener = tf.TransformListener()  # Create a TransformListener

# Wait for the transform between the 'map' frame and 'karl/base_footprint' frame to become available
listener.waitForTransform('map', 'karl/base_footprint', rospy.Time(), rospy.Duration(4.0))

try:
    # Get the current transformation between the 'map' frame and 'karl/base_footprint' frame
    (trans, rot) = listener.lookupTransform('map', 'karl/base_footprint', rospy.Time(0))
    x, y, z = trans  # Extract the x, y, and z coordinates

    print("Robot Coordinates:")
    print("x: ", x)
    print("y: ", y)
    print("z: ", z)

except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    rospy.logerr("Failed to get robot coordinates.")

rospy.spin()  # Keep the Python script running until it is terminated