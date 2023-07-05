import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
#from grobot_utilities.msg import PatrolActionResult 
from std_srvs.srv import Empty
#from PIL import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import torch
import torch.optim as optim
from action import get_action_set, select_exploratory_action, select_greedy_action
from learning import perform_qlearning_step, update_target_net
from model import DQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training

rospy.init_node("test", anonymous=True)
rospy.loginfo("Starting test node")

bridge = CvBridge()
cv_image = None
marker_locations = []
count=0
t=0
reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_world',Empty)
rospy.wait_for_service('/gazebo/reset_world')
reset_simulation_client()
done=False


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

def get_reward(marker_locations,detected):
    if detected==False:
        reward=-100
    else:
        print(marker_locations)
        reward=100-(np.absolute(320-marker_locations[0][1])/320)*100
    return reward

"""
def learn(t):
    #global policy_net,training_losses,episode_rewards
    global count, marker_locations, detected,new_obs,done,cv_image
    if(t>0):
        obs = new_obs
            # Build network
            #obs=get_state(cv_image)
            #print(obs.shape)
        action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        env_action = actions[action_id]
        #for t in range(300):
        #image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)
        marker_locations,detected= detect_aruco_markers(cv_image)
        '''
        if detected:
            count=0
            print("Detected!")
            if(marker_locations[0][1]>240):
                val=-0.2
            else:
                val=0.2
            publisher(0.1,val)
            #print(depth)
        else:
            count+=1
            if count>50:
                publisher(0.1,0.5)
            else:
                publisher(0.0,0.0)
        '''
        
        reward= get_reward(marker_locations,detected)
        for marker_loc in marker_locations:
            marker_id, center_x, center_y = marker_loc
            cv2.circle(cv_image, (center_x, center_y), 10, (0, 255, 0), -1)
            cv2.putText(cv_image, str(marker_id), (center_x-10, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print('TimeStep',t)
        publisher(env_action[0],env_action[1])
        #cv2.imshow('Image2',cv_image)
        new_obs=get_state(cv_image)
        image_filename = str(t)+".jpeg"
        cv2.imwrite('IMAGES/'+image_filename,cv_image)
        print(episode_rewards[-1])
        episode_rewards[-1] = episode_rewards[-1]+reward
        print(reward,"reward")
        print(env_action, episode_rewards[-1],'Hi')
        if episode_rewards[-1]<-5000:
            done=True
        replay_buffer.add(obs, action_id, reward, new_obs, float(done))

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            reset_simulation_client()
            new_obs=get_state(cv_image)
            episode_rewards.append(0.0)
            done=False
        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
            training_losses.append(loss)
        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)
        
        #print(np.all(cv_image==0),"cv_image")
        #print(np.all(new_obs==0), "Okay")

        #print(type(new_obs))
            #publisher(0.0,0.0)
        #image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)#
        #marker_locations,detected= detect_aruco_markers(cv_image)
        #print(marker_locations)
        
        
        #print(detected)
    else:
        new_obs=get_state(cv_image)

"""
# Image callback function for rospy subscriber
def image_callback(img_msg):
    #detected=False
    global cv_image,t,count, marker_locations, detected,new_obs,done
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    
    #cv_image = cv2.resize(cv_image, (96, 96), interpolation=cv2.INTER_AREA)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    t+=1
    print(t)

    if(t>1):
        obs = new_obs
            # Build network
            #obs=get_state(cv_image)
            #print(obs.shape)
        action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        env_action = actions[action_id]
        #for t in range(300):
        #image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)
        marker_locations,detected= detect_aruco_markers(cv_image)
        '''
        if detected:
            count=0
            print("Detected!")
            if(marker_locations[0][1]>240):
                val=-0.2
            else:
                val=0.2
            publisher(0.1,val)
            #print(depth)
        else:
            count+=1
            if count>50:
                publisher(0.1,0.5)
            else:
                publisher(0.0,0.0)
        '''
        
        reward= get_reward(marker_locations,detected)
        for marker_loc in marker_locations:
            marker_id, center_x, center_y = marker_loc
            cv2.circle(cv_image, (center_x, center_y), 10, (0, 255, 0), -1)
            cv2.putText(cv_image, str(marker_id), (center_x-10, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print('TimeStep',t)
        publisher(env_action[0],env_action[1])
        #cv2.imshow('Image2',cv_image)
        new_obs=get_state(cv_image)
        #image_filename = str(t)+".jpeg"
        #cv2.imwrite('IMAGES/'+image_filename,cv_image)
        #print(episode_rewards[-1])
        episode_rewards[-1] = episode_rewards[-1]+reward
        print(reward,"reward")
        print(env_action, episode_rewards[-1],'Hi')
        if episode_rewards[-1]<-5000:
            done=True
        replay_buffer.add(obs, action_id, reward, new_obs, float(done))

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            reset_simulation_client()
            new_obs=get_state(cv_image)
            episode_rewards.append(0.0)
            done=False
        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
            training_losses.append(loss)
        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)
        
        #print(np.all(cv_image==0),"cv_image")
        #print(np.all(new_obs==0), "Okay")

        #print(type(new_obs))
            #publisher(0.0,0.0)
        #image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)#
        #marker_locations,detected= detect_aruco_markers(cv_image)
        #print(marker_locations)
        
        
        #print(detected)
    else:
        new_obs=get_state(cv_image)
    if(t%10000==0):
        torch.save(policy_net.state_dict(), str(t)+'agent.pt')
        visualize_training(episode_rewards, training_losses, str(t)+'agent')
    
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

#def learn():
#    global detected,marker_locations,count,cv_image
#image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)


lr=1e-4
total_timesteps = 100000
buffer_size = 50000
exploration_fraction=0.1
exploration_final_eps=0.02
train_freq=1
action_repeat=4
batch_size=32
learning_starts=1000
#learning_starts=10
gamma=0.99
target_network_update_freq=500
model_identifier='agent'
#global new_obs
#if t>0:
episode_rewards = [0.0]
training_losses = []
actions = get_action_set()
action_size = len(actions)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(action_size, device).to(device)
target_net = DQN(action_size, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

    # Create replay buffer
replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Create the schedule for exploration starting from 1.
exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                            initial_p=1.0,
                            final_p=exploration_final_eps)
t=0
#for t in range(1000):
image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)
    #marker_locations,detected= detect_aruco_markers(cv_image)
cv2.namedWindow("Image Window",1)
#image_sub1= rospy.Subscriber("/limo/depth/image_raw", Image, calc_dist)

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

#learn()
#sent_msg=False
while not rospy.is_shutdown():
    #if not sent_msg:
    #    publisher(0.2,1.5)
    rospy.spin()
