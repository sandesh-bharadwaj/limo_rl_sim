import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import time
import numpy as np
import torch
import torch.optim as optim
from action import get_action_set, select_exploratory_action, select_greedy_action
from learning import perform_qlearning_step, update_target_net
from model import DQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training


import cv2
from cv_bridge import CvBridge, CvBridgeError

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
    rospy.loginfo(cv_image.shape)
    show_image(cv_image)

def twist_callback(msg):
    rospy.loginfo("Received a /cmd_vel message!")
    rospy.loginfo("Linear Components: [%f, %f, %f]"%(msg.linear.x, msg.linear.y, msg.linear.z))
    rospy.loginfo("Angular Components: [%f, %f, %f]"%(msg.angular.x, msg.angular.y, msg.angular.z))

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

def get_reward(cv_image):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(cv_image, arucoDict,parameters=arucoParams)
    centre=[48,48]
    if len(corners)==0:
        reward=-100
    else:
        x1=corners[0][0]
        x2=corners[2][0]#larger
        y1=corners[2][1]
        y2=corners[0][1]#larger
        ad=[(x2-x1)/2,(y2-y1)/2]
        hordist=ad[0]-centre[0]
        reward=100-(np.absolute(hordist)/48)*100
    return reward


def learn(lr=1e-4,
          total_timesteps = 100000,
          buffer_size = 50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          action_repeat=4,
          batch_size=32,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=500,
          model_identifier='agentlast'):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    """
    global cv_image
    episode_rewards = [0.0]
    training_losses = []
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build networks
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

    #reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_world',Empty)

    rospy.wait_for_service('/gazebo/reset_world')
    reset_simulation_client()

    # Initialize environment and get first state
    obs = get_state(cv_image)
    time.sleep(0.1)
    for t in range(total_timesteps):
        #s=cv_image.copy()
        # Select action
        done=False
        action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        env_action = actions[action_id]

        # Perform action frame_skip-times
        #for f in range(action_repeat):
        new_obs=get_state(cv_image)
        rew=get_reward(cv_image)
        episode_rewards[-1] += rew
        # Store transition in the replay buffer.
        #print(obs.shape,"Hi")
        #print(new_obs.shape)
        #new_obs = get_state(new_obs)
        #print(new_obs.shape)
        if(episode_rewards[-1]<-1000):
            done=True
        replay_buffer.add(obs, action_id, rew, new_obs, float(done))

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            

            rospy.wait_for_service('/gazebo/reset_world')
            reset_simulation_client()

            # Initialize environment and get first state
            obs = get_state(cv_image)
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
            training_losses.append(loss)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)

        obs = new_obs

        publisher(env_action[0],env_action[1])

    # Save the trained policy network
    torch.save(policy_net.state_dict(), model_identifier+'.pt')

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier)


if __name__ == '__main__':
    rospy.init_node("test",anonymous=True)
    rospy.loginfo("Starting test node")
    bridge = CvBridge() 
    reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_world',Empty)
    # Subscribe to image stream
    image_sub = rospy.Subscriber("/limo/color/image_raw", Image, image_callback)
    cv2.namedWindow("Image Window",1)
    num_episodes = 1000
    # # Subscribe to cmd_vel stream (to move robot)
    # twist_sub = rospy.Subscriber("/cmd_vel", Twist, twist_callback)
    twist_pub = rospy.Publisher("/cmd_vel",Twist, queue_size=10)
    rospy.on_shutdown(shutdown_hook)
    #sent_msg=False
    while not rospy.is_shutdown():
        #if not sent_msg:
        learn()
        rospy.spin()

    #is_training = True
    

    #x = np.arange(0, len(number_of_steps))
    #plt.plot(x, number_of_steps)
    #plt.show()

