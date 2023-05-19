# limo_rl_sim

## Full Instructions
Do all of this outside of a conda environment (Currently ROS only works with Python 2)


To start using roslaunch (Always do this when in a new terminal, before using any ROS command):
```
cd limo_grobot_ws
source devel/setup.bash
```


To launch visual servoing (Limo + Target Robot):
```
roslaunch limo_gazebo_sim limo_ackerman.launch
```


To run publisher/subscriber script:
```
python src/limo_rl_sim/ugv_sim/limo/limo_gazebo_sim/scripts/training.py
```

To launch visual servoing (Two TurtleBots):
```
roslaunch grobot_bringup visual_servoing.launch
```

To have manual control of Limo (requires limo_ackerman to have been launched):
```
rosrun rqt_robot_steering rqt_robot_steering
```


# For modifying files (Careful with this)
```
ros<launch,run,etc.> <pkg_name> <file_name>
```
The <pkg_name> refers to a folder that exists inside src of the workspace (limo_grobot_ws). <file_name> exists inside one of the folders within the package. usually scripts folder contains python/bash scripts, src contains cpp/python files, launch folders contains launch files. Search using File Explorer (VSCode should help)