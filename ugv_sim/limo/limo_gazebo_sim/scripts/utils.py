import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_state(state): 
    """ Helper function to transform state """ 
    state=cv2.resize(state, (96, 96), interpolation=cv2.INTER_AREA)
    state = np.ascontiguousarray(state, dtype=np.float32) 
    return np.expand_dims(state, axis=0)

def visualize_training(episode_rewards, training_losses, model_identifier):
    """ Visualize training by creating reward + loss plots
    Parameters
    -------
    episode_rewards: list
        list of cumulative rewards per training episode
    training_losses: list
        list of training losses
    model_identifier: string
        identifier of the agent
    """
    plt.plot(np.array(episode_rewards))
    plt.savefig("episode_rewards-"+model_identifier+".png")
    plt.close()
    plt.plot(np.array(training_losses))
    plt.savefig("training_losses-"+model_identifier+".png")
    plt.close()

