import matplotlib.pyplot as plt
import numpy as np
from IPython import display

plt.ion()

def plot(scores, mean_scores, rewards, ngames):
    if ngames!=2:
        plt.close()
    fig, axs = plt.subplots(2)
    
    axs[0].clear()
    axs[0].set_title('Training...')
    axs[0].set_xlabel('Number of Games')
    axs[0].set_ylabel('Score')
    axs[0].plot(scores)
    axs[0].plot(mean_scores)
    axs[0].set_ylim(ymin=0)
    axs[0].legend(['Score', 'Mean Score'])
    
    # Second plot
    axs[1].clear()
    axs[1].set_title('Rewards...')
    axs[1].set_xlabel('Number of Games')
    axs[1].set_ylabel('Reward')
    axs[1].plot(rewards)
    axs[1].set_ylim(ymin=0)
    axs[1].legend(['Reward'])
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)
    
    #if ngames == 100:
    #    plt.savefig('score.png')

def plot_stats(loss):
    plt.plot(range(len(loss)),loss)
    plt.title('Training...')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.savefig('loss2.png')