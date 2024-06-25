import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import pandas as pd

plt.ion()

def plot(scores, mean_scores, rewards, ngames):
    if ngames!=0:
        plt.close('all')
    fig, axs = plt.subplots(2)

    axs[0].clear()
    axs[0].set_title('Score')
    axs[0].set_ylabel('Score')
    axs[0].plot(scores, color='blue')
    axs[0].plot(mean_scores, color='red')
    axs[0].set_ylim(ymin=0)
    axs[0].legend(['Score', 'Mean Score'])
    
    # Second plot
    axs[1].clear()
    axs[1].set_title('Rewards')
    axs[1].set_xlabel('Number of Games')
    axs[1].set_ylabel('Reward')
    axs[1].plot(rewards, color='darkblue')
    axs[1].legend(['Reward'])
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)
    
    if ngames == 100:
        plt.savefig('score.png')
        df = pd.DataFrame({'score': scores, 'mean_score': mean_scores, 'rewards': rewards})
        df.to_csv('score.csv', index=False)

# Plotting the loss
def plot_stats(loss):
    plt.plot(range(len(loss)),loss)
    plt.title('Training...')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.savefig('loss.png')