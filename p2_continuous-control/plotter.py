import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, scores, target_score, moving_avg, num_agents):
        self.scores = scores
        self.target_score = target_score
        self.moving_avg = moving_avg
        self.num_agents = num_agents

    def plot(self):
        """
        Plots training results.
        """
        moving_avg_mean, target = self.plot_agent()
        self.plot_agents(moving_avg_mean, target)

    def plot_agent(self):
        scores_mean = np.mean(self.scores, axis=1)
        moving_avg_mean = np.mean(self.moving_avg, axis=1)
        target = [self.target_score] * len(self.scores)  # Trace a line indicating the target value
        # Plot the main graph with the scores and moving average
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle('Plot of the rewards', fontsize='xx-large')
        ax = fig.add_subplot(111)
        ax.plot(scores_mean, label='Score', color='Blue')
        ax.plot(moving_avg_mean, label='Moving Average',
                color='LightGreen', linewidth=3)
        ax.plot(target, linestyle='--', color='LightCoral', linewidth=1)
        ax.text(0, self.target_score + 1, 'Target', color='LightCoral', fontsize='large')
        ax.set_ylabel('Score')
        ax.set_xlabel('Episode #')
        ax.legend(fontsize='xx-large', loc='lower right')
        plt.show()
        return moving_avg_mean, target

    def plot_agents(self, moving_avg_mean, target):
        if len(self.scores[0]) == self.num_agents:
            fig, axs = plt.subplots(5, 4, figsize=(15, 20), constrained_layout=True, sharex=True, sharey=True)
            fig.suptitle('Rewards for each one of the agents', fontsize='xx-large')

            axs = axs.flatten()
            for idx, (ax, s, m) in enumerate(zip(axs, np.transpose(self.scores), np.transpose(self.moving_avg))):
                ax.plot(s, label='Agent Score', color='DodgerBlue', zorder=2)
                ax.plot(m, label='Agent Moving Avg', color='DarkOrange', zorder=3)
                ax.plot(moving_avg_mean, label='Moving Avg (Total)',
                        color='LightGreen', linewidth=3, alpha=0.655, zorder=1)
                ax.plot(target, linestyle='--', color='LightCoral', linewidth=1, zorder=0)
                ax.text(0, self.target_score + 1, 'Target', color='LightCoral', fontsize='large')

                ax.set_title('Agent #%d' % (idx + 1))
                ax.set_ylabel('Score')
                ax.set_xlabel('Episode #')
                ax.label_outer()
                ax.legend(fontsize='medium')

            plt.show()

