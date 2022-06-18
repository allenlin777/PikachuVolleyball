import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('pikaball')
    plt.xlabel('epoch')
    plt.ylabel('rewards')

def Q_Learning():
    Q_learning_Rewards = np.load("./Rewards/pikaball_rewards.npy").transpose()
    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    Q_learning_std = np.std(Q_learning_Rewards, axis=1)
    initialize_plot()
    plt.plot([i for i in range(3000)], Q_learning_avg,
             label='Q_Learning', color='cornflowerblue')
    plt.fill_between([i for i in range(3000)],
                     Q_learning_avg+Q_learning_std, Q_learning_avg-Q_learning_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/Q_Learning.png")
    plt.show()
    plt.close()

def DQN():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy").transpose()
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)
    initialize_plot()

    plt.plot([i for i in range(3000)], DQN_avg,
             label='DQN', color='salmon')
    plt.fill_between([i for i in range(3000)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN.png")
    plt.show()
    plt.close()

def compare():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy").transpose()
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    Q_learning_Rewards = np.load("./Rewards/pikaball_rewards.npy").transpose()
    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    initialize_plot()
    plt.plot([i for i in range(3000)], DQN_avg, label='DQN', color='salmon')
    plt.plot([i for i in range(3000)],
             Q_learning_avg[:3000], label='Q_learning', color='cornflowerblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/compare.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--Q_Learning", action="store_true")
    parser.add_argument("--DQN", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

        
    if not os.path.exists("./Graphs"):
        os.mkdir("./Graphs")

    if args.Q_Learning:
        Q_Learning()
    elif args.DQN:
        DQN()
    elif args.compare:
        compare()
