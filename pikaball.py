import numpy as np
import gym
import os
from tqdm import tqdm
import random

total_reward = []
episode = 3000
decay = 0.045

doneCount=0

class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.5, GAMMA=0.97, num_bins=3):
        self.env = env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.num_bins = num_bins
        self.qtable = np.zeros((self.num_bins, self.num_bins, self.num_bins, self.num_bins,
                                self.num_bins, self.num_bins, self.num_bins, self.num_bins,
                                self.num_bins, self.num_bins, self.num_bins, self.num_bins,
                                18))

        self.bins = [
            self.init_bins(-1, 1, self.num_bins),  # ball.x
            self.init_bins( 0, 1, self.num_bins),  # ball.y
            self.init_bins(-1, 1, self.num_bins),  # ball.xVelocity
            self.init_bins(-1, 1, self.num_bins),  # ball.yVelocity
            self.init_bins(-1, 0, self.num_bins),  # player1.x
            self.init_bins( 0, 1, self.num_bins),  # player1.y
            self.init_bins(-1, 1, self.num_bins),  # player1.xVelocity
            self.init_bins(-1, 1, self.num_bins),  # player1.yVelocity
            self.init_bins( 0, 1, self.num_bins),  # player2.x
            self.init_bins( 0, 1, self.num_bins),  # player2.y
            self.init_bins(-1, 1, self.num_bins),  # player2.xVelocity
            self.init_bins(-1, 1, self.num_bins),  # player2.yVelocity
        ]

    def init_bins(self, lower_bound, upper_bound, num_bins):
        interval=np.linspace(lower_bound,upper_bound,num_bins+1)
        array=interval[1:-1]
        return array

    def discretize_value(self, value, bins):
        return np.digitize(value,bins)

    def discretize_observation(self, observation):
        state=[]
        for i in range (len(observation[1])):
            state.append(Agent.discretize_value(self,observation[1][i],self.bins[i]))
        return tuple(state)

    def choose_action(self, state):
        if random.uniform(0,1)<self.epsilon:
            action=env.action_space.sample()
        else:
            actionIdx=np.argmax(self.qtable[state])
            action=[random.randrange(0,18),actionIdx]

        return action
            
        

    def learn(self, state, action, reward, next_state, done):
        Qopt=self.qtable[state][action[1]]
        global doneCount
        if done==True:
            Vopt=0
            doneCount+=1
        else:
            Vopt=np.max(self.qtable[next_state])

        self.qtable[state][action[1]]=(1-self.learning_rate)*Qopt+self.learning_rate*(reward+self.gamma*Vopt)
        
        if(doneCount>episode-5 and done==True):
            np.save("./Tables/pikaball_table.npy", self.qtable)

def train(env):
    training_agent = Agent(env)
    rewards = []
    for ep in tqdm(range(episode)):
        state = training_agent.discretize_observation(env.reset())
        done = False

        count = 0
        while True:
            count += 1
            action = training_agent.choose_action(state)

            r=0
            
            next_observation, reward, done, _ = env.step(action)

            if(next_observation[1][0]>0):
                r=(next_observation[1][1]*304/252)**3

            reward=reward*10+r
            
            next_state = training_agent.discretize_observation(
                next_observation)

            training_agent.learn(state, action, reward, next_state, done)

            if done:
                rewards.append(count)
                break

            state = next_state


        if (ep + 1) % 500 == 0:
            training_agent.learning_rate -= decay

    total_reward.append(rewards)


def test(env):
    testing_agent = Agent(env)

    testing_agent.qtable = np.load("./Tables/pikaball_table.npy")
    rewards = []
    winCnt=0

    for _ in range(100):
        state = testing_agent.discretize_observation(testing_agent.env.reset())
        count = 0
        while True:
            count += 1
            action = [random.randrange(0,18),np.argmax(testing_agent.qtable[tuple(state)])]
            next_observation, reward, done, _ = testing_agent.env.step(action)

            next_state = testing_agent.discretize_observation(next_observation)

            if done == True:
                if(reward==1):
                    winCnt+=1
                    rewards.append(1)
                elif(reward==0):
                    rewards.append(0)
                else:
                    rewards.append(-1)
                break

            state = next_state

    print(f"average reward: {np.mean(rewards)}")
    print(f"Win round: {winCnt}")

def seed(seed=20):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    SEED = 20

    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',isPlayer1Computer=True,isPlayer2Computer=False)
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env)
        
    test(env)

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/pikaball_rewards.npy", np.array(total_reward))

    env.close()
