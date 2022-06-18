import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm

total_rewards = []
cnt=0
threshold=0

class replay_buffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def insert(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 12
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.input_state, 32)
        self.fc2 = nn.Linear(32, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        self.env = env
        self.n_actions = 18
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)
        self.target_net = Net(self.n_actions)

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)

    def learn(self):
        '''
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        '''
        global cnt
        cnt+=1
        #1
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        #2
        observations,actions,rewards,next_observations,done=self.buffer.sample(self.batch_size)
        state=torch.FloatTensor(np.array(observations))
        action=torch.LongTensor(actions)
        reward=torch.FloatTensor(rewards)
        next_state=torch.FloatTensor(np.array(next_observations))
        done=torch.FloatTensor(np.array(done))
        #3
        q_eval=self.evaluate_net(state).gather(1,action.unsqueeze(1))
        q_next=self.target_net(next_state).max(1)[0].detach()
        q_target=(q_next*self.gamma)*(1-done)+reward
        #4
        lossFunc=nn.MSELoss()
        loss=lossFunc(q_eval,q_target.unsqueeze(1))
        #5
        self.optimizer.zero_grad()
        #6
        loss.backward()
        #7
        for para in self.evaluate_net.parameters():
            para.grad.data.clamp_(-1,1)
        self.optimizer.step()

        def test(table):
            env1 = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
            SEED = 20
            env1.seed(SEED)
            env1.action_space.seed(SEED)
            rewards = []
            testing_agent = Agent(env1)
            testing_agent.target_net.load_state_dict(table)
            win_cnt, lose_cnt = 0, 0
            for _ in range(10):
                state = env1.reset()
                state = state[1]
                cnt = 0
                while True:
                    Q = testing_agent.target_net.forward(torch.FloatTensor(state)).squeeze(0).detach()
                    action = int(torch.argmax(Q).numpy())
                    next_state, reward, done, _ = env1.step([0, action])
                    next_state = next_state[1]

                    if reward == 0:
                        cnt += 1
                    if done or cnt > 5000:
                        rewards.append(reward)
                        break
                    
                    state = next_state
            return np.mean(rewards)

        if cnt>2995:
            tmp = test(self.target_net.state_dict())
            global threshold
            if tmp >= threshold:
                threshold = tmp
                torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def choose_action(self, state):
        with torch.no_grad():
            x = torch.unsqueeze(torch.FloatTensor(state), 0)
            
            if random.uniform(0,1)<self.epsilon: 
                action=env.action_space.sample()
            else:
                action_value = self.evaluate_net.forward(x)
                action = torch.max(action_value,1)[1].data.numpy()
                action=[0,action[0]]
        return action

def train(env):
    agent = Agent(env)
    episode = 3000
    rewards = []
    for _ in tqdm(range(episode)):
        state = env.reset()
        state=state[1]
        count = 0
        winCnt=0
        loseCnt=0
        r=0
        while True:
            count += 1
            agent.count += 1
            
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state=next_state[1]

            if reward==0:
                count+=1

            if next_state[0]>0:
                r=(next_state[1]*304/252)**3

            if reward==-1:
                r+=reward
            
            agent.buffer.insert(state, action[1], r,next_state, int(done))
            
            if agent.count >= 1000:
                agent.learn()
            if done or count>500:
                rewards.append(count)
                break
            state = next_state
            
    total_rewards.append(rewards)


def test(env):
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    winCnt=0
    loseCnt=0
    
    for _ in range(100):
        state = env.reset()
        state=state[1]
        cnt = 0
        while True:
            Q = testing_agent.target_net.forward(torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, reward, done, _ = env.step([0,action])
            next_state=next_state[1]
            cnt+=1
            if done:
                if reward==1:
                    winCnt+=1
                    rewards.append(1)
                elif reward==0:
                    loseCnt+=1
                    rewards.append(0)
                else:
                    loseCnt+=1
                    rewards.append(-1)
                break
                
            state = next_state
            
    print(f"average reward: {np.mean(rewards)}")
    print(f"win round: {winCnt}")
    print(f"lose round: {loseCnt}")


def seed(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    SEED = 20

    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)
        
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)
    
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()
