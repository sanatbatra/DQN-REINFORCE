import gym
import time
from policy_model import PolicyModel
from torch.distributions import Categorical
import torch
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt


class ReinforceCartPole:
    def __init__(self, env, gamma):
        self.env = env
        self.env.seed(543)
        torch.manual_seed(543)
        self.policy_model = PolicyModel()
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-2)
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()

    def get_action(self, state):
        state_torch = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_model(state_torch)
        first_action_probability = probs[0][0]
        random_no = random.random()
        if random_no <= first_action_probability:
            action = torch.tensor(0, dtype=torch.long)
        else:
            action = torch.tensor(1, dtype=torch.long)


        m = Categorical(probs)
        print(action)
        log_prob = m.log_prob(action)
        # print(log_prob)
        # policy.saved_log_probs.append(m.log_prob(action))
        return action, log_prob

    def get_returns(self, episode_rewards):
        return_sum = 0.0
        returns = []
        for r in reversed(episode_rewards):
            return_sum = r + self.gamma * return_sum
            returns.append(return_sum)

        returns = torch.tensor(list(reversed(returns)))
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns

    def optimize(self, episode_log_probs, episode_rewards):
        returns = self.get_returns(episode_rewards)
        policy_loss = []

        for log_prob, ret in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * ret)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

    def train(self, no_episodes, limit=4000, rendering=False, max_steps=50000):
        running_reward = 10
        plot_running_rewards = []
        plot_episode_nos = []
        for ep in range(1, no_episodes):
            state = self.env.reset()
            episode_rewards = []
            episode_log_probs = []
            ep_reward = 0
            for s in range(max_steps):

                action, log_prob = self.get_action(state)
                episode_log_probs.append(log_prob)
                state, r, done, _ = self.env.step(action.item())
                if rendering:
                    self.env.render()
                episode_rewards.append(r)
                ep_reward += r
                # print(next_state)
                # print(r)
                # print('\n')
                if done:
                    # time.sleep(0.5)
                    break
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            plot_running_rewards.append(running_reward)
            plot_episode_nos.append(ep)
            self.optimize(episode_log_probs, episode_rewards)
            if ep % 25 == 0:
                print('Episode Number: {}\t Latest Reward: {:.2f}\tAverage Running reward: {:.2f}'.format(
                    ep, ep_reward, running_reward))
            #if running_reward > self.env.spec.reward_threshold:
            if running_reward > limit:

                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, s))
                break

        self.plot(plot_running_rewards, plot_episode_nos)

        self.env.close()

    def plot(self, plot_running_rewards, plot_episode_nos):
        plt.plot(plot_episode_nos, plot_running_rewards)
        plt.ylabel('Running Rewards')
        plt.xlabel('Episode No')
        # plt.legend(['AgentScore'], loc='upper left')
        plt.savefig('RunningRewardsVsEpisodeNo.png')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    gamma = 0.99
    reinforce = ReinforceCartPole(env, gamma)
    limit = 4000
    print('Need to reach:', env.spec.reward_threshold)
    reinforce.train(100000, 4000)

