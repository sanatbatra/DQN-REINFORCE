import gym
import torch
import math
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.optim as optim
import operator
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# from pyvirtualdisplay import Display
#
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()
# print('Device: ', device)


class DeepQNetwork:
    def __init__(self, env, batch_size, start_epsilon, end_epsilon, gamma, replay_size, policy_update_frequency,
                 target_update_freq, model_save_frequency, lr, no_of_actions, history_length, polyak_factor, double_dqn,
                 prioritized_replay):
        self.env = env
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.gamma = gamma
        self.policy = Model(history_length).to(device)
        self.target = Model(history_length).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.replay_size = replay_size
        self.no_of_steps_taken = 0
        self.replay_memory = []
        self.policy_update_freq = policy_update_frequency
        self.target_update_freq = target_update_freq
        self.model_save_frequency = model_save_frequency
        # self.optimizer = optim.RMSprop(self.policy.parameters())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.no_of_actions = no_of_actions
        # self.action_list = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        #                     np.array([0.0, 0.0, 0.2]), np.array([-1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]),
        #                     np.array([-1.0, 0.0, 0.2]), np.array([1.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.0])]
        self.action_list = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                            np.array([0.0, 0.0, 0.2]), np.array([-1.0, 0.0, 0.2]), np.array([1.0, 0.0, 0.2]),
                            np.array([0.0, 0.0, 0.0])]
        # self.action_list = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        #                     np.array([0.0, 0.0, 0.2]), np.array([-1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]),
        #                     np.array([-1.0, 0.0, 0.2]), np.array([1.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.0])]
        # self.action_list = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        #                     np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.0])]
        self.loss_func = nn.MSELoss()
        # self.loss_func =
        self.replay_pos = 0
        self.possible_action_nos = [i for i in range(len(self.action_list))]
        self.history_length = history_length
        self.polyak_factor = polyak_factor
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.Experience = namedtuple('Experience', ('s', 'a', 's1', 'r', 'p'))

    def add_to_replay(self, state, action_no, next_state, reward, priority):
        if len(self.replay_memory) < self.replay_size:
            self.replay_memory.append(None)

        self.replay_memory[self.replay_pos] = (state, action_no, next_state, reward, priority)
        self.replay_pos = (self.replay_pos + 1) % self.replay_size

    def train(self, no_of_episodes):
        scores = []
        ma_episodes = []
        mas = []
        for ep in range(no_of_episodes):
            # self.epsilon = self.start_epsilon - (ep*(self.start_epsilon - self.end_epsilon)/float(0.4*no_of_episodes))
            if ep > 400:
                self.epsilon = 0.04
            else:
                self.epsilon = self.end_epsilon + (
                            (self.start_epsilon - self.end_epsilon) / math.exp(self.no_of_steps_taken / 100000.0))

                # if ep+1 % self.target_update_freq:
            #     self.target.load_state_dict(self.policy.state_dict())
            #     for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
            # # target_param.data.copy_(target_param.data + (self.polyak_factor * (policy_param.data - target_param.data)))
            #         target_param.data.copy_((self.polyak_factor*policy_param.data) + (target_param.data*(1.0-self.polyak_factor)))

            ep_score = self.run_episode()
            print('Episode No: ', ep + 1)
            print('Score: ', ep_score)
            print('\n')
            scores.append(ep_score)
            torch.save(self.policy.state_dict(), './policy1.pth')
            if len(scores) >= 100:
                ma = float(sum(scores[-100:])) / 100.0
                mas.append(ma)
                ma_episodes.append(len(scores))
            plt.clf()
            plt.plot([ep + 1 for ep in range(len(scores))], scores)
            if len(scores) >= 100:
                plt.plot(ma_episodes, mas)
                plt.legend(['Score', 'Moving Average'], loc='upper left')
            else:
                plt.legend(['Score'], loc='upper left')
            plt.ylabel('Agent Score')
            plt.xlabel('Episode Number')
            # plt.legend(['AgentScore'], loc='upper left')
            plt.savefig('ScoreVsEpisodeNo1.png')
            plt.savefig('./drive/My Drive/ScoreVsEpisodeNo1.png')

    def get_action(self, state_list):

        if self.no_of_steps_taken % 1000 == 0:
            print('Epsilon:', self.epsilon)
        random_no = random.random()
        if random_no <= self.epsilon:
            # a = random.randint(0, self.no_of_actions-1)
            a = random.choices(self.possible_action_nos, weights=[1, 1, 4, 1, 1, 1, 1], k=1)[0]

        else:
            self.policy.eval()
            with torch.no_grad():
                a = self.policy(state_list).max(1)[1].item()
            self.policy.train()
        return self.action_list[a], a

    def run_policy_optimization(self):
        if len(self.replay_memory) < self.batch_size:
            return
        # self.policy.train()

        if self.prioritized_replay:
            priority_list = list(map(operator.itemgetter(4), self.replay_memory))
            # print(priority_list)
            batch = self.Experience(*zip(*random.choices(self.replay_memory, weights=priority_list, k=self.batch_size)))
        else:
            batch = self.Experience(*zip(*random.sample(self.replay_memory, self.batch_size)))

        states = torch.cat(batch.s)
        actions = torch.cat(batch.a)
        rewards = torch.cat(batch.r)
        # print(actions)

        if self.double_dqn:
            next_state_list = [a for a in batch.s1 if a is not None]
            with torch.no_grad():

                next_state_policy_max_actions = self.policy(torch.cat(next_state_list).to(device)).max(1)[1].detach()
            # print(next_state_policy_max_actions)
            # print(next_state_policy_max_actions.reshape(self.batch_size, 1))

            next_state_target_q_values = self.target(torch.cat(next_state_list).to(device)).detach()
            next_state_target_max_q_values = next_state_target_q_values.gather(1, next_state_policy_max_actions.reshape(
                len(next_state_list), 1)).to(device).detach().reshape(1, len(next_state_list))
            # print(next_state_target_q_values)
            # print(next_state_target_max_q_values)

        else:
            with torch.no_grad():
                next_state_target_max_q_values = \
                self.target(torch.cat([a for a in batch.s1 if a is not None]).to(device)).max(1)[0].detach()

        q_values = self.policy(states).gather(1, actions)
        target_q_values = torch.zeros(self.batch_size).to(device)
        indices = torch.tensor(tuple(map(lambda s: s is not None, batch.s1)), device=device, dtype=torch.bool)
        target_q_values[indices] = next_state_target_max_q_values * self.gamma

        target_q_values = target_q_values + rewards
        # loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
        loss = self.loss_func(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
            # target_param.data.copy_(target_param.data + (self.polyak_factor * (policy_param.data - target_param.data)))
            target_param.data.copy_(
                self.polyak_factor * policy_param.data + target_param.data * (1.0 - self.polyak_factor))

    def run_episode(self, rendering=True, max_timesteps=1000):
        episode_reward = 0.0
        step = 0
        state = self.env.reset()
        state = rgb2gray(state)
        no_neg = 0
        prev_neg = True
        state_list = [state for _ in range(history_length)]
        state_list_list = torch.from_numpy(np.array(state_list, dtype=np.float32)).to(device,
                                                                                      dtype=torch.float).reshape(-1,
                                                                                                                 history_length,
                                                                                                                 96, 96)
        for t in range(max_timesteps):
            action, action_no = self.get_action(state_list_list)
            next_state, r, done, _ = self.env.step(action)

            episode_reward += r
            state_list.pop(0)
            state_list.append(rgb2gray(next_state))
            new_state_list = torch.from_numpy(np.array(state_list, dtype=np.float32)).to(device,
                                                                                         dtype=torch.float).reshape(-1,
                                                                                                                    history_length,
                                                                                                                    96,
                                                                                                                    96)
            step += 1

            if r > 0.0:
                prev_neg = False
                no_neg = 0
                priority = 3
            else:
                no_neg += 1
                priority = 1
                prev_neg = True
                if no_neg > 250:
                    done = True
                    r = -30.0

            if done:
                self.add_to_replay(state_list_list, torch.tensor([[action_no]], device=device, dtype=torch.long), None,
                                   torch.tensor([r], device=device, dtype=torch.float32), priority)
            else:
                self.add_to_replay(state_list_list, torch.tensor([[action_no]], device=device, dtype=torch.long),
                                   new_state_list,
                                   torch.tensor([r], device=device, dtype=torch.float32), priority)

            state_list_list = new_state_list

            if step % 500 == 0:
                print('Step No:', step)

            self.no_of_steps_taken += 1
            if self.no_of_steps_taken % self.policy_update_freq == 0:
                self.run_policy_optimization()

            if self.no_of_steps_taken % self.model_save_frequency == 0:
                torch.save(self.policy.state_dict(), './drive/My Drive/policy1.pth')

            if rendering:
                self.env.render()
            if done:
                break

        self.env.close()

        return episode_reward


if __name__ == '__main__':
    env = gym.make('CarRacing-v0').unwrapped
    start_epsilon = 0.95
    end_epsilon = 0.05
    gamma = 0.9
    replay_size = 100000
    batch_size = 64
    policy_update_frequency = 3  # steps
    target_update_frequency = 10  # episodes
    model_save_frequency = 1000
    lr = 0.0004
    no_of_actions = 7
    history_length = 4
    # polyak_factor = 0.999
    polyak_factor = 0.001
    double_dqn = False
    prioritized_replay = True
    # max_consec_neg =

    agent = DeepQNetwork(env, batch_size, start_epsilon, end_epsilon, gamma, replay_size, policy_update_frequency,
                         target_update_frequency, model_save_frequency, lr, no_of_actions,
                         history_length, polyak_factor, double_dqn, prioritized_replay)
    agent.train(900)