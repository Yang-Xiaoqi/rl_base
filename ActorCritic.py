import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



#Actor Network
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet,self).__init__()
        self.fc1 = nn.Linear(n_states,n_hiddens)
        self.fc2 = nn.Linear(n_hiddens,n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return F.softmax(x, dim=1)


#Critic Network
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hidden):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        return x


# Actor-Critic Network

class ActorCritic:
    def __init__(self,n_states, n_hidden, n_actions, actor_lr, critic_lr, gamma):
        self.gamma = gamma

        self.actor = PolicyNet(n_states, n_hidden, n_actions)
        self.critic = ValueNet(n_states, n_hidden)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

    def take_action(self, state):
        state= torch.tensor(state).unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        td_value = self.critic(states)
        td_target = rewards +self.gamma*self.critic(next_states)*(1-dones)
        td_delta = td_target-td_value

        log_probs = torch.log(self.actor(states).gather(1,actions))

        actor_loss = torch.mean(-log_probs*td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(td_value, td_target))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()






