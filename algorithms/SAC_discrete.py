# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Fictitious Self-Play (NFSP) agent implemented in PyTorch.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import enum
from inspect import Traceback
import os
import random
import traceback
from absl import logging
import numpy as np

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os

from open_spiel.python import rl_agent
from open_spiel.python.pytorch import dqn
from torch.distributions import Categorical

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask next_legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

# Value Net
class ValueNet(nn.Module):
  def __init__(self, state_dim, layer_sizes, action_dim):
    super(ValueNet, self).__init__()
    self.nn = MLP(state_dim, layer_sizes, 1)
        
  def forward(self, state):
    x = self.nn(state)
    return x

# Soft Q Net
class SoftQNet(nn.Module):
  def __init__(self, state_dim, layer_sizes, action_dim):
    super(SoftQNet, self).__init__()
    self.nn = MLP(state_dim, layer_sizes, action_dim)
      
  def forward(self, state):
    x = self.nn(state)
    return x


class PolicyNetwork(nn.Module):
  def __init__(self, state_dim, layer_sizes, action_dim):
    super(PolicyNetwork, self).__init__()
    self._action_dim = action_dim
    self.nn = MLP(state_dim, layer_sizes, action_dim)
      
  def forward(self, state):
    x = self.nn(state)
    #x = torch.softmax(x, dim=1)
    return x
  
  def evaluate(self, state, illegal_actions):
    try:
      legal_actions = 1.0 - illegal_actions
      action_probs_ = F.softmax(self.nn(state), dim=1)
      action_probs = (action_probs_+1e-10)*legal_actions.detach()
      action_probs_sum = torch.sum(action_probs, dim=1)
      action_probs_sum = action_probs_sum + (action_probs_sum == 0.0).float() * 1e-8
      action_probs = action_probs/action_probs_sum.unsqueeze(1)
      action_probs = action_probs*legal_actions.detach()*legal_actions.detach()
    except:
      traceback.print_exc()

    #sample actions
    #action_dist = Categorical(action_probs)
    #actions = action_dist.sample().view(-1, 1)
      
    # Avoid numerical instability.
    #action_probs = torch.gather(action_probs, 1, actions.view(-1,1))
    z = (action_probs == 0.0).float() * 1e-8
    log_action_probs = torch.log(action_probs + z)

    return None, action_probs, log_action_probs
       
  def get_action(self, state):
    action_probs = F.softmax(self.nn(state), dim=1).detach()
    #print(action_probs)
    return action_probs

def disable_gradients(network):
  # Disable calculations of gradients.
  for param in network.parameters():
      param.requires_grad = False

from open_spiel.python.pytorch import dqn
from open_spiel.python.pytorch.dqn import MLP, ReplayBuffer
class SACDiscrete(rl_agent.AbstractAgent):
  """SAC Agent implementation in PyTorch.

  """

  def __init__(self,
               player_id,
               state_representation_size,
               num_actions,
               device = 'cpu',
               gamma=0.99, 
               tau=0.01,
               alpha = 1.0,
               hidden_layers_sizes=128,
               replay_buffer_capacity=10000,
               batch_size=[64, 16],
               replay_buffer_class=ReplayBuffer,
               q_lr=3e-3,
               policy_lr=3e-3,
               update_target_network_every=500,
               learn_every=10,
               discount_factor=1.0,
               min_buffer_size_to_learn=1000,
               double_DQN = True,
               optimizer_str="sgd"):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    if not device == "cpu":
      self.device = 'cuda:' + str(device)
    else:
      self.device = 'cpu'
    self._kwargs = locals()
    self._double_DQN = double_DQN

    self.player_id = player_id
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor

    # TODO(author6) Allow for optional replay buffer config.
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None
    
    # hyperparameters
    self.alpha = alpha
    self.gamma = gamma
    self.tau = tau

    # initialize networks parameters
    self.state_dim = state_representation_size
    self.action_dim = num_actions

    # Create the network instances
    self.q1_net = SoftQNet(self.state_dim, self._layer_sizes, self.action_dim).to(torch.device(self.device))
    self.q2_net = SoftQNet(self.state_dim, self._layer_sizes, self.action_dim).to(torch.device(self.device))
    self.q1_target_net = SoftQNet(self.state_dim, self._layer_sizes, self.action_dim).to(torch.device(self.device))
    self.q2_target_net = SoftQNet(self.state_dim, self._layer_sizes, self.action_dim).to(torch.device(self.device))
    self.policy_net = PolicyNetwork(self.state_dim, self._layer_sizes, self.action_dim).to(torch.device(self.device))

    self.copy_model_over(self.q1_net, self.q1_target_net)
    self.copy_model_over(self.q2_net, self.q2_target_net)
    
    if optimizer_str == "adam":
      self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
      self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
      self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    elif optimizer_str == "sgd":
      self.q1_optimizer = optim.SGD(self.q1_net.parameters(), lr=q_lr)
      self.q2_optimizer = optim.SGD(self.q2_net.parameters(), lr=q_lr)
      self.policy_optimizer = optim.SGD(self.policy_net.parameters(), lr=policy_lr)  
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")
    

    self._q1_losses = []
    self._q2_losses = []
    self._policy_losses = []
    self._losses = []
    self._step_counters = []

  def save_loss(self, q1_value_loss, q2_value_loss, policy_loss):
    self._step_counters.append(self._step_counter)
    self._losses.append([q1_value_loss.item(), q2_value_loss.item(), policy_loss.item()])
    #self._q1_losses.append(q1_value_loss.item())
    #self._q2_losses.append(q2_value_loss.item())
    #self._policy_losses.append(policy_loss.item())
  
  def plot_loss(self):
    import pandas
    import seaborn as sns
    import matplotlib.pyplot as plt
    try:
      data = pandas.DataFrame(self._losses, self._step_counters, columns=['q1', 'q2', 'policy'])
      #data.to_csv('data/CFRs_{}_{}_Result.csv'.format(FLAGS.game, FLAGS.iterations))

      f, ax = plt.subplots(figsize=(7, 5))
      fig = sns.lineplot(data=data, palette="tab10")

      import matplotlib.pyplot as plt
      plt.savefig('loss.png', encoding='utf8')
      plt.show()
    except:
      traceback.print_exc()

  def copy_model_over(self, from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())

  def target_update(self):
    for target_param, param in zip(self.q1_target_net.parameters(), self.q1_net.parameters()):
      target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    for target_param, param in zip(self.q2_target_net.parameters(), self.q2_net.parameters()):
      target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

  def _get_action(self, info_state, legal_actions, greedy=True):
    info_state = torch.Tensor(np.reshape(info_state, [1, -1])).to(torch.device(self.device))
    with torch.no_grad():
      action_probs = self.policy_net.get_action(info_state)[0].cpu()

    try:
      if greedy:
        probs = np.zeros(self._num_actions)
        legal_probs = action_probs[legal_actions]
        action = legal_actions[torch.argmax(legal_probs)]
        probs[action] = 1.0
      else:
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0# 1 is legal
        action_probs = (action_probs.numpy()+1e-10)*legal_actions_mask
        probs = action_probs/sum(action_probs)
        action = np.random.choice(len(probs), size=1, p=probs)[0]
    except:
      print(action_probs)
      return self._get_random_action(legal_actions)
    return action, probs
  
  def _get_random_action(self, legal_actions):
    probs = np.zeros(self._num_actions)
    action = np.random.choice(legal_actions)
    probs[legal_actions] = 1.0 / len(legal_actions)
    return action,probs

  def _get_critic_action(self, info_state, legal_actions):
    probs = np.zeros(self._num_actions)
    info_state = torch.Tensor(np.reshape(info_state, [1, -1])).to(torch.device(self.device))
    qf1 = self.q1_net(info_state)
    qf2 = self.q2_net(info_state)
    q_values = torch.min(qf1, qf2)[0].to(torch.device(self.device))
    legal_q_values = q_values[legal_actions].to(torch.device(self.device))
    action = legal_actions[torch.argmax(legal_q_values.cpu())]
    probs[action] = 1.0
    return action, probs

  def _pick_action(self, info_state, legal_actions, is_evaluation=False, average_sampling=False, critic_action = False, random_action=False):
    if random_action:
      return self._get_random_action(legal_actions)
    elif critic_action:
      return self._get_critic_action(info_state, legal_actions)
    elif average_sampling:
      return self._get_action(info_state, legal_actions, greedy=False)
    elif (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return self._get_random_action(legal_actions)
    elif is_evaluation:
      return self._get_action(info_state, legal_actions, greedy=False)
    else:
      return self._get_action(info_state, legal_actions, greedy=False)

  def step(self, time_step, is_evaluation=False, add_transition_record=True, average_sampling=False, critic_action = False, random_action=False):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      info_state = time_step.observations["info_state"][self.player_id]
      action, probs = self._pick_action(info_state, legal_actions, is_evaluation, average_sampling, critic_action=critic_action, random_action=random_action)
      #print(action)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      if not average_sampling:
        self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_loss_value = self.learn()
      
      #if self._step_counter % self._update_target_network_every == 0:
      #  self.target_update()
      #print(self._prev_timestep, add_transition_record)
      if self._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        if not average_sampling:
          self.add_transition(self._prev_timestep, self._prev_action, time_step)
      if time_step.last():  # prepare for the next episode.
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None

    legal_actions = (prev_time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    
    next_legal_actions = (time_step.observations["legal_actions"][self.player_id])
    next_legal_actions_mask = np.zeros(self._num_actions)
    next_legal_actions_mask[next_legal_actions] = 1.0

    transition = Transition(
        info_state=(
            prev_time_step.observations["info_state"][self.player_id][:]),
        action=prev_action,
        reward=time_step.rewards[self.player_id],
        next_info_state=time_step.observations["info_state"][self.player_id][:],
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask,
        next_legal_actions_mask=next_legal_actions_mask)
    self._replay_buffer.add(transition)

  def learn(self, show=False):
    #print(len(self._replay_buffer), self._batch_size, self._min_buffer_size_to_learn)
    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None
    
    transitions = self._replay_buffer.sample(self._batch_size)
    info_states = torch.Tensor([t.info_state for t in transitions]).to(torch.device(self.device))
    actions = torch.LongTensor([t.action for t in transitions]).unsqueeze(1).to(torch.device(self.device))
    rewards = torch.Tensor([t.reward for t in transitions]).to(torch.device(self.device))
    next_info_states = torch.Tensor([t.next_info_state for t in transitions]).to(torch.device(self.device))
    are_final_steps = torch.Tensor([t.is_final_step for t in transitions]).to(torch.device(self.device))
    legal_actions_mask = torch.Tensor(
        [t.legal_actions_mask for t in transitions]).to(torch.device(self.device))# 1 is the leagal_action
    next_legal_actions_mask = torch.Tensor(
        [t.next_legal_actions_mask for t in transitions]).to(torch.device(self.device))# 1 is the leagal_action
    
    illegal_actions = 1 - legal_actions_mask#1 is the illeagal_action
    next_illegal_actions = 1 - next_legal_actions_mask#1 is the illeagal_action

    # Soft q  loss
    q1_value_loss, q2_value_loss = self.calculate_critic_losses(
      info_states, actions, rewards, next_info_states, are_final_steps, next_illegal_actions)
    
    # Update Soft q
    self.q1_optimizer.zero_grad()
    self.q2_optimizer.zero_grad()
    q1_value_loss.backward()
    q2_value_loss.backward()
    self.q1_optimizer.step()
    self.q2_optimizer.step()

    # Policy loss
    policy_loss,_ = self.calculate_actor_loss(info_states, illegal_actions)

    #self.save_loss(q1_value_loss, q2_value_loss, policy_loss)
    
    # Update Policy
    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    # Policy loss
    if show:
      print(policy_loss.item())
      with torch.no_grad():
        policy_loss,_ = self.calculate_actor_loss(info_states, illegal_actions)
        print(policy_loss.item())

    self.target_update()

  def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, next_illegal_actions):
    with torch.no_grad():
      try:
        _, action_probs, log_action_probs = self.policy_net.evaluate(next_state_batch, next_illegal_actions)#依旧存在问题
        if self._double_DQN:
          qf1_next_target = self.q1_target_net(next_state_batch)
          qf2_next_target = self.q2_target_net(next_state_batch)
          min_qf_next_target = action_probs * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probs)
        else:
          qf1_next_target = self.q1_target_net(next_state_batch)
          min_qf_next_target = action_probs * (qf1_next_target - self.alpha * log_action_probs)
        min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
        next_q_value = reward_batch.unsqueeze(1) + (1.0 - mask_batch.unsqueeze(1)) * self.gamma * (min_qf_next_target)
      except:
        traceback.print_exc()

    qf1 = self.q1_net(state_batch).gather(1, action_batch.long())
    qf2 = self.q2_net(state_batch).gather(1, action_batch.long())
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)


    return qf1_loss, qf2_loss

  def calculate_actor_loss(self, state_batch, illegal_actions):
    """Calculates the loss for the actor. This loss includes the additional entropy term"""
    _, action_probs, log_action_probs = self.policy_net.evaluate(state_batch, illegal_actions)
    with torch.no_grad():
      if self._double_DQN:
        qf1_pi = self.q1_net(state_batch)
        qf2_pi = self.q1_net(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
      else:
        min_qf_pi = self.q1_net(state_batch)
    inside_term = (self.alpha * log_action_probs - min_qf_pi)
    policy_loss = (action_probs * inside_term).sum(dim=1).mean()
    log_action_probabilities = torch.sum(log_action_probs * action_probs, dim=1)
    return policy_loss, log_action_probabilities

  def look_information(self, time_step):
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      info_state = time_step.observations["info_state"][self.player_id]

      with torch.no_grad():
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        legal_actions_mask = torch.Tensor(legal_actions_mask).to(torch.device(self.device))# 1 is the leagal_action
      
        illegal_actions = 1 - legal_actions_mask
        info_state = torch.Tensor(np.reshape(info_state, [1, -1])).to(torch.device(self.device))
        action_probs2 = self.policy_net.get_action(info_state)[0]
        _, action_probs, log_action_probs = self.policy_net.evaluate(info_state, illegal_actions)
        qf1 = self.q1_net(info_state)
        qf2 = self.q2_net(info_state)
        min_qf_pi = torch.min(qf1, qf2)
        inside_term = self.alpha * log_action_probs - min_qf_pi
        a = (action_probs * self.alpha * log_action_probs).sum(dim=1)
        b = (action_probs * (- min_qf_pi)).sum(dim=1)
        print('entropy', a)
        print('value', b)
        print(torch.abs(a/b))
        #policy_loss = (action_probs * inside_term)
        #print('qf:', min_qf_pi.numpy())
        #print('action_probs:', action_probs.numpy())
        #print('policy_loss:', policy_loss.numpy())
        #print(self._get_action(info_state, legal_actions, greedy=False))
    

class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)
