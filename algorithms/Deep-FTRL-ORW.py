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
import os
import random
from absl import logging
import numpy as np

import torch
import torch.nn.functional as F
import traceback
from open_spiel.python import rl_agent
from open_spiel.python.pytorch import dqn
from .SAC_discrete import SACDiscrete, PolicyNetwork
from open_spiel.python.pytorch.dqn import MLP


Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask action")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

MODE = enum.Enum("mode", "best_response average_policy")


class DeepFTRLORWagent(rl_agent.AbstractAgent):

  def __init__(self,
               player_id,
               state_representation_size,
               num_actions,
               hidden_layers_sizes,
               reservoir_buffer_capacity,
               anticipatory_param,
               device = 'cpu',
               batch_size=128,
               rl_learning_rate=0.01,
               sl_learning_rate=0.01,
               min_buffer_size_to_learn=1000,
               learn_every=64,
               optimizer_str="sgd",
               max_action_prob = True,
               **kwargs):
    """Initialize the `entropy-NFSP` agent."""
    if not device == "cpu":
      self.device = 'cuda:' + str(device)
    else:
      self.device = 'cpu'
    self._max_action_prob = max_action_prob
    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._state_representation_size = state_representation_size
    self._num_actions = num_actions
    self._hidden_layers_sizes = hidden_layers_sizes

    self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._step_counter = 0

    # Inner RL agent
    kwargs.update({
        "batch_size": batch_size,
        "learn_every": learn_every,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "optimizer_str": optimizer_str,
    })
    self._kwargs = kwargs
    self._rl_agent = SACDiscrete(player_id, state_representation_size,
                             num_actions, device=device, hidden_layers_sizes=hidden_layers_sizes, **kwargs)

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._last_sl_loss_value = None

    # Average policy network.
    self._avg_network = PolicyNetwork(state_representation_size,
                                self._layer_sizes, num_actions).to(torch.device(self.device))

    self._savers = [
        ("q1_network", self._rl_agent.q1_net),
        ("q2_network", self._rl_agent.q2_net),
        ("avg_network", self._avg_network)
    ]

    if optimizer_str == "adam":
      self.optimizer = torch.optim.Adam(
          self._avg_network.parameters(), lr=sl_learning_rate)
    elif optimizer_str == "sgd":
      self.optimizer = torch.optim.SGD(
          self._avg_network.parameters(), lr=sl_learning_rate)
    else:
      raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")

    self._losses = []
    self._step_counters = []
    self._sample_episode_policy()

  @contextlib.contextmanager
  def temp_mode_as(self, mode):
    """Context manager to temporarily overwrite the mode."""
    previous_mode = self._mode
    self._mode = mode
    yield
    self._mode = previous_mode

  def _refresh_rl_agent(self):
    self._rl_agent = SACDiscrete(self.player_id, self._state_representation_size,
                             self._num_actions, hidden_layers_sizes=self._hidden_layers_sizes, **self._kwargs)
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._last_sl_loss_value = None

  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  def _act(self, info_state, legal_actions):
    info_state = np.reshape(info_state, [1, -1])
    with torch.no_grad():
      action_values = self._avg_network(torch.Tensor(info_state).to(torch.device(self.device)))
      action_probs = F.softmax(action_values, dim=1).detach().to(torch.device(self.device)).cpu()

    self._last_action_values = action_values[0]
    # Remove illegal actions, normalize probs
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = action_probs[0][legal_actions]
    probs /= sum(probs)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  @property
  def mode(self):
    return self._mode

  @property
  def loss(self):
    return (self._last_sl_loss_value, self._last_rl_loss_value().detach())

  def look_information(self, time_step):
    self._rl_agent.look_information(time_step)

  def step(self, time_step, is_evaluation=False, average_sampling=False):
    """Returns the action to be taken and updates the Q-networks if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    #if self._mode == MODE.best_response:
    #  agent_output = self._rl_agent.step(time_step, is_evaluation=is_evaluation, average_sampling=average_sampling)
    #  if not is_evaluation and not time_step.last():
    #    self._add_transition(time_step, agent_output)
    #  #if self._prev_timestep and not is_evaluation and not time_step.last():
    #  #  self._rl_agent.add_transition(self._prev_timestep, self._prev_action,
    #  #                                time_step)

    #elif self._mode == MODE.average_policy:
    #  # Act step: don't act at terminal info states.
    #  if not time_step.last():
    #    info_state = time_step.observations["info_state"][self.player_id]
    #    legal_actions = time_step.observations["legal_actions"][self.player_id]
    #    action, probs = self._act(info_state, legal_actions)
    #    agent_output = rl_agent.StepOutput(action=action, probs=probs)
    if self._mode == MODE.best_response:
      agent_output = self._rl_agent.step(time_step, is_evaluation)
      if not is_evaluation and not time_step.last():
        self._add_transition(time_step, agent_output)

    elif self._mode == MODE.average_policy:
      # Act step: don't act at terminal info states.
      if not time_step.last():
        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        action, probs = self._act(info_state, legal_actions)
        agent_output = rl_agent.StepOutput(action=action, probs=probs)

      if self._prev_timestep and not is_evaluation:
        self._rl_agent.add_transition(self._prev_timestep, self._prev_action,
                                      time_step)
      
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    if not is_evaluation:
      self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_sl_loss_value = self._learn()
        # If learn step not triggered by rl policy, learn.
        if self._mode == MODE.average_policy:
          self._rl_agent.learn()

      # Prepare for the next episode.
      if time_step.last():
        self._sample_episode_policy()
        self._prev_timestep = None
        self._prev_action = None
        self._rl_agent._prev_action = None
        self._rl_agent._prev_timestep = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = agent_output.action

    return agent_output

  def _add_transition(self, time_step, agent_output):
    """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
    legal_actions = time_step.observations["legal_actions"][self.player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(time_step.observations["info_state"][self.player_id][:]),
        action_probs=agent_output.probs,
        legal_actions_mask=legal_actions_mask,
        action=agent_output.action)
    self._reservoir_buffer.add(transition)

  def _learn(self):
    """Compute the loss on sampled transitions and perform a avg-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if (len(self._reservoir_buffer) < self._batch_size or
        len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._reservoir_buffer.sample(self._batch_size)
    info_states = torch.Tensor([t.info_state for t in transitions]).to(torch.device(self.device))
    action_probs = torch.Tensor([t.action_probs for t in transitions]).to(torch.device(self.device))
    actions = torch.Tensor([t.action for t in transitions]).to(torch.device(self.device))
    legal_actions_mask = torch.Tensor(
        [t.legal_actions_mask for t in transitions]).to(torch.device(self.device))
    illegal_actions = 1.0 - legal_actions_mask
    
    self.optimizer.zero_grad()
    if self._max_action_prob:
      loss = F.cross_entropy(self._avg_network(info_states), torch.max(action_probs, dim=1)[1])
    else:
      loss = F.cross_entropy(self._avg_network(info_states), actions.long())
    # _,traget_probs,_ = self._avg_network.evaluate(info_states, illegal_actions)
    # loss = F.mse_loss(traget_probs, action_probs)
    loss.backward()
    self.optimizer.step()
    #self.save_loss(loss)
    return loss.detach()

  def save_loss(self, loss):
    self._step_counters.append(self._step_counter)
    self._losses.append(loss.item())

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
  
  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, checkpoint_dir):
    """Saves the average policy network and the inner RL agent's q-network.

    Note that this does not save the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory where checkpoints will be saved.
    """
    for name, model in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      torch.save(model.to(torch.device('cpu')).state_dict(), path)
      logging.info("Saved to path: %s", path)

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      if os.path.exists(path):
        return True
    return False

  def restore(self, checkpoint_dir):
    """Restores the average policy network and the inner RL agent's q-network.

    Note that this does not restore the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory from which checkpoints will be restored.
    """
    for name, model in self._savers:
      full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
      logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
      model.load_state_dict(torch.load(full_checkpoint_dir))
      model.to(torch.device(self.device))


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
