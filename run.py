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

"""Tests for open_spiel.python.algorithms.nfsp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python import algorithms
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch.dqn import ReplayBuffer
#from open_spiel.python.pytorch import nfsp
import pyspiel


import torch
import random
import numpy as np
import traceback
import copy
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
#from algorithms import policy_gradient
#from algorithms import nfsp
from algorithms import Deep-FTRL-ORW
from multiprocessing import Process
import traceback

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "game",
    'kuhn_poker',
    #"battleship(allow_repeated_shots=False,board_height=2,board_width=2,num_shots=3,ship_sizes=[1;2],ship_values=[1.0;2.0])",
    #"dark_hex",#big
    #'liars_dice',#big
    #'kuhn_poker',
    #'leduc_poker',#mid
    #"turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=4,players=2,points_order=descending))",
    "Name of the game")
flags.DEFINE_integer("num_update_iteration", 200, "how number of to update iterations")
flags.DEFINE_integer("print_freq", int(2e3), "How often to print the exploitability")
flags.DEFINE_integer("num_train_episodes", int(2e5),
                     "Number of training episodes in every iteration.")
flags.DEFINE_integer("SL_nums", 128,
                     "Number of SL training.")
flags.DEFINE_list("hidden_layers_sizes", [
    128
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("batch_size", 32, "Number of batch_size in the avg-net and Q-net.")#64
flags.DEFINE_integer("learn_every", 16, "Number of learn_every in the avg-net and Q-net.")#32
flags.DEFINE_integer("min_buffer_size_to_learn", 100, "Number of min_buffer_size_to_learn in the avg-net and Q-net.")
flags.DEFINE_integer("update_target_network_every", 500, "Number of update_target_network_every in the Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(1e5),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_float("q_lr", 1e-1,
                   "q_lr of SAC and DQN.")                   
flags.DEFINE_float("policy_lr", 1e-1,
                   "policy_lr of SAC.")    
flags.DEFINE_float("SL_lr", 1e-1,#origin is 5e-2. 1e-1 in 1, 2e-1 in2
                   "SL_lr of nfsp and DeepFTRLORW.") 
flags.DEFINE_integer("multi_num", 1,
                     "Size of the multi_excute_num.")
flags.DEFINE_float("alpha_base", 3,
                    "alpha_base")
flags.DEFINE_string("device",'0',"device")              
ratio = 1

def get_env():
  #game = FLAGS.game
  num_players = 2
  game = pyspiel.load_game(FLAGS.game)
  env_configs = {"players": num_players, "imp_info":True, "num_cards":4}
  env = rl_environment.Environment(game, enable_legality_check=True)
  return env


def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  #torch.backends.cudnn.deterministic = True


class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    #print(prob_dict)
    return prob_dict




def get_see_time_step():
  env = get_env()
  time_step = env.reset()
  return copy.copy(time_step)


def write_txt(string, name=None):
  f=open(name+'.txt','w',encoding='utf-8')
  f.write(string)
  f.close()


def write_add_txt(string, name=None):
  f=open(name+'.txt','a',encoding='utf-8')
  f.write(string)
  f.close()


def delete_txt(name=None):
  import os
  try:
    os.remove(name+'.txt')
  except:
    pass


def main(unused_argv):
  def update_and_compute_exploitability_process(name, res, seed=0, see_temp=None):
    import traceback
    try:
      if name=='DeepFTRLORW':
        setup_seed(seed)
        res['xindex'], res['data'] = run_DeepFTRLORW(see_temp)
      traceback.print_exc()



  def run_DeepFTRLORW(see_temp=None):
    delete_txt('DeepFTRLORW')
    #game = FLAGS.game
    env = get_env()

    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "q_lr":FLAGS.q_lr,
        "policy_lr":FLAGS.policy_lr,
    }
    agents = [
          Deep-FTRL-ORW.DeepFTRLORWagent(  # pylint: disable=g-complex-comprehension
              player_id,
              state_representation_size=info_state_size,
              num_actions=num_actions,
              hidden_layers_sizes=hidden_layers_sizes,
              device = FLAGS.device,
              reservoir_buffer_capacity=FLAGS.reservoir_buffer_capacity,
              anticipatory_param=FLAGS.anticipatory_param,
              batch_size=FLAGS.batch_size,
              min_buffer_size_to_learn=FLAGS.min_buffer_size_to_learn,
              learn_every=FLAGS.learn_every,
              sl_learning_rate=FLAGS.SL_lr,
              **kwargs) for player_id in [0, 1]
      ]

    expl_policies_avg = NFSPPolicies(env, agents, DeepFTRLORW.MODE.average_policy)
    expl_policies_cur = NFSPPolicies(env, agents, DeepFTRLORW.MODE.best_response)
    
    times = []
    conv = []
    see = True

    for i in range(FLAGS.num_train_episodes):
      if i==0 or (i+1) % FLAGS.num_update_iteration == 0:
        iteration = int((i+1) /FLAGS.num_update_iteration)
        for agent in agents:
          import math
          #agent._rl_agent.alpha = FLAGS.alpha_base*math.pow(iteration+2, 0.5)/math.pow(iteration+1, 1)#*1/math.sqrt(iteration+1)
          agent._rl_agent.alpha = FLAGS.alpha_base/math.pow(iteration+1, 1)#*1/math.sqrt(iteration+1)
      if i==0 or (i+1) % FLAGS.print_freq == 0:
        output_str =''
        iteration = int((i+1) /FLAGS.print_freq)
        for agent in agents:
          output_str = output_str# + see_policy(agent, see_temp, 'DeepFTRLORW')
          pass
        expa = exploitability.exploitability(env.game, expl_policies_avg)
        expc = exploitability.exploitability(env.game, expl_policies_cur)
        temp_str = "DeepFTRLORW {} Exploitability AVG {:.5f}  CUR:{:.5f}".format(i, expa, expc)+'\n'
        output_str = output_str + temp_str
        write_add_txt(output_str, 'DeepFTRLORW')
        print(temp_str, end='')
        times.append(i)
        conv.append(expa)

      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = agents[current_player]
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])
      for agent in agents:
        agent.step(time_step)
    
    agents[0]._rl_agent.plot_loss()
    return times, conv


  def multi_excute(returnres, name):
    try:
      from multiprocessing import Manager
      manager = Manager()
      res = []
      for i in range(FLAGS.multi_num):
        res.append(manager.dict())

      see_step = None#get_see_time_step()

      tasks = []
      for i in range(FLAGS.multi_num):
        tasks.append(Process(target=update_and_compute_exploitability_process, args=(name, res[i], i, copy.copy(see_step),)))

      for t in tasks:
       t.start()

      for t in tasks:
       t.join()

      # for t in tasks[0:2]:
      #   t.start()

      # for t in tasks[0:2]:
      #   t.join()

      # for t in tasks[2:4]:
      #   t.start()

      # for t in tasks[2:4]:
      #   t.join()
      
      _exploitability = []
      for i in range(len(res)):
        resconv = res[i].values()[1]
        time = res[i].values()[0]
        if i==0:
          for item in resconv:
            _exploitability.append([item])
        else:
          for j in range(len(resconv)):
            _exploitability[j].append(resconv[j])
      
      returnres['xindex'] = time 
      returnres['data'] = _exploitability
    except:
      traceback.print_exc()


  def run():
    from multiprocessing import Manager  
    multi_num = FLAGS.multi_num 
    manager = Manager()
    tasks = []
    res = []
    names = ['DeepFTRLORW']

    for i in range(len(names)):
      res.append(manager.dict())
      tasks.append(Process(target=multi_excute, args=(res[i], names[i],)))


    for t in tasks:
      t.start()
      # t.join()

    for t in tasks:
     t.join()
    
    data = []
    for i in range(len(res)):
      data.append(res[i].values()[1])
      times = res[i].values()[0]
    

  import time
  time_start=time.time()
  run()
  time_end=time.time()
  print('totally cost :{:.5f} minutes'.format((time_end-time_start)/60.0))
  

if __name__ == "__main__":
  import time
  time_start=time.time()
  app.run(main)
