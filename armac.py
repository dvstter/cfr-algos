import collections
import torch as T
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
import numpy as np
import numpy.random as rn
import pyspiel
from tqdm import tqdm
from open_spiel.python import policy
from copy import deepcopy

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
      idx = rn.randint(0, self._add_calls + 1)
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
    return rn.choice(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)

class CReLU(nn.Module):
  def __init__(self):
    super(CReLU, self).__init__()

  def forward(self, x):
    x = T.cat((x,-x), 1)
    return F.relu(x)

class AccumulativeRegrets(nn.Module):
  def __init__(self, information_state_size, num_actions, hidden=128):
    super(AccumulativeRegrets, self).__init__()
    self.l1 = nn.Linear(information_state_size, hidden)
    self.l2 = nn.Linear(hidden*6, num_actions)
    self.crelu1 = CReLU()
    self.crelu2 = CReLU()
    self.relu = nn.ReLU()

    self.dummy_param = nn.Parameter(T.empty(0)) # to get current device

  def forward(self, state, legal_actions):
    h1 = self.l1(state)
    h2 = T.cat([self.crelu1(h1), h1], 1)
    h3 = self.crelu2(h2)
    h4 = self.l2(h3)
    legal_actions = T.cat(h4.size()[0] * [T.LongTensor(legal_actions).unsqueeze(0)])
    legal_actions = legal_actions.to(self.dummy_param.device)
    return T.gather(self.relu(h4), 1, legal_actions)

  def get_distribution_without_grad(self, state, legal_actions):
    with T.no_grad():
      q_values = self.forward(T.FloatTensor(state).unsqueeze(0).to(self.dummy_param.device), legal_actions).cpu()
      if q_values.sum() == .0:
        return np.ones(len(legal_actions)) / len(legal_actions)
      else:
        return (q_values / q_values.sum()).squeeze().numpy()

class HistoryValue(nn.Module):
  def __init__(self, history_size, num_actions, embedding_size, hidden=128):
    super(HistoryValue, self).__init__()
    self.l1 = nn.Linear(history_size+embedding_size, hidden)
    self.l2 = nn.Linear(hidden*6, 1)
    self.embedding = nn.Embedding(num_actions, embedding_size)
    self.crelu1 = CReLU()
    self.crelu2 = CReLU()

    self.dummy_param = nn.Parameter(T.empty(0)) # to get current device

  def forward(self, state, action):
    input = T.cat([state, self.embedding(action)], dim=1)
    h1 = self.l1(input)
    h2 = T.cat([self.crelu1(h1), h1], 1)
    h3 = self.crelu2(h2)
    h4 = self.l2(h3)
    return h4

  def get_max_value_without_grad(self, state, legal_actions):
    return self.get_values_without_grad(state, legal_actions).max()

  def get_value_without_grad(self, state, action):
    state = T.unsqueeze(T.FloatTensor([state]).to(self.dummy_param.device), 0)
    with T.no_grad():
      return self.forward(state, action).cpu().item()

  def get_values_without_grad(self, state, legal_actions : list):
    states = T.cat(len(legal_actions) * [T.FloatTensor(state).unsqueeze(0)]).to(self.dummy_param.device)
    with T.no_grad():
      return self.forward(states, T.LongTensor(legal_actions).to(self.dummy_param.device)).squeeze().cpu().numpy()

class AverageStrategy(nn.Module):
  def __init__(self, information_state_size, num_actions, hidden=128):
    super(AverageStrategy, self).__init__()
    self.l1 = nn.Linear(information_state_size, hidden)
    self.l2 = nn.Linear(hidden*6, num_actions)
    self.crelu1 = CReLU()
    self.crelu2 = CReLU()
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1) # difference from class AccumulativeRegrets

    self.dummy_param = nn.Parameter(T.empty(0)) # to get current device

  def forward(self, state, legal_actions):
    h1 = self.l1(state)
    h2 = T.cat([self.crelu1(h1), h1], 1)
    h3 = self.crelu2(h2)
    h4 = self.l2(h3)
    legal_actions = T.cat(h4.size()[0] * [T.LongTensor(legal_actions).unsqueeze(0)])
    legal_actions = legal_actions.to(self.dummy_param.device)
    return T.gather(self.softmax(self.relu(h4)), 1, legal_actions)

  def get_distribution_without_grad(self, state, legal_actions):
    with T.no_grad():
      q_values = self.forward(T.FloatTensor(state).unsqueeze(0).to(self.dummy_param.device), legal_actions).cpu()
      if q_values.sum() == .0:
        return np.ones(len(legal_actions)) / len(legal_actions)
      else:
        return (q_values / q_values.sum()).squeeze().numpy()

#TRJ = collections.namedtuple('Record', 'history state action legal_actions player_seat regrets retrospective_policy')
TRJ = collections.namedtuple('Record', 'history state action legal_actions current_player regrets retrospective_policy')
class Trajectory:
  def __init__(self, player_seat, rival_param_index, reward=None, capacity=100):
    #self.player = player
    self.player_seat = player_seat
    self.rival_param_index = rival_param_index
    self.reward = reward
    self.buffer = []

  def add(self, history, state, action, legal_actions, acting_player, regrets, retrospective_policy):
    temp = TRJ(history, state, action, legal_actions, acting_player, regrets, retrospective_policy)
    self.buffer.append(temp)

  def __iter__(self):
    return iter(self.buffer)

  def __str__(self):
    return f'Trajectory : player {self.player_seat} rival_param_index {self.rival_param_index} reward {self.reward} length {len(self.buffer)}'

class ArmacSolver(policy.Policy):
  def __init__(self,
               game,
               device: str = 'cuda' if T.cuda.is_available() else 'cpu',
               num_learning: int = 64,
               lr_history_value: float = 1e-3,
               lr_accumulative_regrets: float = 1e-3,
               lr_average_strategy: float = 1e-3,
               learning_rate: float = 1e-4,
               batch_size: int = 1024,
               memory_capacity: int = int(1e6),
               reset_parameters: bool = False):
    all_players = list(range(game.num_players()))
    super(ArmacSolver, self).__init__(game, all_players)
    self._game = game
    self._device = device
    self._root_node = game.new_initial_state()
    self._feature_size = len(self._root_node.information_state_tensor(0))
    self._history_size = self._feature_size + 8
    self._num_actions = game.num_distinct_actions()
    self._num_learning = num_learning
    self._num_players = game.num_players()
    # This class will first focus on two-player zero-sum poker games
    assert(self._num_players == 2)
    self._learning_rate = learning_rate
    self._batch_size = batch_size
    self._memory_capacity = memory_capacity

    self._retrospective_params = []
    self._history_value = HistoryValue(self._history_size, self._num_actions, 5).to(device)
    self._past_history_value = HistoryValue(self._history_size, self._num_actions, 5).to(device)
    self._accumulative_regrets = AccumulativeRegrets(self._feature_size, self._num_actions).to(device)
    self._past_accumulative_regrets = AccumulativeRegrets(self._feature_size, self._num_actions).to(device)
    self._average_strategy = AverageStrategy(self._feature_size, self._num_actions).to(device)
    self._epoch_buffer = ReservoirBuffer(memory_capacity)
    self._optimizer_accumulative_regrets = O.Adam(self._accumulative_regrets.parameters(), lr=lr_accumulative_regrets)
    self._optimizer_history_value = O.Adam(self._history_value.parameters(), lr=lr_history_value)
    self._optimizer_average_strategy = O.Adam(self._average_strategy.parameters(), lr=lr_average_strategy)

    self._loss_MSE = nn.MSELoss().to(device)
    self._loss_div = nn.KLDivLoss().to(device)

    if reset_parameters:
      for network in [self._history_value, self._accumulative_regrets, self._past_history_value, self._past_accumulative_regrets, self._average_strategy]:
        self.__class__.reset_parameters(network)

    self._player_seat = 1
    self._counter = 0

  @staticmethod
  def reset_parameters(network):
    for par in network.state_dict().values():
      par.fill_(.0)

  @staticmethod
  def full_history_state(state):
    return state.information_state_tensor(0)[:8] + state.information_state_tensor(1)

#  def _load_past_accumulative_regrets(self, param_index):
#    if param_index >= len(self._retrospective_params):
#      self._past_accumulative_regrets.load_state_dict(deepcopy(self._accumulative_regrets.state_dict()))
#    else:
#      self._past_accumulative_regrets.load_state_dict(deepcopy(self._retrospective_params[param_index]))

  def _load_past_parameters(self, param_index):
    if param_index >= len(self._retrospective_params):
      self._past_accumulative_regrets.load_state_dict(deepcopy(self._accumulative_regrets.state_dict()))
      self._past_history_value.load_state_dict(deepcopy(self._history_value.state_dict()))
    else:
      self._past_accumulative_regrets.load_state_dict(self._retrospective_params[param_index][0])
      self._past_history_value.load_state_dict(self._retrospective_params[param_index][1])

  def solve(self, num_iterations, num_traversals):
    for _ in range(num_iterations):
      self._epoch_buffer.clear()
      for _ in range(num_traversals):
        self._player_seat = (self._player_seat + 1) % self._num_players
        retrospective_param_len = len(self._retrospective_params)
        param_index = rn.randint(0, retrospective_param_len) if retrospective_param_len > 0 else 0
        trajectory_buffer = Trajectory(self._player_seat, param_index, capacity=num_traversals)
#        self._load_past_accumulative_regrets(param_index)
        self._load_past_parameters(param_index)
        terminal_state = self._traverse_game_tree(self._root_node,
                                                  self._player_seat,
                                                  trajectory_buffer)
        reward = terminal_state.returns()[self._player_seat]
        trajectory_buffer.reward = reward
        self._epoch_buffer.add(trajectory_buffer)

      for _ in range(self._num_learning):
        self._learning()

      self._retrospective_params.append([deepcopy(self._accumulative_regrets.state_dict()), deepcopy(self._history_value.state_dict())])

  def _learning(self):
    batch_trajectories = self._epoch_buffer.sample(self._batch_size)
    for trajectory in batch_trajectories:
      reward = T.FloatTensor([trajectory.reward]).to(self._device)
      player_seat = trajectory.player_seat
      for idx in range(len(trajectory.buffer)):
        record = trajectory.buffer[idx]
        next_record = trajectory.buffer[idx+1]  if (idx+1) < len(trajectory.buffer) else None

        history = T.FloatTensor(record.history).unsqueeze(0).to(self._device)
        state = T.FloatTensor(record.state).unsqueeze(0).to(self._device)
        next_state = next_record.state if next_record else None
        next_history = next_record.history if next_record else None
        action = T.LongTensor(record.action).to(self._device)
        legal_actions = record.legal_actions
        next_legal_actions = next_record.legal_actions if next_record else None
        current_player = record.current_player
        retrospective_policy = T.FloatTensor(record.retrospective_policy).to(self._device)

        # train critic
        self._optimizer_history_value.zero_grad()
        if next_state:
          l1 = self._loss_MSE(self._history_value(history, action), T.FloatTensor([self._history_value.get_max_value_without_grad(next_history, next_legal_actions)]).unsqueeze(0).to(self._device))
        else:
          l1 = self._loss_MSE(self._history_value(history, action), reward.unsqueeze(0))
        l1.backward()
        self._optimizer_history_value.step()

        # train accumulative regret
        if current_player == player_seat:
          regrets = T.FloatTensor(record.regrets).to(self._device)
          self._optimizer_accumulative_regrets.zero_grad()
          l2 = self._loss_MSE(self._accumulative_regrets(state, legal_actions), regrets.unsqueeze(0))
          l2.backward()
          self._optimizer_accumulative_regrets.step()

        # train average actor
        else:
          assert(record.regrets is None) # just for test
          self._optimizer_average_strategy.zero_grad()
          l3 = self._loss_div(self._average_strategy(state, legal_actions), retrospective_policy)
          l3.backward()
          self._optimizer_average_strategy.step()

  def _traverse_game_tree(self, state, player_seat, buffer):
    if state.is_terminal():
      return state

    elif state.is_chance_node():
      action = rn.choice([i[0] for i in state.chance_outcomes()])
      return self._traverse_game_tree(state.child(action), player_seat, buffer)

    else:
      # sample action to traverse game tree for both players
      actor_network = self._accumulative_regrets if state.current_player() == 0 else self._past_accumulative_regrets
      information_state = state.information_state_tensor(state.current_player()) # not a torch tensor
      legal_actions = state.legal_actions()
      strategy = actor_network.get_distribution_without_grad(information_state, legal_actions)
      action = rn.choice(legal_actions, 1, p=strategy)
      full_history_state = self.__class__.full_history_state(state)

      past_strategy = self._past_accumulative_regrets.get_distribution_without_grad(information_state, legal_actions)
      if state.current_player() == player_seat:
        past_history_values = self._past_history_value.get_values_without_grad(full_history_state, legal_actions)
        regrets = past_history_values - (past_strategy * past_history_values).sum()
        buffer.add(full_history_state, information_state, action, legal_actions, state.current_player(), regrets, past_strategy)
      else:
        buffer.add(full_history_state, information_state, action, legal_actions, state.current_player(), None, past_strategy)
        pass

      return self._traverse_game_tree(state.child(action), player_seat, buffer)

  def action_probabilities(self, state, player_id=None):
    legal_actions = state.legal_actions()
    probabilities = self._average_strategy.get_distribution_without_grad(state.information_state_tensor(state.current_player()), legal_actions)
    return {legal_actions[i]: probabilities[i] for i in range(len(legal_actions))}


def main():
  current = 0
  iterations = list(range(100, 100100, 100))
  rn.seed(1234)
  G = pyspiel.load_game('leduc_poker')
  armac = ArmacSolver(G,
                  num_learning=3,
                  batch_size=8)
  f = open('armac_result.txt', 'wt')
  # for testing
#  iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#  iterations = [x//10 for x in iterations]

  with tqdm(total=iterations[-1]) as pbar:
    for iteration_ in iterations:
      armac.solve(num_iterations=(iteration_ - current), num_traversals = 16)
      conv = pyspiel.nash_conv(G, policy.python_policy_to_pyspiel_policy(policy.tabular_policy_from_callable(G, armac.action_probabilities)))
      f.write(f'{iteration_} {conv}\n')
      f.flush()
      pbar.update(iteration_ - current)
      current = iteration_

  f.close()


if __name__ == '__main__':
  main()