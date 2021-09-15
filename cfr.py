import pyspiel
from open_spiel.python import policy
import numpy as np
import numpy.random as rn
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


class InformationNode:
  def __init__(self, information_state, legal_actions):
    self._information_state = information_state
    self._legal_actions = legal_actions
    self._action_dim = len(legal_actions)
    self._accumulative_regrets = np.zeros(self._action_dim)
    self._average_strategy = np.zeros(self._action_dim)

  def _normalize(self, strategy):
    if strategy.sum() == .0:
      return np.array([1. / self._action_dim] * self._action_dim)
    else:
       return np.array(strategy) / strategy.sum()

  def get_strategy(self, regret_matching_plus=False):
    strategy = self._accumulative_regrets.clip(.0)
    if regret_matching_plus:
      self._accumulative_regrets = strategy
    strategy = self._normalize(strategy)
    return strategy

  def update_average_strategy(self, strategy, realization, iteration=1):
    if not iteration:
      iteration = 1
    self._average_strategy += strategy * realization * iteration

  def update_accumulative_regrets(self, regrets, external_reach_probability):
    self._accumulative_regrets += regrets * external_reach_probability

  def get_average_strategy(self):
    return self._normalize(self._average_strategy)


class CFRSolver(policy.Policy):
  def __init__(self,
               game,
               alternative_update=False,
               chance_sampling=False,
               linear_averaging=False,
               regret_matching_plus=False):
    all_players = list(range(game.num_players()))
    super(CFRSolver, self).__init__(game, all_players)
    self._information_set = {}
    self._game = game
    self._num_players = game.num_players()
    self._alternative_update = alternative_update
    self._chance_sampling = chance_sampling
    self._linear_averaing = linear_averaging
    self._regret_matching_plus = regret_matching_plus
    self._iteration = 1

  def _get_node(self, information_state, legal_actions):
    if information_state in self._information_set.keys():
      return self._information_set[information_state]
    else:
      ifnode = InformationNode(information_state, legal_actions)
      self._information_set[information_state] = ifnode
      return ifnode

  def solve(self, num_iterations=1000, progress=None):
    for iteration_ in range(num_iterations):
      progress.update()
      self._traverse_game(self._game.new_initial_state(), np.ones(3), iteration_ % self._num_players)
      if self._linear_averaing:
        self._iteration += 1

  def _traverse_game(self, state, reach_probabilities, update_player):
    if state.is_terminal():
      return np.array(state.rewards())

    elif state.is_chance_node():
      actions, probabilities = zip(*state.chance_outcomes())
      if not self._chance_sampling:
        rewards = np.zeros(2)
        for i in range(len(actions)):
          new_reach_probabilities = reach_probabilities.copy()
          new_reach_probabilities[-1] *= probabilities[i]
          rewards += self._traverse_game(state.child(actions[i]), new_reach_probabilities, update_player) * probabilities[i]
        return rewards
      else:
        action_ = rn.choice(actions, p=probabilities)
        return self._traverse_game(state.child(action_), reach_probabilities, update_player)

    elif state.is_player_node():
      if all(reach_probabilities[:-1] == 0):
        return np.zeros(self._num_players)

      player = state.current_player()
      information_state = state.information_state_string()
      legal_actions = state.legal_actions()
      ifnode = self._get_node(information_state, legal_actions)
      strategy = ifnode.get_strategy(self._regret_matching_plus)
      utils = []
      for probability, action_ in zip(strategy, legal_actions):
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[player] *= probability
        util_ = self._traverse_game(state.child(action_), new_reach_probabilities, update_player)[player]
        utils.append(util_)

      node_util = (np.array(utils) * strategy).sum()
      regrets = np.array(utils) - node_util
      rewards = np.array([node_util, -node_util]) if player == 0 else np.array([-node_util, node_util])

      # update current player's accumulative regrets and average policy
      erp = lambda pros, p: np.prod(list(pros)[p+1:] + list(pros)[:p])
      external_reach_probability = erp(reach_probabilities, player)
      ifnode.update_accumulative_regrets(regrets, external_reach_probability)
      ifnode.update_average_strategy(strategy, reach_probabilities[player], self._iteration)

      if self._alternative_update and player == update_player:
        return rewards

      # update rival's accumulative regrets and average policy
      rival = 1 if player == 0 else 0
      external_reach_probability = erp(reach_probabilities, rival)
      ifnode.update_accumulative_regrets(regrets, external_reach_probability)
      ifnode.update_average_strategy(strategy, reach_probabilities[rival], self._iteration)

      return rewards

  def action_probabilities(self, state, player_id=None):
    legal_actions = state.legal_actions()
    information_state = state.information_state_string(state.current_player())
    ifnode = self._get_node(information_state, legal_actions)
    average_strategy = ifnode.get_average_strategy()

    return {legal_actions[i]: average_strategy[i] for i in range(len(legal_actions))}

class CFRPlusSolver(CFRSolver):
  def __init__(self, game):
    super(CFRPlusSolver, self).__init__(game, alternative_update=True, regret_matching_plus=True, linear_averaging=True)


def config():
  parser = argparse.ArgumentParser(description='normal cfr calculation.')
  parser.add_argument('game', type=str, help='testbed game(must be 2-player zero-sum game)')
  parser.add_argument('iterations', type=int, help='total iterations for running')
  parser.add_argument('--seed', type=int, help='random seed')
  parser.add_argument('--nash', type=int, default=1, help='frequency for exploitability test')
  parser.add_argument('--no-draw', action='store_false', help='draw exploitability or not(default yes)')
  parser.add_argument('--linear', action='store_true', help='using linear averaging(default no)')
  parser.add_argument('--alternative', action='store_true', help='using alternative update(default no)')
  parser.add_argument('--chance', action='store_true', help='using chance sampling(default no)')
  parser.add_argument('--plus', action='store_true', help='using CFR plus(default no)')
  return parser.parse_args()


def main():
  args = config()
  game, iterations, seed, nash_frequency, draw, linear, alternative, chance, plus = args.game, \
                                                                                    args.iterations, \
                                                                                    args.seed, \
                                                                                    args.nash, \
                                                                                    args.no_draw, \
                                                                                    args.linear, \
                                                                                    args.alternative, \
                                                                                    args.chance, \
                                                                                    args.plus
  temp = lambda b: 'y' if b else 'n'
  filename = f'cfr_{iterations}_l{temp(linear)}_a{temp(alternative)}_c{temp(chance)}' if not plus else f'cfr_{iterations}_plus'
  rn.seed(seed)
  G = pyspiel.load_game(game)
  cfr = CFRPlusSolver(G) if plus else CFRSolver(G, linear_averaging=linear, alternative_update=alternative, chance_sampling=chance)
  f = open(f'{filename}.txt', 'wt')

  x_ = []
  y_ = []

  with tqdm(total=iterations) as pbar:
    current = 0
    while current != iterations:
      cfr.solve(num_iterations=nash_frequency, progress=pbar)
      conv = pyspiel.nash_conv(G, policy.python_policy_to_pyspiel_policy(policy.tabular_policy_from_callable(G, cfr.action_probabilities)))
      current += nash_frequency
      x_.append(current)
      y_.append(conv)
      f.write(f'{current} {conv}\n')
      f.flush()

  if draw:
    plt.xlabel('iteration')
    plt.ylabel('exploitability')
    plt.legend()
    plt.plot(x_, y_)
    plt.xscale('log')
    plt.savefig(f'{filename}.png')

  f.close()


if __name__ == '__main__':
  main()