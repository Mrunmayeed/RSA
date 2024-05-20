import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gymnasium import spaces


class NetworkEnv(gym.Env):

    def __init__(self, gml_file, num_requests=100, case=1):
        self.graph = nx.read_gml(gml_file)
        self.num_requests = num_requests
        self.case = case
        self.min_ht = 10
        self.max_ht = 20
        self.nodes = list(self.graph.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}

        self.edge_utilization_history = {edge: [] for edge in self.graph.edges}

        self.observation_space = spaces.Dict(
            {
                'link_utilization': spaces.Box(0, 10, shape=(len(self.graph.edges),), dtype=int),
                'req': spaces.Box(0, len(self.nodes) - 1, shape=(2,), dtype=int),
                'current_node': spaces.Discrete(len(self.nodes))
            }
        )
        self.action_space = spaces.Discrete(len(self.nodes))
        self.round = 0
        self.current_requests = []  # Track current requests with their holding times
        self.reset()

    def _generate_req(self, case=0):
        if case == 1:
            src_dst_pair = ('San Diego Supercomputer Center', 'Jon Von Neumann Center, Princeton, NJ')
        else:
            src_dst_pair = np.random.choice(self.nodes, 2, replace=False)
        holding_time = np.random.randint(self.min_ht, self.max_ht)
        return {
            'src': self.node_to_idx[src_dst_pair[0]],
            'dst': self.node_to_idx[src_dst_pair[1]],
            'holding_time': holding_time
        }

    def _get_valid_actions(self, current_node):
        neighbors = list(self.graph.neighbors(self.idx_to_node[current_node]))
        return [self.node_to_idx[neighbor] for neighbor in neighbors]

    def _update_action_space(self, current_node):
        valid_actions = self._get_valid_actions(current_node)
        self.action_space = spaces.Discrete(len(valid_actions))
        self.valid_actions = valid_actions

    def _get_obs(self):
        link_utilization = np.array([self.graph.edges[edge]['utilization'] for edge in self.graph.edges])
        return {
            'link_utilization': link_utilization,
            'req': np.array([self._req['src'], self._req['dst']]),
            'current_node': self._current_node
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for edge in self.graph.edges:
            self.graph.edges[edge]['utilization'] = 0
        self.current_requests = []
        self._req = self._generate_req(case=0)
        self._current_node = self._req['src']
        self._update_action_space(self._current_node)
        self.round = 0

        observation = self._get_obs()
        info = {}

        return observation, info

    def _release_resources(self):
        for request in self.current_requests:
            request['holding_time'] -= 1
            if request['holding_time'] <= 0:
                for edge in request['path']:
                    self.graph.edges[edge]['utilization'] -= 1
        self.current_requests = [req for req in self.current_requests if req['holding_time'] > 0]

    def step(self, action):
        self.round += 1
        terminated = (self.round == self.num_requests)


        ## As the action space is dynamic, this presents to a bug in the gymnasium.
        # The action selected is sometimes out if the action space. This causes intermittent failures in training.
        # To handle such cases the following patch fix was introduced.
        if action not in self.action_space:
            while action not in self.action_space:
                action -= 1
            if action<0:
                action=0

        selected_neighbor_idx = self.valid_actions[action]
        selected_edge = (self.nodes[self._current_node], self.nodes[selected_neighbor_idx])

        blocking_reward = -1
        reward = 0

        if self.graph.edges[selected_edge]['utilization'] + 1 <= 10:
            self.graph.edges[selected_edge]['utilization'] += 1
            reward = 1
            self.current_requests.append({
                'path': [selected_edge],
                'holding_time': self._req['holding_time']
            })
            self._current_node = selected_neighbor_idx
            self._update_action_space(self._current_node)
        else:
            reward = blocking_reward

        if self._current_node == self._req['dst']:
            terminated = True

        self._release_resources()
        self._req = self._generate_req() if terminated else self._req
        observation = self._get_obs()

        if terminated:
            self._store_edge_utilization()
            ## After 100 episodes, the graph is created.
            if len(list(self.edge_utilization_history.values())[0])>100:
                self.plot_edge_utilization()

        info = {}

        return observation, reward, terminated, terminated, info

    def _store_edge_utilization(self):
        for edge in self.graph.edges:
            self.edge_utilization_history[edge].append(self.graph.edges[edge]['utilization'])

    def plot_edge_utilization(self):
        print("Printing graph")
        plt.figure(figsize=(12, 8))
        ##Summarize and calculate utilization
        util = np.zeros(101)
        for edge, utilization in self.edge_utilization_history.items():
            util = np.array(utilization)+util
        plt.plot(util/(10*15))
        plt.xlabel("Episodes")
        plt.ylabel("Mean Utilization")
        plt.title("Edge Utilization Over Episodes")
        plt.legend()
        plt.show()