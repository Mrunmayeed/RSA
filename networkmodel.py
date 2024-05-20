import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

min_ht = 4
max_ht = 10

class Request:
    """This class represents a request. Each request is characterized by source and destination nodes and holding time (represented by an integer).

    The holding time of a request is the number of time slots for this request in the network. You should remove a request that exhausted its holding time.
    """

    def __init__(self, s, t, ht):
        self.s = s
        self.t = t
        self.ht = ht

    def __str__(self) -> str:
        return f'req({self.s}, {self.t}, {self.ht})'

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        # used by set()
        return self.id


class EdgeStats:
    """This class saves all state information of the system. In particular, the remaining capacity after request mapping and a list of mapped requests should be stored.
    """

    def __init__(self, u, v, cap) -> None:
        self.id = (u, v)
        self.u = u
        self.v = v
        # remaining capacity
        self.cap = cap

        # spectrum state (a list of requests, showing color <-> request mapping). Each index of this list represents a color
        self.__slots = [None] * cap
        # a list of the remaining holding times corresponding to the mapped requests
        self.__hts = [0] * cap

    def __str__(self) -> str:
        return f'{self.id}, cap = {self.cap}: {self.reqs}'

    def add_request(self, req: Request, color: int):
        """update self.__slots by adding a request to the specific color slot

        Args:
            req (Request): a request
            color (int): a color to be used for the request
        """
        self.__slots[color] = req
        self.__hts[color] = req.ht

    def remove_requests(self):
        """update self.__slots by removing the leaving requests based on self.__hts; Also, self.__hts should be updated in this function.
        """
        for i in range(len(self.__hts)):
            if self.__hts[i] > 0:
                self.__hts[i] -= 1
                if self.__hts[i] == 0:
                    self.__slots[i] = None

    def get_available_colors(self) -> list[int]:
        """return a list of integers available to accept requests
        """
        return [i for i in range(self.cap) if self.__hts[i] == 0]

    def show_spectrum_state(self):
        """Come up with a representation to show the utilization state of a link (by colors)
        """
        print(f"Link {self.id}: {self.__slots}")


def generate_requests(num_reqs: int, g: nx.Graph, case=0) -> list[Request]:
    """Generate a set of requests, given the number of requests and an optical network (topology)

    Args:
        num_reqs (int): the number of requests
        g (nx.Graph): network topology

    Returns:
        list[Request]: a list of request instances
    """
    nodes = g.nodes
    requests = []

    for _ in range(num_reqs):
        if case==1:
            u, v = ('San Diego Supercomputer Center', 'Jon Von Neumann Center, Princeton, NJ')
        else:
            u, v = np.random.choice(nodes, 2, replace= False)
        ht = np.random.randint(min_ht, max_ht)
        requests.append(Request(u, v, ht))

    return requests

def generate_graph() -> nx.Graph:
    """Generate a networkx graph instance importing a GML file. Set the capacity attribute to all links based on a random distribution.

    Returns:
        nx.Graph: a weighted graph
    """
    return nx.read_gml('nsfnet.gml')

def generate_edgestats(capacity:int, g:nx.Graph):
    """Generate a networkx graph instance importing a GML file. Set the capacity attribute to all links based on a random distribution.

    Returns:
        nx.Graph: a weighted graph
    """
    edgestats = {}
    for e in g.edges:
        edgestats[tuple(sorted(e))] = EdgeStats(*e, capacity)

    return edgestats



def route(g: nx.Graph, estats: list[EdgeStats], req: Request) -> list[EdgeStats]:
    """Use a routing algorithm to decide a mapping of requests onto a network topology. The results of mapping should be reflected. Consider available colors on links on a path.

    Args:
        g (nx.Graph): a network topology
        req (Request): a request to map

    Returns:
        list[EdgeStats]: updated EdgeStats
    """
    try:
        path = nx.shortest_path(g, req.s, req.t)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = tuple(sorted([u,v]))
            available_colors = estats[e].get_available_colors()
            print(f"Request {req} mapped to Edge {u}-{v} available color {available_colors}")
            if available_colors:
                # color = min(available_colors)
                ## Random selection heuristic approach
                color = np.random.choice(available_colors)
                estats[e].add_request(req, color)
                print(f"Request {req} mapped to Edge {u}-{v} with color {color}")
            else:
                print(f"No available colors found to map request {req} along path {path}")
                return {}

        return estats
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # 1. generate a network
    G = generate_graph()

    num_reqs= 100
    capacity = 10

    # 2. generate a list of requests (num_reqs)
    requests = generate_requests(num_reqs, G)

    # 3. prepare an EdgeStats instance for each edge.
    edge_stats = generate_edgestats(capacity, G)

    # 4. this simulation follows the discrete event simulation concept. Each time slot is defined by an arrival of a request
    blocked_request = {i:0 for i in range(num_reqs)}
    utilization = {e:{i:0.0 for i in range(num_reqs)} for e in edge_stats}
    blocked_requests = 0
    for i in range(len(requests)):
        req = requests[i]
        # 4.1 use the route function to map the request onto the topology (update EdgeStats)
        edge_stat = route(G, edge_stats, req)

        if edge_stat:
            # 4.2 remove all requests that exhausted their holding times (use remove_requests)
            edge_stats = edge_stat
            for e,estat in edge_stat.items():
                estat.remove_requests()
                avl = len(estat.get_available_colors())
                utilization[e][i] = (capacity - avl)/capacity
        else:
            # 4.3 Count blocked requests
            blocked_requests += 1
            blocked_request[i] = 1

    # Visualization
    plt.figure(figsize=(12, 12))
    util = np.zeros(100)
    # key = utilization
    # Iterate over all keys in the dictionary
    for val in utilization.values():
        util = util + np.array(list(val.values()))

    util = util/15
    plt.plot(util)

    # Adding title and labels
    plt.title('Utilization over time')
    plt.xlabel('time')
    plt.ylabel('Mean Utilization')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()


    print(f"Number of blocked requests: {blocked_requests}")