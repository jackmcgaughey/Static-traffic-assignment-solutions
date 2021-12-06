import numpy as np
import sys 
 
## the code constructing the graph and Dijkstra's algorithm is written by Alexey Klochay 
## https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
        
    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]

def dijkstra_algorithm(graph, start_node):
  unvisited_nodes = list(graph.get_nodes())
  shortest_path = {}
  previous_nodes = {}
  # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
  max_value = sys.maxsize
  for node in unvisited_nodes:
      shortest_path[node] = max_value
  # However, we initialize the starting node's value with 0   
  shortest_path[start_node] = 0
  while unvisited_nodes:
    current_min_node = None
    for node in unvisited_nodes: # Iterate over the nodes
        if current_min_node == None:
            current_min_node = node
        elif shortest_path[node] < shortest_path[current_min_node]:
            current_min_node = node
  # The code block below retrieves the current node's neighbors and updates their distances
    neighbors = graph.get_outgoing_edges(current_min_node)
    for neighbor in neighbors:
        tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
        if tentative_value < shortest_path[neighbor]:
            shortest_path[neighbor] = tentative_value
            # We also update the best path to the current node
            previous_nodes[neighbor] = current_min_node
    unvisited_nodes.remove(current_min_node)
  return previous_nodes, shortest_path

def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
 
    # Add the start node manually
    path.append(start_node)
    
    return list(reversed(path))

## end of Alexey Klochay's code

def get_times(x, node_adjacency):
  #refer to node_adjacency
  travel_times = np.zeros_like(x)
  i = 0
  for m in x:
    j = 0 
    for n in m:
      if n != 0 or node_adjacency[i][j] != 0:
        travel_times[i][j] = travel_time(n)
      j += 1 
    i += 1
  return travel_times

def get_shortest_paths(travel_times): 
  paths = [] 
  init_graph = {}
  for node in nodes:
    init_graph[node] = {}
  i = 0
  for m in travel_times:
    j=0
    for n in m:
      if n != 0:
        init_graph[i][j] = travel_times[i][j]
      j+=1
    i+=1
  graph = Graph(nodes, init_graph)
  for start, end, demand in od_pairs:
    previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=start)
    paths.append([print_result(previous_nodes,shortest_path, start, end),demand])
    #put the demand on the respective link
  time_for_paths = []
  for path,_ in paths:
    p_t = 0
    last = 0
    for k in range(len(path)):
      if k != 0:
        p_t += t[last][path[k]]
      last = path[k]
    time_for_paths.append(p_t)
  return paths, time_for_paths

def get_x_star(shortest_paths,x_hat):
  x_star = np.zeros_like(x_hat)
  for path, demand in shortest_paths:
    last = 0
    for i in range(len(path)):
      if i != 0:
        x_star[last][path[i]] += demand
      last = path[i]
  return x_star

def average_excess_cost(t,x,d):
  k = kappa #travel time on shortest path
  TSTT = 0 
  i = 0
  for m in t:
    j=0
    for n in m:
      if n != 0:
        TSTT += x[i][j] * n
      j+=1
    i+=1
  SPTT = np.dot(k,d)
  return (TSTT - SPTT) / np.dot(d,np.ones_like(d))

def get_x_hat(x_star, x_hat, iterator):
  lam = 1 / iterator
  new_x = np.dot(lam, x_star) + np.dot((1-lam), x_hat)
  return new_x

def travel_time(x):
  return 10 + x/100


if __name__ == "__main__":
    nodes = [0,1,2,3,4,5]
    iterator = 1
    epsilon = 1e-6
    od_pairs = [(0,2,5000),(1,3,10000)]
    D = [5000,10000]
    num_nodes = 6
    node_adjacency = np.array([
        [0,0,1,0,1,0],
        [0,0,0,1,1,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,1],
        [0,0,1,1,0,0]
    ])
    x = node_adjacency
    t = get_times(x, node_adjacency)
    shortest_paths, kappa = get_shortest_paths(t)
    x_star = get_x_star(shortest_paths,x)
    t = get_times(x_star, node_adjacency)
    shortest_paths, kappa = get_shortest_paths(t)
    x_hat = x_star
    aec = average_excess_cost(t,x_hat,D)
    while aec > epsilon:
        x_hat = get_x_hat(x_star, x_hat, iterator)
        t = get_times(x_hat, node_adjacency)
        shortest_paths, kappa = get_shortest_paths(t)
        aec = average_excess_cost(t,x_hat,D)
        shortest_paths, kappa = get_shortest_paths(t)
        x_star = get_x_star(shortest_paths,x)
        print(aec) 
        iterator += 1
    print(t)
