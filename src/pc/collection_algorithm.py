from collections import defaultdict
import heapq
class Graph:

    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self,u,v):
        self.graph[u].append(v)


class node:

    def __init__(self,value,distance_from_robot=0,distance_from_closest_node=0):
        self.value = value
        self.distance_from_robot = distance_from_robot
        self.distance_from_closest_node = distance_from_closest_node

def calculate_node(stack):
    sorted_stack = []
    for i in stack:
        real_value = stack[i].value - (max(i.distance_from_closest_node, i.distance_from_robot)-min(i.distance_from_closest_node, i.distance_from_robot))
        sorted_stack.append(real_value)


def get_distance(node_a,node_b):
    distance = node_a-node_b


def get_heuristic(current, goal)->float:
    distance = current+goal
    return distance

def A_star(graph, frontier, start_node,goal_node):
    heapq.heappush(frontier,(0,0,start_node))
    visited = set()
    backtracked = {}
    while frontier:
        current_f, current_g, current_node = heapq.heappop(frontier)
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor in graph[current_node]:
            new_g = current_g - get_distance(current_node, neighbor)
            h = get_heuristic(neighbor,goal_node)
            f = new_g+h
            backtracked[neighbor] = current_node
            heapq.heappush(frontier,(f,new_g,neighbor))
