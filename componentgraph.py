import networkx as nx
from PriorityQueue import Subgoal
class ComponentGraph():
    def __init__(self,normal,augmented,support,node2comps,node2comps_reachable):
        self.normal = normal
        self.augmented = augmented
        self.support = support
        self.reversed = normal.reverse()
        self.node2comps = node2comps
        self.node2comps_reachable = node2comps_reachable
    def get_bridges(self,from_comps, to_comps):
        bridges = set([(bridge[0],\
                    bridge[1],\
                    bridge[2]['node_id'],\
                    tuple(bridge[2]['neighbours']),\
                    tuple(bridge[2]['stop_cells'])) \
                   for bridge in nx.edge_boundary(self.augmented,from_comps,to_comps,data=True) if bridge[0] in from_comps and bridge[1] in to_comps])
        return bridges
    def get_supports(self,helper_comps,to_comps):
        supports = set([(support[0],\
                    support[1],\
                    support[2]['support_cell'],\
                    support[2]['gate_cell'],\
                    support[2]['stop_cell']) \
                   for support in nx.edge_boundary(self.support,helper_comps,to_comps,data=True) if support[0] in helper_comps and support[1] in to_comps])
        return supports

    def get_reachable_comps(self,node_idx,reverse=False):
        reachable_comps = []
        comps = [[self.node2comps_reachable[node_idx]]] if reverse else self.node2comps[node_idx]
        graph = self.reversed if reverse else self.normal
        for comp in comps:
            reachable_comps += list(nx.single_source_shortest_path_length(graph,comp[0]).keys())
        reachable_comps = set(reachable_comps)
        return reachable_comps

class Component():
    def __init__(self,component):
        pass
    def get_all_cells(self):
        pass
    def get_all_reachable_cells(self):
        pass
    def get_all_bridges(self):
        pass

class SubgoalEvaluator():
    def __init__(self,low_level_graph,relaxed_low_level_graph,target_idx,helpers_idx):
        self.low_level_graph = low_level_graph
        self.relaxed_low_level_graph = relaxed_low_level_graph
        self.all_pairs_distance = dict(nx.all_pairs_shortest_path(low_level_graph))
        self.target_distance_relaxed = dict(nx.single_source_dijkstra_path_length(relaxed_low_level_graph,target_idx))
        self.helpers_distances_relaxed = {helper_idx:dict(nx.single_source_dijkstra_path_length(relaxed_low_level_graph,helper_idx)) for helper_idx in helpers_idx}
        for i in range(32*32):
            for k,v in self.helpers_distances_relaxed.items():
                if i not in v:
                    self.helpers_distances_relaxed[k][i] = 10 ** 9
            if i not in self.target_distance_relaxed:
                self.target_distance_relaxed[i] = 10*9
    def filter_subgoals(self,bridges_target2goal,supports_helper2goal):
        target_gates = [bridge[2] for bridge in bridges_target2goal]
        supports_helper2goal = [[support for support in helper if support[3] not in target_gates] for helper in supports_helper2goal]
        return supports_helper2goal
    def validate(self,helpers_comps,goal_comps,helpers_idx,goal_idx,cg):
        for j,helper_comps in enumerate(helpers_comps):
            helper_supports = set([(support[0], \
                             support[1], \
                             support[2]['support_cell'], \
                             support[2]['gate_cell'], \
                             support[2]['stop_cell']) \
                            for support in nx.edge_boundary(cg.support, helper_comps, goal_comps, data=True) if
                            support[0] in helpers_comps and support[1] in goal_comps]) # warning, edge_boundary behaves differently than expected
            for support in helper_supports:
                helper2goal_dist = len(self.all_pairs_distance[helpers_idx[j]][support[2]]) - 1\
                + len(self.all_pairs_distance[support[3]][goal_idx]) - 1

    def evaluate_helper_subgoals(self,target_robot_idx,goal_idx,helpers_idx,supports_helper2goal,cost_so_far,previous_subgoals):
        helper_subgoals = []
        for j,helper_supports in enumerate(supports_helper2goal):
            for support in helper_supports: # iterate all support for one robot # (from_comp,to_comp,support_cell_id,gate_cell_id,stop_cell_id)
                helper2goal_dist = len(self.all_pairs_distance[helpers_idx[j]][support[2]]) - 1\
                + len(self.all_pairs_distance[support[3]][goal_idx]) - 1 # helper to support + gate_cell to goal distance
                if support[4] in self.all_pairs_distance[target_robot_idx].keys():
                    target2gate_dist = len(self.all_pairs_distance[target_robot_idx][support[4]])
                else:
                    target2gate_dist = 10**6
                target2gate_dist_relaxed = self.target_distance_relaxed[support[4]]+1 # to stop + 1 to gate
                subgoal = Subgoal(possible_cost=helper2goal_dist+target2gate_dist+cost_so_far,\
                           priority=helper2goal_dist+target2gate_dist_relaxed+cost_so_far,\
                           is_subgoal_for_target=True,\
                           cost_so_far=helper2goal_dist+cost_so_far,\
                           previous_subgoals=previous_subgoals,
                           info={'target_idx': target_robot_idx, \
                             'gate_cell_idx': support[3], \
                             'goal_idx' : support[4],\
                             'support_cell_idx': support[2], \
                             'stop_cell_idx': support[4], \
                             'helper_idxs': [v for i,v in enumerate(helpers_idx) if i!=j],\
                             'helper_idx': helpers_idx[j], \
                             'helper_id':j
                             })
                helper_subgoals.append(subgoal)
        return helper_subgoals

    def evaluate_target_subgoals(self,target_robot_idx,goal_idx,helpers_idx,bridges_target2goal,cost_so_far,previous_subgoals):
        target_subgoals = []
        for bridge in bridges_target2goal: # (from_comp_id, to_comp_id, gate_cell_id, tuple(support_ids),tuple(stop_cell_ids))
            for i in range(2): # 2 possible stop positions
                target2goal_dist = len(self.all_pairs_distance[target_robot_idx][bridge[4][1-i]])\
                                      + len(self.all_pairs_distance[bridge[2]][goal_idx]) - 1 # +1 is there to account for stop to gate step
                for j,helper_idx in enumerate(helpers_idx):
                    if bridge[3][i] in self.all_pairs_distance[helper_idx].keys():
                        helper2support_dist = len(self.all_pairs_distance[helper_idx][bridge[3][i]]) - 1
                    else:
                        helper2support_dist = 10**6 # TODO you probably don't need to append

                    helper2support_dist_relaxed = self.helpers_distances_relaxed[helper_idx][bridge[3][i]]
                    subgoal = Subgoal(possible_cost=target2goal_dist+helper2support_dist+cost_so_far,\
                               priority=target2goal_dist+helper2support_dist_relaxed+cost_so_far,\
                               is_subgoal_for_target=False,\
                               cost_so_far=target2goal_dist+cost_so_far,\
                               previous_subgoals=previous_subgoals,\
                               info={'target_idx': helper_idx, \
                               'gate_cell_idx': bridge[2],\
                               'goal_idx': bridge[3][i],\
                               'support_cell_idx': bridge[3][i],\
                               'stop_cell_idx': bridge[4][1 - i], \
                                'helper_idxs': [v for i, v in enumerate(helpers_idx) if i != j], \
                                'helper_idx':helper_idx,\
                               'helper_id':j
                               }
                                      )
                    target_subgoals.append(subgoal)
        return target_subgoals




