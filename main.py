import wx
import sys
import model
from componentgraph import ComponentGraph,SubgoalEvaluator
from PriorityQueue import PriorityQueue,Subgoal
import heapq as hq
import ricochet
import pickle
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import itertools
import copy
from collections import namedtuple,defaultdict
import matplotlib.pyplot as plt


example = namedtuple('example', ['grid', 'robots', 'token', 'path', 'expl'])

code2color = {'G': 'green', 'Y': 'yellow', 'B': 'blue', 'R': 'red'}

class View(wx.Panel):
    def __init__(self, parent, game):
        wx.Panel.__init__(self, parent, style=wx.WANTS_CHARS)
        self.game = game
        self.color = None
        self.path = None
        self.undo = []
        self.lines = []
        self.steps_with_info = []
        self.connected={}
        self.nodes={}
        self.evaluator = None
        self.last_loaded = 0 # TODO remove
        self.num_unsolved = 0
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
    def solve_with_expl(self):
        self.steps_with_info = []
        self.path = ricochet.search(self.game, self.callback)
        self.path_copy = copy.deepcopy(self.path)
        self.on_solve_sync()
        if len(self.steps_with_info)==0:
            return None
        explanations = self.compute_explanation()
        return explanations

    def create_data(self):
        num_examples = 100000

        for j in range(10):
            dataset = []
            for i in range(num_examples):
                self.path = None
                self.undo = []
                self.lines = []
                self.connected = {}
                self.nodes = {}
                self.game = model.Game()
                game_desc = copy.deepcopy(self.game)
                expl = self.solve_with_expl()
                if expl==None:
                    continue
                path_copy = copy.deepcopy(self.path)
                dataset.append(example(game_desc.grid,game_desc.robots,game_desc.token,path_copy,expl))
                print(i)
            with open(f'data2/dataset{j}.pkl','wb') as f:
                pickle.dump(dataset,f)

    def solve(self):
        self.steps_with_info = []
        robots_temp = copy.deepcopy(self.game.robots)
        #self.path = self.game.search()
        self.path = ricochet.search(self.game, self.callback)
        self.path_copy = copy.deepcopy(self.path)
        print (', '.join(''.join(move) for move in self.path))
        self.on_solve_sync()
        self.game.last = None
        self.game.moves = 0
        self.game.robots = robots_temp
        print(self.steps_with_info)
        explanations = self.compute_explanation()
        self.on_solve()

    def get_positions(self,edge):
        if abs(edge[0]-edge[1]) >= 32:
            positions = range(min(edge),max(edge)+1,32)
        else:
            positions = range(min(edge),max(edge)+1)
        return positions

    def check_if_avoidance(self,steps,explained,color,helped_color,explanations):
        last_color_occurance = 0

        helped_done = False
        help_started = False

        positions_of_main = []
        for i,step in enumerate(steps):
            if step[0] == color and explained[i]==0:
                last_color_occurance = i

                positions_of_main += self.get_positions((step[1],step[2]))
        positions_of_main = set(positions_of_main)
        for i in range(len(steps[:last_color_occurance+1])): #check
            reverse_ix = len(steps[:last_color_occurance]) - i #check
            step = steps[reverse_ix]
            if step[0]!=color and explained[reverse_ix]==0:
                intersection = step[1] in positions_of_main

                if helped_color != step[0] and intersection:
                    explanations.append((step[0],'Avoid',color,reverse_ix))
                    explanations,explained[0:reverse_ix+1] = self.check_if_avoidance(steps[:reverse_ix+1],explained[0:reverse_ix+1],step[0],None,explanations)
                if helped_done and intersection:
                    explanations.append((helped_color,'Avoid',color,reverse_ix))

                if helped_color == step[0] and intersection and not helped_done:
                    help_started = True
                if helped_color == step[0] and not intersection and help_started:
                    helped_done = True

            else:
                explained[reverse_ix] = 1
        return explanations, explained


    def explain_part(self,steps,explained):
        explanations = []
        helping_color,index = steps[-1][-1]
        helped_color = steps[-1][0]
        if helping_color!=None:
            explanations.append((helping_color,'Help',helped_color,len(steps)))
            explanations, explained = self.check_if_avoidance(steps,explained,helping_color,helped_color,explanations)
        else:
            explanations.append((helped_color,'get','goal',len(steps)))
            explanations,explained = self.check_if_avoidance(steps,explained,helped_color,helping_color,explanations)
        return explanations,explained



    def compute_explanation(self):
        explained = [0 for i in range(len(self.steps_with_info))]
        explanations = []
        meet_ts = []
        for i,step in enumerate(self.steps_with_info):
            if step[3][0]!=None:
                meet_ts.append(i)
        meet_ts.append(len(self.steps_with_info))
        for meet_t in meet_ts:
            explained_part,explained[:meet_t+1] = self.explain_part(self.steps_with_info[:meet_t+1],explained[:meet_t+1])
            explanations += explained_part
        #print(explanations)
        #print(explained)
        return explanations

    def callback(self, depth, nodes, inner, hits):
        print('Depth: %d, Nodes: %d (%d inner, %d hits)' % (depth, nodes, inner, hits))
    def on_solve(self):
        if not self.path:
            return
        self.do_move(*self.path.pop(0))
        self.Refresh()
        wx.CallLater(500, self.on_solve)
    def on_solve_sync(self):
        if not self.path_copy:
            return
        self.do_move_sync(*self.path_copy.pop(0))
        self.on_solve_sync()

    def do_move_sync(self, color, direction):
        start = self.game.robots[color]
        end,info = self.game.compute_move_with_info(color, direction)
        data = self.game.do_move(color, direction)
        self.steps_with_info.append((color,start,end,info))

    def do_move(self, color, direction):
        start = self.game.robots[color]
        end = self.game.compute_move(color, direction)
        data = self.game.do_move(color, direction)
        self.undo.append(data)
        self.lines.append((color, start, end))

    def undo_move(self):
        self.game.undo_move(self.undo.pop(-1))
        self.lines.pop(-1)
    def bfs(self):
        paths = self.game.bfs(model.BLUE)
        self.connected[model.BLUE] = paths
        paths = self.game.bfs(model.GREEN)
        self.connected[model.GREEN] = paths
        paths = self.game.bfs(model.YELLOW)
        self.connected[model.YELLOW] = paths
        paths = self.game.bfs(model.RED)
        self.connected[model.RED] = paths
        #paths = self.game.bfs('goal')
        #self.connected['goal'] = paths
        self.Refresh()



    def get_possible_subgoals(self,S,T,mG):
        crossing_edges = nx.edge_boundary(mG,S,T,data=True)
        possible_subgoals = []
        for e in crossing_edges:
            possible_subgoals += e[2]['neighbours']
        return set(possible_subgoals)


    def show_direct_path(self):
        pass
    def show_intersections(self):
        pass

    def find_solution(self,target_robot_idx,goal_idx,helpers_idx,cg,evaluator,cost_so_far,previous_subgoals):
        target_robot_comps = cg.get_reachable_comps(target_robot_idx)
        goal_comps = cg.get_reachable_comps(goal_idx,reverse=True)
        helpers_comps = [cg.get_reachable_comps(ix) for ix in helpers_idx]
        #return supports_helper2goal
        #print(len(bridges_target2goal))
        #print([len(sup) for sup in supports_helper2goal])
        #supports_helper2goal = evaluator.filter_subgoals(bridges_target2goal,supports_helper2goal)
        #print([len(sup) for sup in supports_helper2goal])

        bridges_target2goal = cg.get_bridges(target_robot_comps,goal_comps)  # (from_comp_id, to_comp_id, gate_cell_id, tuple(support_ids),tuple(stop_cell_ids))
        supports_helper2goal = [cg.get_supports(helper_comps, goal_comps) for helper_comps in helpers_comps]  # (from_comp,to_comp,support_cell_id,gate_cell_id,stop_cell_id)
        #evaluator.validate(helpers_comps,goal_comps,helpers_idx,goal_idx,cg)
        evaluated_target_subgoals = evaluator.evaluate_target_subgoals(target_robot_idx, goal_idx, helpers_idx,bridges_target2goal,cost_so_far,previous_subgoals)
        evaluated_helper_subgoals = evaluator.evaluate_helper_subgoals(target_robot_idx, goal_idx, helpers_idx,supports_helper2goal,cost_so_far,previous_subgoals)
        #print(len(evaluated_helper_subgoals),len(evaluated_target_subgoals))
        return evaluated_target_subgoals,evaluated_helper_subgoals


        # target_robot_reachable_comps = []
        # for comp_id in robots_in_comps[target_robot]:
        #     target_robot_reachable_comps += cg.get_reachable(comp_id)
        #
        # target_robot_reachable_comps = set(target_robot_reachable_comps)
        # if goal_id in target_robot_reachable_comps:
        #     print('no other robot needed')
        #
        # else:
        #     goal_reachable_comps = self.get_reachable(robots_in_comps['goal'][0],reversedHG)
        #     possible_subgoals = self.get_possible_subgoals(target_robot_reachable_comps,goal_reachable_comps,mHG)
        #     reachable_cells = self.get_reachable_cells(helper_robot_ids,HG)
        #     intersections = reachable_cells.intersection(possible_subgoals)
        #     if len(intersections)>0:
        #         print('reachable with one intersection')
        #         self.nodes[target_robot] = intersections

    def get_subgoal_chain(self,sol):
        target=sol.info['target_idx']
        goal=sol.info['goal_idx']
        target_stop = sol.info['stop_cell_idx']
        target_gate = sol.info['gate_cell_idx']
        subgoals = []

        if sol.is_subgoal_for_target:
            if sol.previous_subgoals is not None:
                super_goal,super_target,subgoals = self.get_subgoal_chain(sol.previous_subgoals)
                subgoals += [(super_target,target_stop),(target_stop,target_gate),(target_gate,super_goal)]


        else:
            if sol.previous_subgoals is not None:
                super_goal, super_target, subgoals = self.get_subgoal_chain(sol.previous_subgoals)
                subgoals += [(super_target, target_stop), (target_stop, target_gate), (target_gate, super_goal)]
        return goal,target,subgoals


    def verify(self,solution):
        subgoal_chain = self.get_subgoal_chain(solution)
        return True

    def get_incumbent_and_filter_subgoals(self,evaluated_target_subgoals,evaluated_helper_subgoals,incumbent,reachable_nodes):
        #TODO don't verify incumbent again
        solqeue = sorted([incumbent]+evaluated_target_subgoals+evaluated_helper_subgoals, key=lambda x: x.possible_cost)
        best = incumbent
        for sol in solqeue:
            if sol.possible_cost >= 10**6:
                break
            if self.verify(sol): # get the first valid sol
                best = sol
                break
        incumbent_cost = best.possible_cost
        filtered_target = [a for a in evaluated_target_subgoals if (a.priority < incumbent_cost and a.info['goal_idx'] in reachable_nodes)]
        filtered_helper = [a for a in evaluated_helper_subgoals if a.priority < incumbent_cost]
        #print("target:",len(evaluated_target_subgoals), "helper:",len(evaluated_helper_subgoals))
        #print(len(filtered_target), len(filtered_helper))
        return best, filtered_target+filtered_helper

    def flatten_subgoals(self,subgoal,sb_list):
        info = subgoal.info
        info['is_for_target'] = subgoal.is_subgoal_for_target
        prev = subgoal.previous_subgoals
        sb_list.append(info)
        if prev != None:
            sb_list = self.flatten_subgoals(prev,sb_list)
        return sb_list

    def close_traj(self,trajs):
        pass
    
    def reconstruct_plan(self,sb_list):
        id2col = {v: k for k, v in self.game.robots.items()}
        info = sb_list[-1]
        plan_graph = nx.DiGraph()
        trajcolors = [id2col[info['target_idx']]]
        trajcolor = trajcolors[-1]
        trajs = [[str(info['target_idx'])+trajcolor,str(info['goal_idx'])+trajcolor]]
        for i in range(len(sb_list)-1):
            info = sb_list[-(i+2)]
            trajcolor = trajcolors[-1]
            plan_graph.add_edge(str(info['stop_cell_idx']) + trajcolor, str(info['gate_cell_idx']) + trajcolor)
            trajs[-1].insert(1, str(info['gate_cell_idx']) + trajcolor)
            trajs[-1].insert(1, str(info['stop_cell_idx']) + trajcolor)
            if not info['is_for_target']:
                hlpcol = id2col[info['target_idx']] 
                trajs.append([str(info['target_idx'])+hlpcol,str(info['goal_idx'])+hlpcol])
                trajcolors.append(hlpcol)
                plan_graph.add_edge(str(info['goal_idx'])+hlpcol,str(info['gate_cell_idx'])+trajcolor)
            else:
                hlpcol = id2col[info['helper_idx']]
                plan_graph.add_edge(str(info['helper_idx'])+hlpcol, str(info['support_cell_idx'])+hlpcol)
                plan_graph.add_edge(str(info['support_cell_idx'])+hlpcol,str(info['gate_cell_idx'])+trajcolor)
                trajs.insert(0,[str(info['helper_idx'])+hlpcol,str(info['support_cell_idx'])+hlpcol])
        colormap = {}
        colors = {'G':'green','Y':'yellow','B':'blue','R':'red'}
        for traj in trajs:
            c = traj[0][-1] #id2col[traj[0]]
            colormap[traj[-1]]=colors[c]
            for i in range(len(traj)-1):
                colormap[traj[i]]=colors[c]
                if not plan_graph.has_edge(traj[i],traj[i+1]):
                    plan_graph.add_edge(traj[i],traj[i+1],edge_color='black')
        colmap = []
        for node in plan_graph:
            colmap.append(colormap[node])
        return trajs,plan_graph,colmap#,edgecolmap

    def get_path_between_subgoals(self,s1,s2):
        return [s1,s2]

    def get_plan(self,node,G,actions,is_support=False):
        node_col = node[-1]
        parrents = list(G.predecessors(node))
        assert len(parrents)<3
        predec = None
        parrent = None
        for p in parrents:
            if p[-1] == node_col:
                parrent = p
            else:
                predec = p
        if parrent != None:
            actions.insert(0,[parrent,node])
            actions = self.get_plan(parrent,G,actions)
        if predec != None:
            actions = self.get_plan(predec,G,actions,is_support=True)
        if is_support==True and parrent==None:
            actions.insert(0,[node,node]) # we need to maintain the info about non moving helpers too
        return actions

    def get_move_and_cells(self,out_n,in_n,color):
        start,end = min(out_n,in_n),max(out_n,in_n)
        if abs(out_n - in_n) >= 32: # TODO change constant, vertical case
            positions = list(range(out_n,in_n + 1, 32))
            if out_n < in_n: # go North
                return [('S',color,out_n)],positions
            else: # go south
                return [('N',color,out_n)],positions
        else:
            positions = list(range(out_n,in_n + 1))
            if out_n < in_n: # go West
                return  [('E',color,out_n)],positions
            else: # go east
                return [('W',color,out_n)],positions


    def get_moves_and_cells(self,microgoals,color):
        moves,cells = [],[]
        for i in range(len(microgoals)-1):
            ms,cs = self.get_move_and_cells(microgoals[i],microgoals[i+1],color)
            moves += ms
            cells += cs
        return moves, cells

    def get_possible_moves(self,color):
        possible_moves = []
        for direction in model.DIRECTIONS:
            if self.game.can_move(color, direction):
                possible_moves.append(direction)
        return possible_moves

    def avoid(self,cell,cells,color):
        print('avoid unused')
        possible = self.get_possible_moves(color)
        moves = []
        for dire in possible:
                neigh = self.game.compute_hyp_move(cell, dire)
                #print(dire,neigh)
                if neigh not in cells:
                    moves.append((dire,color,cell))
                    return moves
        print("no possible helper avoidance")
        return moves

    def avoid_and_get_back(self,cell,cells,color):
        print('avoid used')
        moves_out = []
        moves_back = []
        back_dir = {'N':'S',"S":"N","E":"W","W":"E"}
        possible = self.get_possible_moves(color)
        for dire in possible:
                neigh = self.game.compute_hyp_move(cell, dire)
                if neigh not in cells and cell in self.evaluator.all_pairs_distance[neigh]:
                    moves_out.append((dire,color,cell))
                    moves_back.append((back_dir[dire],color,neigh))
                    return moves_out, moves_back
        print("no possible helper avoidance")
        return moves_out,moves_back

    def get_pre_post(self,cells,current,active):
        cells = set(cells)
        not_used = set(self.game.robots.keys()) - active
        used = active - set([current])
        blocking_not_used = []
        blocking_used = []
        pre,post = [],[]
        for r in not_used:
            cell = self.game.robots[r]
            if cell in cells:
                pre += self.avoid(cell,cells,r)

        for r in used:
            cell = self.game.robots[r]
            if cell in cells:
                pre_,post_ = self.avoid_and_get_back(cell,cells,r)
                pre += pre_
                post += post_
        return pre,post

    def macro2micro(self,macro,active_robots):
        out_n,in_n = macro
        if out_n==in_n:
            return []
        current_robot = out_n[-1]
        out_n,in_n = int(out_n[:-1]),int(in_n[:-1])
        if in_n in self.evaluator.all_pairs_distance[out_n]:
            microgoals = self.evaluator.all_pairs_distance[out_n][in_n]
            moves,cells = self.get_moves_and_cells(microgoals,current_robot)
        else:
            moves,cells = self.get_move_and_cells(out_n,in_n,current_robot)

        pre,post = self.get_pre_post(cells,current_robot,active_robots)
        moves = pre + moves + post
        for move in moves:
            self.game.do_move_unsafe(move[1],move[0])
        return moves

    def get_active(self,moves):
        active = []
        for m in moves:
            active.append(m[0][-1])
        return set(active)

    def macros2micros(self,macromoves):
        micromoves = []
        for i,macromove in  enumerate(macromoves):
            active_robots = self.get_active(macromoves)
            micromoves += self.macro2micro(macromove,active_robots)
        return micromoves

    def graph2moves_naive(self,G):
        leaves = [x for x in G.nodes() if G.out_degree(x) == 0]
        assert len(leaves)==1
        # TODO check for cycles
        goal = leaves[0]
        actions = self.get_plan(goal,G,[])
        return actions

    def plot_solution(self,graph,colmap=None,labelsdict=None,name='Graph.png'):
        pos = graphviz_layout(graph)
        plt.figure(3, figsize=(12, 12))
        if labelsdict:
            nx.draw(graph, pos,labels = labelsdict, with_labels=True, node_size=150, node_color=colmap, font_size=6, font_weight='bold')
        elif colmap and labelsdict==None:
            nx.draw(graph, pos,with_labels = True,node_size=500, node_color=colmap, font_size=4,font_weight='bold')

        else:
            nx.draw(graph, pos, with_labels=True, node_size=500, font_size=4, font_weight='bold')
        plt.savefig(name)
        plt.clf()


    def get_move_and_cells2(self,out_n,in_n,color):
        start,end = min(out_n,in_n),max(out_n,in_n)
        if abs(out_n - in_n) >= 32: # TODO change constant, vertical case
            positions = list(range(start,end + 1, 32))
            if out_n < in_n: # go North
                return [('S',color,out_n)],positions
            else: # go south
                positions.reverse()
                return [('N',color,out_n)],positions
        else:
            positions = list(range(start,end + 1))
            if out_n < in_n: # go West
                return  [('E',color,out_n)],positions
            else: # go east
                positions.reverse()
                return [('W',color,out_n)],positions

    def get_moves_and_cells2(self,microgoals,color):
        moves,cells = [],[]
        for i in range(len(microgoals)-1):
            ms,cs = self.get_move_and_cells2(microgoals[i],microgoals[i+1],color)
            moves += ms
            cells.append(cs)
        return moves, cells
    def get_microedge(self,node,micro):
        out_n,in_n = int(node[0][:-1]),int(node[1][:-1])
        microgoals = self.evaluator.all_pairs_distance[out_n][in_n]
        color = node[0][-1]
        moves, cells = self.get_moves_and_cells2(microgoals, color)
        assert len(moves)==len(cells)==(len(microgoals)-1)
        edge_names = []
        for i in range(len(microgoals)-2):
            source,target = node[0]+node[1]+"__"+str(microgoals[i])+str(microgoals[i+1]), node[0]+node[1]+"__"+str(microgoals[i+1])+str(microgoals[i+2])
            micro.add_edge(source,target)
            micro.nodes[source]['cells']=cells[i]
            micro.nodes[source]['color']=code2color[node[0][-1]]
            micro.nodes[source]['move']=moves[i]
            micro.nodes[source]['name2']=(microgoals[i],microgoals[i+1])
            edge_names.append(source)
        
        if len(microgoals)==2:
            source = node[0]+node[1]+"__"+str(microgoals[0])+str(microgoals[1])
            micro.add_node(source)

        
        if len(microgoals)==1:
            last = node[0] + node[1] + "__" + str(microgoals[0])+str(microgoals[0])
            micro.add_node(last)
            micro.nodes[last]['cells']=[microgoals[0]]
            micro.nodes[last]['color']=code2color[node[0][-1]]
            micro.nodes[last]['move']=node
            micro.nodes[last]['name2']=(microgoals[0],microgoals[0])

        else: 
            last = node[0] + node[1] + "__" + str(microgoals[-2])+str(microgoals[-1])
            micro.nodes[last]['cells']=cells[-1]
            micro.nodes[last]['color']=code2color[node[0][-1]]
            micro.nodes[last]['move']=moves[-1]
            micro.nodes[last]['name2']=(microgoals[-2],microgoals[-1])
        edge_names.append(last)
        return edge_names


    def get_micrograph(self,goal,macro,micro,prev):
        parrents = list(macro.predecessors(goal))
        assert len(parrents)<3
        if len(parrents)!=2:
            chain = self.get_microedge(goal, micro)
            micro.add_edge(chain[-1],prev)
            if len(parrents)==1:
                chain, micro = self.get_micrograph(parrents[0], macro, micro, chain[0])

        else:
            micro.add_edge(goal,prev)
            out_n,in_n = int(goal[0][:-1]),int(goal[1][:-1])
            color = goal[0][-1]
            move,cells = self.get_move_and_cells2(out_n,in_n,color)
            micro.nodes[goal]['color']=code2color[color]
            micro.nodes[goal]['name2']=(out_n,in_n)
            micro.nodes[goal]['move'] = move[0]
            micro.nodes[goal]['cells'] = cells
            micro.nodes[goal]['junction'] = True
            for p in parrents:
                chain,micro = self.get_micrograph(p,macro,micro,goal)
        return chain,micro

    def are_in_same_chain(self,G,i,j):
        if i==j:
            return True
        i_desc = nx.descendants(G, i)
        j_desc = nx.descendants(G,j)
        if i in j_desc or j in i_desc:
            return True
        else:
            return False


    def add_blocking_edges(self,graph):
        last_dic = defaultdict(list)
        cells_dic = defaultdict(list)
        for k,v in graph.nodes.items():
            cells = v['cells']
            last = cells[-1]
            last_dic[last].append(k)
            for c in cells[1:]:
                cells_dic[c].append(k)
        for k,v in last_dic.items():
            if k in cells_dic:
                for i in cells_dic[k]:
                    for j in v:
                        if self.are_in_same_chain(graph,i,j):
                            continue
                        graph.add_edge(i,j,constraint=True)
        return graph

    def add_starting(self,graph):
        sources = [x for x in graph.nodes() if graph.in_degree(x) == 0]
        for s in sources:
            start = s+"_start"
            sdata = graph.nodes[s]
            graph.add_edge(start,s)
            graph.nodes[start]['color']=sdata['color']
            graph.nodes[start]['name2']=sdata['name2'][0]
            graph.nodes[start]['cells']=[sdata['cells'][0]]
        return graph

    def macrograph2micrograph(self,macro):
        leaves = [x for x in macro.nodes() if macro.out_degree(x) == 0]
        assert len(leaves) == 1
        # TODO check for cycles
        goal = leaves[0]
        micro = nx.DiGraph()
        chain,micro = self.get_micrograph(goal, macro, micro,'end')
        micro.remove_node('end')
        #micro = self.add_starting(micro)
        #micro = self.add_blocking_edges(micro)

        colmap = [micro.nodes[node]['color'] for node in micro.nodes]
        namemap = {node:micro.nodes[node]['name2'] for node in micro.nodes}
        return micro,colmap,namemap

    def edges2nodes(self,nodegraph):
        new_edges = []
        edges2remove = []
        G = nodegraph.copy()
        for edge in G.edges():
            if edge[0][-1]!=edge[1][-1]:
                parrents = list(G.predecessors(edge[1]))
                for p in parrents:
                    if p!=edge[0]:
                        the_other_parrent=p
                new_edges.append((edge[0],the_other_parrent))
                edges2remove.append((edge[0],edge[1]))
        for e in new_edges:
            G.add_edge(e[0],e[1])
        for e in edges2remove:
            G.remove_edge(e[0],e[1])
        eG = nx.line_graph(G)
        nodes2remove = []
        edges2remove = []
        edge2add = []
        for n in eG.nodes:
            if n[0][-1]!=n[1][-1]:
                par = list(eG.predecessors(n))
                child = list(eG.successors(n))
                assert len(child)==1
                assert len(par)<2
                if len(par)==0:
                    nodes2remove.append(n)
                    edge2add+=[((n[0],n[0]),child[0])]
                    edges2remove.append((n,child[0]))
                else:
                    edges2remove += [(par[0],n),(n,child[0])]
                    edge2add.append((par[0],child[0]))
                    nodes2remove.append(n)
                print(edges2remove)


        for e in edges2remove:
            print(e)
            eG.remove_edge(e[0],e[1])
        for n in nodes2remove:
            eG.remove_node(n)
        for e in edge2add:
            eG.add_edge(e[0],e[1])
        return eG



    def next_free(self,graph,s):
        free = None
        sources = set([x for x in graph.nodes() if graph.in_degree(x) == 0])-set([s])
        blocked = []
        for s in sources:
            for e in graph.out_edges(s):
                if 'constraint' in graph.edges[e]:
                    blocked.append(e[1])
        while True:
            if 'junction' in graph.nodes[s] or s in blocked:
                break
            if len(graph.out_edges(s))==0:
                break
            for edge in graph.out_edges(s):
                if 'constraint' in graph.edges[edge]:
                    continue
                else:
                    next=edge[1]
                    is_free = True
                    for e in graph.in_edges(next):
                        if 'constraint' in graph.edges[e]:
                            is_free = False
                    if is_free:
                        free = next
                    s = next
                    break
        return free

    def unitprop(self,graph,sources):
        upchains  = []
        while True:  # unitprop
            new_sources = []
            for i, s in enumerate(sources):
                last = self.next_free(graph, s)
                if last != None:
                    upchain = nx.ancestors(graph, last)
                    upchains.append([graph.nodes[n]['name2'] for n in upchain])
                    for n in upchain:
                        graph.remove_node(n)
                else:
                    last = s
                new_sources.append(last)
            if sources == new_sources:
                break
            sources = new_sources
        return upchains,graph,sources

    def find_schedule(self,graph):
        graph = graph.copy()
        sources = [x for x in graph.nodes() if graph.in_degree(x) == 0]
        upchains,graph,sources = self.unitprop(graph,sources)
        print(upchains)
        return upchains

    def save_partial_sol(self,graph,ix):
        sol_str = ""
        if type(graph)!=list:
            move_translate_dic = {"N":'u',
                     "E":'r',
                     "S":'d',
                     "W":'l'}
            dd = defaultdict(lambda:defaultdict(list))
            colors_used = []
            for k,v in graph.nodes.items():
                #cell = v['name2'][0]
                cells = v['cells']
                move = v['move'][0]
                if len(v['move'])==2: # some moves correspond to staying..the move is set to the name of the node. i.e. (27R,27R)
                    continue
                dir = move_translate_dic[move]
                #x,y =model.xy(cell)
                #print(y+1,x+1)
                #print(dir)
                #print("*"*10)
                col = v['move'][1]
                for c in cells[:-1]:
                    dd[c][col].append(dir)
                colors_used.append(col)
            colors_used = set(colors_used)
            non_moves = []
            for i in range(32*32):
                for c in model.COLORS:
                    if c not in colors_used:
                        continue
                    for m in move_translate_dic.values():
                        if i in dd and c in dd[i] and m in dd[i][c]:
                            continue
                        else:
                            x,y = model.xy(i)
                            x,y = x+1,y+1
                            sol_str += f"{c} {y} {x} {m}\n"
        with open(f"../partial_sol/{ix}.partial",'w') as f:
            f.write(sol_str)

    def check_solution(self,sol):
        if sol.possible_cost > 10**6:
            #print('no solution found')
            self.num_unsolved += 1
            return []
        sb_list = self.flatten_subgoals(sol,[])
        trajs,graph,colmap = self.reconstruct_plan(sb_list)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        egraph = self.edges2nodes(graph)
        #self.plot_solution(graph,colmap)
        #self.plot_solution(egraph,name='graph2.png')
        microegraph,microcolmap,namemap = self.macrograph2micrograph(egraph)
        self.plot_solution(microegraph,microcolmap,namemap,name='graph3.png')
        plan = self.find_schedule(microegraph)
        return microegraph
        #macromoves = self.graph2moves_naive(graph)
        #print(macromoves)
        #micromoves = self.macros2micros(macromoves)
        #return micromoves

    def get_graph(self):
        target_robot = self.game.token[0]
        paths,nodes_in_comp,\
        augmented_graph,\
        support_graph,\
        normal_graph,\
        pos_dic,\
        goal_idx,\
        id2comps,\
        id2comp_reachable,\
        low_level_graph,\
        relaxed_low_level_graph,\
        reachable_nodes,\
        robots_in_comps = self.game.get_graph()

        target_robot_idx = pos_dic[target_robot]
        helpers_idxs = [v for k,v in pos_dic.items() if k!=target_robot]
        #helpers_cols = [k for k,v in pos_dic.items() if k!=target_robot]
        cg = ComponentGraph(normal_graph, augmented_graph,support_graph,id2comps,id2comp_reachable)
        evaluator = SubgoalEvaluator(low_level_graph,relaxed_low_level_graph,target_robot_idx,helpers_idxs)
        self.evaluator = evaluator
        #helper_robot_ids =list(itertools.chain(*[v   for k,v in robots_in_comps.items() if (k!=target_robot and k!='goal')])) #ids of components where the robots are
        #goal_id = robots_in_comps['goal'][0]
        first_bound = (len(evaluator.all_pairs_distance[target_robot_idx][goal_idx])-1) if goal_idx in evaluator.all_pairs_distance[target_robot_idx] else 10**7
        finalgoal = Subgoal(priority=0,possible_cost=first_bound,is_subgoal_for_target=True,cost_so_far=0, previous_subgoals=None,\
                              info={'target_idx': target_robot_idx, \
                             'goal_idx': goal_idx, \
                              'stop_cell_idx':None,\
                              'gate_cell_idx':None,\
                             'helper_idxs': helpers_idxs, \
                             })
        frontier = PriorityQueue()
        frontier.put(finalgoal)
        incumbent = finalgoal
        for i in range(5000):
            if len(frontier.elements)==0:
                #print(f"no more subgoals on {i}th iteration")
                break
            #else: print('frontier size:',len(frontier.elements))
            current = frontier.get()
            target_robot_idx = current.info['target_idx'] # TODO move inside the find solution
            goal_idx = current.info['goal_idx']
            helpers_idxs = current.info['helper_idxs']
            cost_so_far = current.cost_so_far

            # if current == goal: check if best priority == best

            evaluated_target_subgoals, evaluated_helper_subgoals = self.find_solution(target_robot_idx,goal_idx,helpers_idxs,cg,evaluator,cost_so_far,current)
            incumbent,filtered = self.get_incumbent_and_filter_subgoals(evaluated_target_subgoals,evaluated_helper_subgoals,incumbent,reachable_nodes)
            frontier.merge(filtered)
        #print('final solution',incumbent)
        micromoves = self.check_solution(incumbent)
        return micromoves
        #for next in graph.neighbors(current):
        #    new_cost = cost_so_far[current] + graph.cost(current, next)
        #    if next not in cost_so_far or new_cost < cost_so_far[next]:
        #        cost_so_far[next] = new_cost
        #        priority = new_cost + heuristic(next, goal)
        #        frontier.put(next, priority)
        #        came_from[next] = current

        #for j,helper in enumerate(bridges):
        #    intersections = [b[3] for b in helper]
        #    self.nodes[helpers_cols[j]] = intersections



        # comp_ids = []
        # for i, color in enumerate(model.COLORS):
        #     self.connected[color] = paths[i][1] # 0 is the index of the component
        #     comp_ids.append(paths[i][0])
        # intersections_per_path = []
        # for i in comp_ids:
        #     intersections = []
        #     for j in comp_ids:
        #         if i < j and mHG.has_edge(i,j):
        #             for k,edge in mHG.get_edge_data(i, j).items():
        #                 if edge['possible']==True:
        #                     intersections.append(edge['neighbours'][1])
        #     intersections_per_path.append(intersections)
        # for i, color in enumerate(model.COLORS):
        #         self.nodes[color] = intersections_per_path[i] # one item in intersections_per_path is a list of intersection nodes

    def save_sol(self,micromoves, i):
        target_robot = self.game.token[0]
        move_translate_dic = {"N":'u',
                 "E":'r',
                 "S":'d',
                 "W":'l'}
        #if self.game.robots[target_robot] != self.game.token_id:
            #self.num_unsolved += 1
            #print('unsolved')

        print(self.num_unsolved)
        #print("problem num: ",i)
        #print("moves: ",micromoves)
        if i!=0:
            print("unsolved ration: ",self.num_unsolved/i)
        num_moves = len(micromoves)
        num_moves_str = str(num_moves)+"\n"
        string_parts = [num_moves_str]
        for m in micromoves:
            color = m[1]
            dire = move_translate_dic[m[0]]
            string_parts.append(f"{color} {dire}\n")
        #string = "\n".join(string_parts)
        with open(f"../solutions/naive_avoidance/{i}.rr.sol",'w') as f:
            f.writelines(string_parts)


    def on_size(self, event):
        event.Skip()
        self.Refresh()
    def on_key_down(self, event):
        code = event.GetKeyCode()
        if code == wx.WXK_ESCAPE:
            self.GetParent().Close()
        elif code >= 32 and code < 128:
            value = chr(code)
            if value in model.COLORS:
                self.color = value
            elif value == 'S':
                self.solve()
            elif value == 'P':
                #costs = []
                #self.game.grid, self.game.robots, self.game.token, self.game.token_id = self.game.load_txt(i)
                #cost_obtained = self.get_graph()
                import time
                for i in range(100):

                    print(str(i)+"*"*15)
                    self.game.grid,self.game.robots,self.game.token,self.game.token_id = self.game.load_txt(i)
                    self.Refresh()
                    micromoves = self.get_graph()
                    #self.save_partial_sol(micromoves, i)
                    #self.save_sol(micromoves,i)
                    #costs.append(cost_obtained)
                #with open('results3-5.pkl','wb') as f:
                #    pickle.dump(costs,f)
                self.Refresh()
            elif value == 'V': # load saved games 1 by 1
                #self.last_loaded=39
                self.game.grid, self.game.robots, self.game.token, self.game.token_id = self.game.load_txt(self.last_loaded)
                self.Refresh()
                micromoves = self.get_graph()

                #self.save_partial_sol(micromoves,self.last_loaded)
                #self.save_sol(micromoves,self.last_loaded)
                self.last_loaded += 1
            elif value=='I':
                self.game.grid, self.game.robots, self.game.token, self.game.token_id = self.game.load_txt(self.last_loaded-1)
                self.Refresh()
                self.last_loaded += 1
            elif value=='W':
                with open('temp.pkl','wb') as f:
                    pickle.dump(self.game,f)
            elif value=='L':
                with open('temp.pkl','rb') as f:
                    self.game= pickle.load(f)
                self.Refresh()
            elif value=='F':
                self.get_graph()
            elif value == 'M':
                game = None
                for i in range(10000):
                    self.game = model.Game()

                    #if i==11:
                    #    with open('temp.pkl', 'wb') as f:
                    #        pickle.dump(self.game,f)
                    self.game.save_txt(i)
            elif value == 'U' and self.undo:
                self.undo_move()
                self.Refresh()
            elif value == 'A':
                self.bfs()
                self.Refresh()
            elif value == 'Q':
                self.create_data()
            elif value == 'N':
                self.path = None
                self.undo = []
                self.lines = []
                self.connected = {}
                self.nodes = {}
                self.game = model.Game()
                self.Refresh()
        elif self.color:
            lookup = {
                wx.WXK_UP: model.NORTH,
                wx.WXK_RIGHT: model.EAST,
                wx.WXK_DOWN: model.SOUTH,
                wx.WXK_LEFT: model.WEST,
            }
            if code in lookup:
                color = self.color
                direction = lookup[code]
                try:
                    self.do_move(color, direction)
                except Exception:
                    pass
                self.Refresh()
    def on_paint(self, event):
        colors = {
            model.RED: wx.Colour(178, 34, 34),
            model.GREEN: wx.Colour(50, 205, 50),
            model.BLUE: wx.Colour(65, 105, 225),
            model.YELLOW: wx.Colour(255, 215, 0),
            'goal': wx.Colour(220, 70, 50),
        }
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.LIGHT_GREY_BRUSH)
        dc.Clear()
        w, h = self.GetClientSize()
        p = 40
        size = min((w - p) / 32, (h - p) / 32)
        wall = size / 6
        ox = (w - size * 32) / 2
        oy = (h - size * 32) / 2
        dc.SetDeviceOrigin(ox, oy)
        dc.SetClippingRegion(0, 0, size * 32 + 1, size * 32 + 1)
        dc.SetBrush(wx.WHITE_BRUSH)
        dc.DrawRectangle(0, 0, size * 32 + 1, size * 32 + 1)
        for color, start, end in self.lines:
            dc.SetPen(wx.Pen(colors[color], 3, wx.DOT))
            x1, y1 = model.xy(start)
            x1, y1 = x1 * size + size / 2, y1 * size + size / 2
            x2, y2 = model.xy(end)
            x2, y2 = x2 * size + size / 2, y2 * size + size / 2
            dc.DrawLine(x1, y1, x2, y2)

        for color,paths in self.connected.items():
            color_offset=model.COLOR_OFFSETS[color]
            width = 4 if color=='goal' else 4
            for (start,end) in paths:
               dc.SetPen(wx.Pen(colors[color], width, wx.DOT))
               x1, y1 = model.xy(start)
               x1, y1 = x1 * size + size / 2 + color_offset, y1 * size + size / 2 + color_offset
               x2, y2 = model.xy(end)
               x2, y2 = x2 * size + size / 2 + color_offset, y2 * size + size / 2 + color_offset
               dc.DrawLine(x1, y1, x2, y2)
        for color,nodes in self.nodes.items():
            color_offset = 0 #model.COLOR_OFFSETS[color]
            width = 2
            print(color)
            for node in nodes:
               dc.SetPen(wx.Pen(colors[color], width, wx.DOT))
               x1, y1 = model.xy(node)
               x1, y1 = x1 * size + size / 2 + color_offset, y1 * size + size / 2 + color_offset
               dc.DrawCircle(x1,y1, size/10)
        font = wx.Font(pointSize = 7, family = wx.DEFAULT,
               style = wx.NORMAL, weight = wx.NORMAL,
               faceName = 'Consolas')
        dc.SetFont(font)
        for j in range(32):
            for i in range(32):
                x = i * size
                y = j * size
                index = model.idx(i, j)
                cell  = self.game.grid[index]
                robot = self.game.get_robot(index)
                # border
                dc.SetPen(wx.BLACK_PEN)
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRectangle(x, y, size + 1, size + 1)
                # token
                if self.game.token in cell:
                    dc.SetBrush(wx.Brush(colors[self.game.token[0]]))
                    dc.DrawRectangle(x, y, size + 1, size + 1)
                if i in (15, 16) and j in (15, 16):
                    dc.SetBrush(wx.LIGHT_GREY_BRUSH)
                    dc.DrawRectangle(x, y, size + 1, size + 1)
                # robot
                if robot:
                    dc.SetBrush(wx.Brush(colors[robot]))
                    dc.DrawCircle(x + size / 2, y + size / 2, size / 3)
                # walls
                dc.SetBrush(wx.BLACK_BRUSH)
                dc.DrawText(str(32*j+i), x, y)
                if model.NORTH in cell:

                    dc.DrawRectangle(x, y, size + 1, wall)
                    dc.DrawCircle(x, y, wall - 1)
                    dc.DrawCircle(x + size, y, wall - 1)
                if model.EAST in cell:
                    dc.DrawRectangle(x + size + 1, y, -wall, size + 1)
                    dc.DrawCircle(x + size, y, wall - 1)
                    dc.DrawCircle(x + size, y + size, wall - 1)
                if model.SOUTH in cell:
                    dc.DrawRectangle(x, y + size + 1, size + 1, -wall)
                    dc.DrawCircle(x, y + size, wall - 1)
                    dc.DrawCircle(x + size, y + size, wall - 1)
                if model.WEST in cell:
                    dc.DrawCircle(x, y, wall - 1)
                    dc.DrawCircle(x, y + size, wall - 1)
                    dc.DrawRectangle(x, y, wall, size + 1)
        dc.DrawText(str(self.game.moves), wall + 1, wall + 1)

class Frame(wx.Frame):
    def __init__(self, seed=None):
        wx.Frame.__init__(self, None, -1, 'Ricochet Robot!')
        game = model.Game(seed)
        #game = model.Game.hardest()
        self.view = View(self, game)
        self.view.SetSize((800, 800))
        self.Fit()

def main():
    app = wx.App(False)
    seed = int(sys.argv[1]) if len(sys.argv) == 2 else None
    frame = Frame(seed)
    frame.Center()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
