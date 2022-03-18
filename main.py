import wx
import sys
import model
import ricochet
import pickle
import networkx as nx
import itertools
import copy
from collections import namedtuple

example = namedtuple('example', ['grid', 'robots', 'token', 'path', 'expl'])


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
        if abs(edge[0]-edge[1]) >= 16:
            positions = range(min(edge),max(edge)+1,16)
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

    def get_reachable(self,comp_id,G):
        reachable_comps = list(nx.single_source_shortest_path_length(G,comp_id).keys())
        return reachable_comps

    def get_possible_subgoals(self,S,T,mG):
        crossing_edges = nx.edge_boundary(mG,S,T,data=True)
        possible_subgoals = []
        for e in crossing_edges:
            possible_subgoals += e[2]['neighbours']
        return set(possible_subgoals)
    def get_reachable_cells(self,node_ids,G):
        reachable_comps = []
        for node_id in node_ids:
            reachable_comps += self.get_reachable(node_id,G)
        reachable_comps = set(reachable_comps)
        reachable_cells = []
        for c in reachable_comps:
            reachable_cells += G.nodes[c]['nodes']
        return set(reachable_cells)

    def show_direct_path(self):
        pass
    def show_intersections(self):
        pass

    def get_graph(self):
        target_robot = self.game.token[0]

        paths,nodes_in_comp,mHG,HG,robots_in_comps = self.game.get_graph()
        helper_robot_ids =list(itertools.chain(*[v   for k,v in robots_in_comps.items() if (k!=target_robot and k!='goal')]))
        reversedHG = HG.reverse()

        target_robot_reachable_comps = []
        for comp_id in robots_in_comps[target_robot]:
            target_robot_reachable_comps += self.get_reachable(comp_id,HG)
        target_robot_reachable_comps = set(target_robot_reachable_comps)
        if robots_in_comps['goal'][0] in target_robot_reachable_comps:
            print('no other robot needed')

        else:
            goal_reachable_comps = self.get_reachable(robots_in_comps['goal'][0],reversedHG)
            possible_subgoals = self.get_possible_subgoals(target_robot_reachable_comps,goal_reachable_comps,mHG)
            reachable_cells = self.get_reachable_cells(helper_robot_ids,HG)
            intersections = reachable_cells.intersection(possible_subgoals)
            if len(intersections)>0:
                print('reachable with one intersection')
                self.nodes[target_robot] = intersections

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
        self.Refresh()
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
            elif value=='W':
                with open('temp.pkl','wb') as f:
                    pickle.dump(f,self.game)
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
        size = min((w - p) / 16, (h - p) / 16)
        wall = size / 8
        ox = (w - size * 16) / 2
        oy = (h - size * 16) / 2
        dc.SetDeviceOrigin(ox, oy)
        dc.SetClippingRegion(0, 0, size * 16 + 1, size * 16 + 1)
        dc.SetBrush(wx.WHITE_BRUSH)
        dc.DrawRectangle(0, 0, size * 16 + 1, size * 16 + 1)
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
            for node in nodes:
               dc.SetPen(wx.Pen(colors[color], width, wx.DOT))
               x1, y1 = model.xy(node)
               x1, y1 = x1 * size + size / 2 + color_offset, y1 * size + size / 2 + color_offset
               dc.DrawCircle(x1,y1, size/10)
        for j in range(16):
            for i in range(16):
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
                if i in (7, 8) and j in (7, 8):
                    dc.SetBrush(wx.LIGHT_GREY_BRUSH)
                    dc.DrawRectangle(x, y, size + 1, size + 1)
                # robot
                if robot:
                    dc.SetBrush(wx.Brush(colors[robot]))
                    dc.DrawCircle(x + size / 2, y + size / 2, size / 3)
                # walls
                dc.SetBrush(wx.BLACK_BRUSH)
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
