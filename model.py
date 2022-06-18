import itertools
import random
import numpy
import networkx as nx
from collections import defaultdict

# Directions
NORTH = 'N'
EAST = 'E'
SOUTH = 'S'
WEST = 'W'


DIRECTIONS = [NORTH, EAST, SOUTH, WEST]

REVERSE = {
    NORTH: SOUTH,
    EAST: WEST,
    SOUTH: NORTH,
    WEST: EAST,
}

OFFSET = {
    NORTH: -32,
    EAST: 1,
    SOUTH: 32,
    WEST: -1,
}

# Masks
M_NORTH = 0x01
M_EAST  = 0x02
M_SOUTH = 0x04
M_WEST  = 0x08
M_ROBOT = 0x10

M_LOOKUP = {
    NORTH: M_NORTH,
    EAST: M_EAST,
    SOUTH: M_SOUTH,
    WEST: M_WEST,
}

# Colors
RED = 'R'
GREEN = 'G'
BLUE = 'B'
YELLOW = 'Y'

const = 4
COLORS = [RED, GREEN, BLUE, YELLOW]
COLOR_OFFSETS={BLUE:0,RED:1*const,GREEN:2*const,YELLOW:3*const,'goal':4*const}
# Shapes
CIRCLE = 'C'
TRIANGLE = 'T'
SQUARE = 'Q'
HEXAGON = 'H'

SHAPES = [CIRCLE, TRIANGLE, SQUARE, HEXAGON]

# Tokens
TOKENS = [''.join(token) for token in itertools.product(COLORS, SHAPES)]

# Quadrants
QUAD_1A = (
    'NW,N,N,N,NE,NW,N,N,'
    'W,S,X,X,X,X,SEYH,W,'
    'WE,NWGT,X,X,X,X,N,X,'
    'W,X,X,X,X,X,X,X,'
    'W,X,X,X,X,X,S,X,'
    'SW,X,X,X,X,X,NEBQ,W,'
    'NW,X,E,SWRC,X,X,X,S,'
    'W,X,X,N,X,X,E,NW'
)

QUAD_1B = (
    'NW,NE,NW,N,NS,N,N,N,'
    'W,S,X,E,NWRC,X,X,X,'
    'W,NEGT,W,X,X,X,X,X,'
    'W,X,X,X,X,X,SEYH,W,'
    'W,X,X,X,X,X,N,X,'
    'SW,X,X,X,X,X,X,X,'
    'NW,X,E,SWBQ,X,X,X,S,'
    'W,X,X,N,X,X,E,NW'
)

QUAD_2A = (
    'NW,N,N,NE,NW,N,N,N,'
    'W,X,X,X,X,E,SWBC,X,'
    'W,S,X,X,X,X,N,X,'
    'W,NEYT,W,X,X,S,X,X,'
    'W,X,X,X,E,NWGQ,X,X,'
    'W,X,SERH,W,X,X,X,X,'
    'SW,X,N,X,X,X,X,S,'
    'NW,X,X,X,X,X,E,NW'
)

QUAD_2B = (
    'NW,N,N,N,NE,NW,N,N,'
    'W,X,SERH,W,X,X,X,X,'
    'W,X,N,X,X,X,X,X,'
    'WE,SWGQ,X,X,X,X,S,X,'
    'SW,N,X,X,X,E,NWYT,X,'
    'NW,X,X,X,X,S,X,X,'
    'W,X,X,X,X,NEBC,W,S,'
    'W,X,X,X,X,X,E,NW'
)

QUAD_3A = (
    'NW,N,N,NE,NW,N,N,N,'
    'W,X,X,X,X,SEGH,W,X,'
    'WE,SWRQ,X,X,X,N,X,X,'
    'SW,N,X,X,X,X,S,X,'
    'NW,X,X,X,X,E,NWYC,X,'
    'W,X,S,X,X,X,X,X,'
    'W,X,NEBT,W,X,X,X,S,'
    'W,X,X,X,X,X,E,NW'
)

QUAD_3B = (
    'NW,N,NS,N,NE,NW,N,N,'
    'W,E,NWYC,X,X,X,X,X,'
    'W,X,X,X,X,X,X,X,'
    'W,X,X,X,X,E,SWBT,X,'
    'SW,X,X,X,S,X,N,X,'
    'NW,X,X,X,NERQ,W,X,X,'
    'W,SEGH,W,X,X,X,X,S,'
    'W,N,X,X,X,X,E,NW'
)

QUAD_4A = (
    'NW,N,N,NE,NW,N,N,N,'
    'W,X,X,X,X,X,X,X,'
    'W,X,X,X,X,SEBH,W,X,'
    'W,X,S,X,X,N,X,X,'
    'SW,X,NEGC,W,X,X,X,X,'
    'NW,S,X,X,X,X,E,SWRT,'
    'WE,NWYQ,X,X,X,X,X,NS,'
    'W,X,X,X,X,X,E,NW'
)

QUAD_4B = (
    'NW,N,N,NE,NW,N,N,N,'
    'WE,SWRT,X,X,X,X,S,X,'
    'W,N,X,X,X,X,NEGC,W,'
    'W,X,X,X,X,X,X,X,'
    'W,X,SEBH,W,X,X,X,S,'
    'SW,X,N,X,X,X,E,NWYQ,'
    'NW,X,X,X,X,X,X,S,'
    'W,X,X,X,X,X,E,NW'
)

QUADS = [
    (QUAD_1A, QUAD_1B),
    (QUAD_2A, QUAD_2B),
    (QUAD_3A, QUAD_3B),
    (QUAD_4A, QUAD_4B),
]

# Rotation
ROTATE_QUAD = [
    56, 48, 40, 32, 24, 16,  8,  0, 
    57, 49, 41, 33, 25, 17,  9,  1, 
    58, 50, 42, 34, 26, 18, 10,  2, 
    59, 51, 43, 35, 27, 19, 11,  3, 
    60, 52, 44, 36, 28, 20, 12,  4, 
    61, 53, 45, 37, 29, 21, 13,  5, 
    62, 54, 46, 38, 30, 22, 14,  6, 
    63, 55, 47, 39, 31, 23, 15,  7,
]

ROTATE_WALL = {
    NORTH: EAST,
    EAST: SOUTH,
    SOUTH: WEST,
    WEST: NORTH,
}

# Helper Functions
def idx(x, y, size=32):
    return y * size + x

def xy(index, size=32):
    x = index % size
    y = index // size
    return (x, y)

def rotate_quad(data, times=1):
    for i in range(times):
        result = [data[index] for index in ROTATE_QUAD]
        result = [''.join(ROTATE_WALL.get(c, c) for c in x) for x in result]
        data = result
    return data

def create_grid(quads=None):
    if quads is None:
        quads = [random.choice(pair) for pair in QUADS]
        random.shuffle(quads)
    quads = [quad.split(',') for quad in quads]
    quads = [rotate_quad(quads[i], i) for i in [0, 1, 3, 2]]
    result = [None for i in range(32 * 32)]
    for i, quad in enumerate(quads):
        dx, dy = xy(i, 2)
        for j, data in enumerate(quad):
            x, y = xy(j, 8)
            x += dx * 8
            y += dy * 8
            index = int(idx(int(x), int(y)))
            result[index] = data
    return result


def replace(input,char):
    if input == 'X':
        return char
    return "".join(set(input+char))

def draw_center(result, num_quads, cells_per_quad):
    size = num_quads * cells_per_quad
    print(size)
    id  = size * (size-1) //2 -1
    result[id] = 'NW'
    result[id+1] = 'NE'
    result[id+num_quads*cells_per_quad] = 'SW'
    result[id+num_quads*cells_per_quad+1] = 'SE'
    result[id-1] = replace(result[id-1],'E')
    result[id+2] = replace(result[id+2],'W')
    result[id+num_quads*cells_per_quad-1] = replace(result[id+num_quads*cells_per_quad-1],'E')
    result[id+num_quads*cells_per_quad+2] = replace(result[id+num_quads*cells_per_quad+2],'W')
    result[id-num_quads*cells_per_quad] = replace(result[id-num_quads*cells_per_quad],'S')
    result[id-num_quads*cells_per_quad+1] = replace(result[id-num_quads*cells_per_quad+1],'S')
    result[id+2*num_quads*cells_per_quad] = replace(result[id+2*num_quads*cells_per_quad],'N')
    result[id+2*num_quads*cells_per_quad+1] = replace(result[id+2*num_quads*cells_per_quad+1],'N')
    return  result

def create_random_grid():
    num_quads = 8
    cells_per_quad = 4
    obstacle_type_vertical = ('N','S')
    obstacle_type_horizontal = ('W','E')
    cells_per_row = num_quads*cells_per_quad
    result = []
    for i in range((cells_per_row)**2):
        if i%cells_per_row==0:
            result.append('W')
        elif i%cells_per_row==cells_per_row-1:
            result.append('E')
        elif i in range(cells_per_row):
            result.append('N')
        elif i in range(cells_per_row*(cells_per_row-1),cells_per_row**2):
            result.append('S')
        else:
            result.append('X')
    result[0],result[cells_per_row-1],result[cells_per_row*(cells_per_row-1)],result[cells_per_row**2-1] = 'NW','NE','SW','SE'


    obstacle_cells = []
    for i in range(num_quads):
        for j in range(num_quads):
                while True:
                    quad_x = random.choice(range(1,cells_per_quad-1))
                    quad_y = random.choice(range(1,cells_per_quad-1))
                    id = (i*cells_per_quad + quad_y)*num_quads*cells_per_quad + j*cells_per_quad + quad_x
                    if id in list(range(cells_per_row)) + list(range(0,cells_per_row*(cells_per_row-1),cells_per_row)) + list(range(cells_per_row-1,cells_per_row**2-1,cells_per_row)) + list(range(cells_per_row*(cells_per_row-1),cells_per_row**2-1)):
                        continue
                    else: break
                obstacle_cells.append(id)
                obstacle_horizontal = random.choice(obstacle_type_horizontal)
                obstacle_vertical = random.choice(obstacle_type_vertical)
                horizontal_id,horizontal_neigh_type = (id - 1,'E') if obstacle_horizontal == 'W' else (id + 1,'W')
                vertical_id,vertical_neigh_type = (id - num_quads * cells_per_quad,'S') if obstacle_vertical == 'N' else (id + num_quads*cells_per_quad,'N')
                result[id] = replace(result[id],obstacle_vertical+obstacle_horizontal)
                result[horizontal_id] = replace(result[horizontal_id],horizontal_neigh_type)
                result[vertical_id] = replace(result[vertical_id],vertical_neigh_type)
    side_coefs = [(('E','W'),0,0,1),
                  (('S','N'),0,1,0),
                  (('E','W'),num_quads*cells_per_quad * (num_quads*cells_per_quad-1),0,1),
                  (('S','N'),num_quads*cells_per_quad-1,1,0)]
    for side in side_coefs:
        start = side[1]
        for i in range(num_quads):

            j = random.choice(range(cells_per_quad-1))
            modulo = start+i*side[2]*cells_per_quad*cells_per_quad*num_quads + i*cells_per_quad*side[3]
            j_inc = j*side[3] + j*side[2]*cells_per_quad*num_quads
            id = modulo+j_inc
            if numpy.random.binomial(1,0.7):
                result[id]= replace(result[id],side[0][0])
                increment = id + 1*side[3] + num_quads*cells_per_quad*side[2]
                result[increment] = replace(result[increment],side[0][1])

    target = random.choice(obstacle_cells)
    target_robot = random.choice(COLORS)+'T'
    result = draw_center(result,num_quads,cells_per_quad)
    result[target]= result[target]+target_robot
    return cells_per_row, target_robot, target ,result

def to_mask(cell):
    result = 0
    for letter, mask in M_LOOKUP.items():
        if letter in cell:
            result |= mask
    return result

translate_dic = {"N":'u',
                 "E":'r',
                 "S":'b',
                 "W":'l'}
translate_dic_rev = {}
for k,v in translate_dic.items():
    translate_dic_rev[v]=k

def to_cell_code(cell):
    result = []
    for letter, mask in M_LOOKUP.items():
        if letter in cell:
            result.append(letter)
    return result

# Game
class Game(object):
    @staticmethod
    def hardest():
        quads = [QUAD_2B, QUAD_4B, QUAD_3B, QUAD_1B]
        robots = [226, 48, 43, 18]
        token = 'BT'
        return Game(quads=quads, robots=robots, token=token)
    def __init__(self, seed=None, quads=None, robots=None, token=None):
        if seed:
            random.seed(seed)
        if quads:
            self.grid = create_grid(quads)
        else: self.cells_per_row,token, self.token_id, self.grid = create_random_grid()
        if robots is None:
            self.robots = self.place_robots(token)
        else:
            self.robots = dict(zip(COLORS, robots))
        self.token = token or random.choice(TOKENS)
        self.moves = 0
        self.last = None
    def place_robots(self,token):
        result = {}
        used = set()
        for color in COLORS:
            while True:
                index = random.randint(0, 255)
                if index in (119, 120, 135, 136):
                    continue
                if self.grid[index][-2:] in TOKENS:
                    continue
                if index in used:
                    continue
                if index==self.token_id:
                    continue
                result[color] = index
                used.add(index)
                break
        return result
    def get_robot(self, index):
        for color, position in self.robots.items():
            if position == index:
                return color
        return None
    def can_move(self, color, direction):
        if self.last == (color, REVERSE[direction]):
            return False
        index = self.robots[color]
        if direction in self.grid[index]:
            return False
        new_index = index + OFFSET[direction]
        if new_index in self.robots.values():
            return False
        return True

    def compute_move_with_info(self, color, direction):
        index = self.robots[color]
        robots = self.robots.values()
        robot_blocked_at_index = (None,None)
        while True:
            if direction in self.grid[index]: # check if there is a wall in the direction
                break
            new_index = index + OFFSET[direction]
            if new_index in robots:
                for k,v in self.robots.items():  #TODO rewrite
                    if v == new_index:
                        block_color = k
                        break
                robot_blocked_at_index = (block_color,new_index)
                break
            index = new_index
        return index,robot_blocked_at_index

    def compute_move(self, color, direction):
        index = self.robots[color]
        robots = self.robots.values()
        while True:
            if direction in self.grid[index]:
                break
            new_index = index + OFFSET[direction]
            if new_index in robots:
                break
            index = new_index
        return index

    def do_move_unsafe(self, color, direction):
        start = self.robots[color]
        last = self.last
        #if last == (color, REVERSE[direction]):
        #    raise Exception
        end = self.compute_move(color, direction)
        self.robots[color] = end
        self.last = (color, direction)
        return (color, start, last)

    def do_move(self, color, direction):
        start = self.robots[color]
        last = self.last
        #if last == (color, REVERSE[direction]):
        #    raise Exception
        end = self.compute_move(color, direction)
        if start == end:
            raise Exception
        self.moves += 1
        self.robots[color] = end
        self.last = (color, direction)
        return (color, start, last)
    def undo_move(self, data):
        color, start, last = data
        self.moves -= 1
        self.robots[color] = start
        self.last = last
    def get_moves(self, colors=None):
        result = []
        colors = colors or COLORS
        for color in colors:
            for direction in DIRECTIONS:
                if self.can_move(color, direction):
                    result.append((color, direction))
        return result

    def could_move(self, index, direction):

        if direction in self.grid[index]:
            return False
        new_index = index + OFFSET[direction]
        if new_index in self.robots.values():
            return False
        return True

    def compute_hyp_move(self, start_index, direction):
        index = start_index
        robots = self.robots.values()
        while True:
            if direction in self.grid[index]:
                break
            new_index = index + OFFSET[direction]
            if new_index in robots:
                break
            index = new_index
        return index


    def get_neighbours(self,index):
        neighs = []
        for direction in DIRECTIONS:
            if self.could_move(index, direction):
                neighs.append(self.compute_hyp_move(index,direction))
        return neighs

    def get_goal_idx(self):
        return numpy.argmax([self.token in i for i in self.grid])

    def bfs(self,color):
        if color == 'goal':
            pos_dic = {}
            robot_pos = self.get_goal_idx()
            for col,ix in self.robots.items():
                pos_dic[col]=ix
                self.robots[col]=-10

        else:
            robot_pos = self.robots[color]
            self.robots[color] = -10 # temporarily remove robot from the board
        visited = []  # List to keep track of visited nodes.
        queue = []  # Initialize a queue
        paths = []


        def bfs0(visited, node):
            visited.append(node)
            queue.append(node)

            while queue:
                s = queue.pop(0)

                for neighbour in self.get_neighbours(s):
                    paths.append((s, neighbour))
                    if neighbour not in visited:
                        visited.append(neighbour)
                        queue.append(neighbour)

        bfs0(visited, robot_pos)
        if color=='goal':
            for col,ix in pos_dic.items():
                self.robots[col]=ix
        else:
            self.robots[color] = robot_pos # return the robot back
        return paths
    def get_edges_in_comp(self,comp,G):
        edges = []
        for (u,v) in G.edges:
            if (u in comp) and (v in comp):
                edges.append((u,v))
        return edges
    def get_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.grid)))
        pos_dic = {}
        for col,ix in self.robots.items():
            pos_dic[col]=ix
            self.robots[col]=-10 # temporarily remove robots from the grid
        reachable_nodes = []
        for cell in range(len(self.grid)):
            x,y = xy(cell)
            if (x==16 or x==15) and (y==16 or y==15):
                continue
            for neighbour in self.get_neighbours(cell):
                G.add_edge(cell,neighbour)
                reachable_nodes.append(neighbour)
        reachable_nodes = set(reachable_nodes)
        strongly_connected_comps = list(enumerate(filter(lambda x:len(x)>=2,nx.algorithms.components.strongly_connected_components(G))))
        relaxedG = nx.DiGraph(G)
        for e in relaxedG.edges():
            relaxedG[e[0]][e[1]]['weight'] = 1
        weight = 2

        sorted_comps = sorted(strongly_connected_comps,key=lambda x:-len(x[1]))
        edges_in_comps = []
        for i,comp in sorted_comps:
            if len(comp)<2:
                print('error')
            edges = self.get_edges_in_comp(comp,G) # ineffective, iterates over all edges in G and checks if they are in comp
            if len(edges)<2:
                print('error')
            edges_in_comps.append((i,edges))
        nodes_in_comps = []
        robots_in_comps = defaultdict(list)
        goal_idx = self.get_goal_idx()
        for i,comp in edges_in_comps:
            nodes = []
            #get nodes along edges
            for edge in comp:
                edge_nodes = self.get_nodes_along_edge(edge) # returns list of triples, one for each position on the edge. The last two elements contain possible stops to get to that position
                for inner_node in edge_nodes[1:-1]:
                    relaxedG.add_edge(edge_nodes[0][0],inner_node[0],weight=weight)
                    relaxedG.add_edge(edge_nodes[-1][0],inner_node[0],weight=weight)
                    for inner_node2 in edge_nodes[1:-1]:
                        relaxedG.add_edge(inner_node[0], inner_node2[0], weight=weight)
                        relaxedG.add_edge(inner_node2[0], inner_node[0], weight=weight)

                nodes += edge_nodes
            nodes_idxs = list(map(lambda x:x[0],nodes))
            if len(nodes)<2:
                print('error')
            #check if robot is in the component
            for k,v in pos_dic.items():
                if v in nodes_idxs:
                    robots_in_comps[k].append(i)
            # check if the goal is in the component
            if goal_idx in nodes_idxs:
                robots_in_comps['goal'].append(i)
            nodes_in_comps.append((i,nodes))


        id2comp = {}
        # create metagraph
        HG = nx.DiGraph()
        for i,comp in strongly_connected_comps:
            HG.add_node(i,nodes=comp)
            for node in comp:
                id2comp[node] = i
        for u,v in G.edges: # edges are directional.
            if u in id2comp.keys() and v in id2comp.keys():

                HG.add_edge(id2comp[u],id2comp[v])

        id2comp_full = defaultdict(list)

        for i,nodes in nodes_in_comps:
            for node in nodes:
                id2comp_full[node[0]].append((i,node[1:])) #each node will have the component and the possible neighs for stopping and stop positions
        mHG = nx.MultiDiGraph()
        mHG.add_nodes_from(HG)
        supportHG = nx.MultiDiGraph()
        supportHG.add_nodes_from(HG)
        for k,v in id2comp_full.items():
            v = set(v)
            if len(v)==1:
                continue
            else:
                for i in v:
                    for j in v:
                        if i[0]==j[0] or (i[0],j[0]) in HG.edges: # don't add selfloops or duplicate edges
                            continue
                        else:
                            # check not(mHG.get_edge_data(i[0],j[0]) and mHG.get_edge_data(i[0],j[0])[0]['node_id']==k) and 
                            if (i[1][0] in reachable_nodes) or (i[1][1] in reachable_nodes):
                               mHG.add_edge(i[0],j[0],possible=True,node_id=k,neighbours=i[1][:2],stop_cells=i[1][2:])




                            if i[1][0] in reachable_nodes and i[1][0] in id2comp.keys():
                                support_cell_comp = id2comp[i[1][0]]
                                supportHG.add_edge(support_cell_comp, j[0], support=True, stop_cell=i[1][3] ,support_cell=i[1][0], gate_cell=k)


                            if i[1][1] in reachable_nodes and i[1][1] in id2comp.keys():
                                support_cell_comp = id2comp[i[1][1]]
                                supportHG.add_edge(support_cell_comp, j[0], support=True, stop_cell=i[1][2], support_cell=i[1][1], gate_cell=k)

        for col, ix in pos_dic.items():
            self.robots[col] = ix
        return edges_in_comps,\
               nodes_in_comps,\
               mHG,\
               supportHG,\
               HG,\
               pos_dic,\
               goal_idx,\
               id2comp_full,\
               id2comp,\
               G,\
               relaxedG,\
               reachable_nodes,\
               robots_in_comps # in which components are the robots

    def get_nodes_along_edge(self,edge):
        # TODO add possible stops from along the edge
        if abs(edge[0]-edge[1]) >= 32: # vertical edge
            center = range(min(edge),max(edge)+1,32)
            before = list(range(min(edge)-32,max(edge)-31,32))
            after = list(range(min(edge)+32,max(edge)+33,32))
        else: # horizontal
            before = list(range(min(edge)-1,max(edge)))
            center = range(min(edge),max(edge)+1)
            after = list(range(min(edge) + 1, max(edge)+2))
        last = [max(edge) for i in range(len(center))]
        first = [min(edge) for i in range(len(center))]
        first[0],first[-1],last[0],last[-1],before[0],before[-1],after[0],after[-1]=-1,-1,-1,-1,-1,-1,-1,-1
        zipped = [tuple(t) for t in zip(center,before,after,first,last)]
        #zipped[0][1:],zipped[-1][1:] = [-1,-1,-1,-1],[-1,-1,-1,-1]
        return zipped

    def over(self):
        color = self.token[0]
        return self.token in self.grid[self.robots[color]]
    def key(self):
        return tuple(self.robots.itervalues())
    def search(self):
        max_depth = 1
        while True:
            #print 'Searching to depth:', max_depth
            result = self._search([], set(), 0, max_depth)
            if result is not None:
                return result
            max_depth += 1
    def _search(self, path, memo, depth, max_depth):
        if self.over():
            return list(path)
        if depth == max_depth:
            return None
        key = (depth, self.key())
        if key in memo:
            return None
        memo.add(key)
        if depth == max_depth - 1:
            colors = [self.token[0]]
        else:
            colors = None
        moves = self.get_moves(colors)
        for move in moves:
            data = self.do_move(*move)
            path.append(move)
            result = self._search(path, memo, depth + 1, max_depth)
            path.pop(-1)
            self.undo_move(data)
            if result:
                return result
        return None
    def export(self):
        grid = []
        token = None
        robots = [self.robots[color] for color in COLORS]
        for index, cell in enumerate(self.grid):
            mask = to_mask(cell)
            if index in robots:
                mask |= M_ROBOT
            grid.append(mask)
            if self.token in cell:
                token = index
        robot = COLORS.index(self.token[0])
        return {
            'grid': grid,
            'robot': robot,
            'token': token,
            'robots': robots,
        }

    def load_txt(self,filenum):
        with open(f'../export/{str(filenum)}.rr','r') as f:
            lines = f.readlines()
            dim = int(lines[0].rstrip())
            robots = {}
                
            for i in range(1,5):
                robot_info = lines[i].rstrip().split(" ")
                y,x = robot_info[1:]
                robots[robot_info[0]] = idx(int(x)-1,int(y)-1)
            target_info =  lines[5].rstrip().split(" ")
            target = target_info[0]+"T"
            y,x = target_info[1:]
            target_id = idx(int(x)-1,int(y)-1)
            obstacles = defaultdict(list)
            for line in lines[7:]:
                y,x,obst = line.rstrip().split(" ")
                cell_id = idx(int(x)-1,int(y)-1)
                obstacles[cell_id].append(translate_dic_rev[obst])
            obstacles[target_id].append(target)
            grid = []
            for i in range(dim*dim):
                if i in obstacles.keys():
                    grid.append("".join(obstacles[i]))
                else:
                    grid.append("X")
            return grid,robots,target,target_id

    def save_txt(self,filenum=0,):
        grid = []
        token = None
        robots = [(color,self.robots[color]) for color in COLORS]
        for index, cell in enumerate(self.grid):
            codes = to_cell_code(cell)
            if self.token in cell:
                token = index
            for code in codes:
                if code in ['N','W','E','S']:
                    x,y = xy(index,self.cells_per_row)
                    x,y = x+1,y+1
                    grid.append((x,y,translate_dic[code]))
        target_color = self.token[0]
        textdata = f'{self.cells_per_row}\n'
        for color, ix in robots:
            x,y = xy(ix,self.cells_per_row)
            x,y = x+1,y+1
            textdata += color + " " + str(y) + " " + str(x) + "\n"

        x,y = xy(token,self.cells_per_row)
        x,y = x+1,y+1
        textdata += target_color + " " + str(y) + " " + str(x) + "\n" + str(len(grid)) + "\n"

        for x,y,c in grid:
            textdata += str(y) + " " + str(x) + " " + c + "\n"
        with open(f'export/{str(filenum)}.rr','w') as f:
            f.write(textdata)

        print(textdata)
