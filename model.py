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
    NORTH: -16,
    EAST: 1,
    SOUTH: 16,
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
def idx(x, y, size=16):
    return y * size + x

def xy(index, size=16):
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
    result = [None for i in range(16 * 16)]
    for i, quad in enumerate(quads):
        dx, dy = xy(i, 2)
        for j, data in enumerate(quad):
            x, y = xy(j, 8)
            x += dx * 8
            y += dy * 8
            index = int(idx(int(x), int(y)))
            result[index] = data
    return result

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
        self.grid = create_grid(quads)
        if robots is None:
            self.robots = self.place_robots()
        else:
            self.robots = dict(zip(COLORS, robots))
        self.token = token or random.choice(TOKENS)
        self.moves = 0
        self.last = None
    def place_robots(self):
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
            self.robots[col]=-10
        for cell in range(len(self.grid)):
            x,y = xy(cell)
            if (x==8 or x==7) and (y==8 or x==7):
                continue
            for neighbour in self.get_neighbours(cell):
                G.add_edge(cell,neighbour)
        strongly_connected_comps = list(enumerate(filter(lambda x:len(x)>=2,nx.algorithms.components.strongly_connected_components(G))))

        id2comp = {}
        HG = nx.DiGraph()
        for i,comp in strongly_connected_comps:
            HG.add_node(i,nodes=comp)
            for node in comp:
                id2comp[node] = i
        for u,v in G.edges:
            if u in id2comp.keys() and v in id2comp.keys():

                HG.add_edge(id2comp[u],id2comp[v])

        sorted_comps = sorted(strongly_connected_comps,key=lambda x:-len(x[1]))
        edges_in_comps = []
        for i,comp in sorted_comps:
            if len(comp)<2:
                print('error')
            edges = self.get_edges_in_comp(comp,G)
            if len(edges)<2:
                print('error')
            edges_in_comps.append((i,edges))
        nodes_in_comps = []
        robots_in_comps = defaultdict(list)
        goal_idx = self.get_goal_idx()
        for i,comp in edges_in_comps:
            nodes = []
            for edge in comp:
                edge_nodes = self.get_nodes_along_edge(edge)


                nodes += edge_nodes
            nodes_idxs = list(map(lambda x:x[0],nodes))
            if len(nodes)<2:
                print('error')
            for k,v in pos_dic.items():
                if v in nodes_idxs:
                    robots_in_comps[k].append(i)
            if goal_idx in nodes_idxs:
                robots_in_comps['goal'].append(i)
            nodes_in_comps.append((i,nodes))
        id2comp_full = defaultdict(list)

        for i,nodes in nodes_in_comps:
            for node in nodes:
                id2comp_full[node[0]].append((i,node[1:]))
        mHG = nx.MultiDiGraph(HG)
        for k,v in id2comp_full.items():
            if len(v)==1:
                continue
            else:
                for i in v:
                    for j in v:
                        if i[0]==j[0] or (i[0],j[0]) in HG.edges:
                            continue
                        else:
                            if mHG.get_edge_data(i[0],j[0]) and mHG.get_edge_data(i[0],j[0])[0]['node_id']==k:
                                continue
                            mHG.add_edge(i[0],j[0],possible=True,node_id=k,neighbours=i[1])



        for col, ix in pos_dic.items():
            self.robots[col] = ix
        return edges_in_comps,nodes_in_comps,mHG,HG,robots_in_comps
    def get_nodes_along_edge(self,edge):
        if abs(edge[0]-edge[1]) >= 16:
            center = range(min(edge),max(edge)+1,16)
            above = range(min(edge)-16,max(edge)-15,16)
            bellow = range(min(edge)+16,max(edge)+17,16)
            zipped = [list(t) for t in zip(center,above,bellow)]
            zipped[0][1],zipped[0][2],zipped[-1][1],zipped[-1][2] = -1,-1,-1,-1
            return zipped

        else:
            left = range(min(edge)-1,max(edge))
            center = range(min(edge),max(edge)+1)
            right = range(min(edge) + 1, max(edge)+2)
            zipped = [list(t) for t in zip(center,left,right)]
            zipped[0][1],zipped[0][2],zipped[-1][1],zipped[-1][2] = -1,-1,-1,-1
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

    def save_txt(self,filenum=0):
        grid = []
        token = None
        robots = [(color,self.robots[color]) for color in COLORS]
        for index, cell in enumerate(self.grid):
            codes = to_cell_code(cell)
            if self.token in cell:
                token = index
            for code in codes:
                if code in ['N','W','E','S']:
                    x,y = xy(index)
                    x,y = x+1,y+1
                    grid.append((x,y,translate_dic[code]))
        target_color = self.token[0]
        textdata = '16\n'
        for color, ix in robots:
            x,y = xy(ix)
            x,y = x+1,y+1
            textdata += color + " " + str(y) + " " + str(x) + "\n"

        x,y = xy(token)
        x,y = x+1,y+1
        textdata += target_color + " " + str(y) + " " + str(x) + "\n" + str(len(grid)) + "\n"

        for x,y,c in grid:
            textdata += str(y) + " " + str(x) + " " + c + "\n"
        with open(f'export/{str(filenum)}.rr','w') as f:
            f.write(textdata)

        print(textdata)
