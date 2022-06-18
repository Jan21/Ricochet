import itertools
import heapq as hq
from dataclasses import dataclass, field
from typing import Any,List,Tuple

@dataclass(order=True)
class Subgoal:
    priority: int
    possible_cost: int=field(compare=False)
    cost_so_far: int=field(compare=False)
    is_subgoal_for_target : bool=field(compare=False)
    previous_subgoals: list
    info: Any=field(compare=False)


class PriorityQueue:
    def __init__(self):
        self.elements: List[Subgoal] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: Subgoal):
        hq.heappush(self.elements,item)

    def get(self) -> Subgoal:
        return hq.heappop(self.elements)

    def merge(self,items: List):
        for item in items:
            hq.heappush(self.elements,item)


