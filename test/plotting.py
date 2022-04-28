import numpy
from matplotlib import pyplot
from dataclasses import dataclass
from typing import Dict
import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')
from data_classes import Level

@dataclass
class PlottingNode():
    # Horizontal line in Dendogram
    y: float # Horizontal distance
    left: Dict[int, dict] # Left cluster branch
    right: Dict[int, dict] # Right cluster branch

# TODO: remove?
class PlottingTree():
    tree: Dict[PlottingNode, dict]

    def __init__(self):
        return

def first_pass(levels: Dict[int, Level]):
    level_ids = list(levels.keys())
    level_ids.sort(reverse=True)

    # First split
    highest_level = level_ids[0]
    left_cluster = min(levels[highest_level].merged_clusters)
    right_cluster = max(levels[highest_level].merged_clusters)
    node = PlottingNode(y=highest_level, left={left_cluster: ''}, right={right_cluster: ''})

    tree = {'root':node}
    # track path to top of tree

    for level_id in level_ids[1:]:
        previous = levels[level_id].new_cluster
        left_cluster = min(levels[level_id].merged_clusters)
        right_cluster = max(levels[level_id].merged_clusters)

        node = PlottingNode(y=level_id, left={left_cluster: ''}, right={right_cluster: ''})

        # place node in tree
        

    return tree

def main():
    # clusters = [0,1,2]
    level1 = Level(
        level_id = 1,
        clusters = [],
        merged_clusters = [1,2],
        new_cluster = 3
    )
    # clusters = [0,3]
    level2 = Level(
        level_id = 2,
        clusters = [],
        merged_clusters = [0,3],
        new_cluster = 4
    )
    # clusters = [4]

    levels = [level1, level2]
    levels = dict(zip([1,2], levels))
    # Number of initial Clusters = number of Levels + 1

    first_pass(levels=levels)
    return

if __name__ == '__main__':
    main()