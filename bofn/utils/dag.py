from typing import List, Optional


class DAG(object):
    
    def __init__(self, parent_nodes:List[List[Optional[int]]]):
        self.parent_nodes = parent_nodes
        self.n_nodes = len(parent_nodes)
        self.root_nodes = []
        for k in range(self.n_nodes):
            if len(parent_nodes[k]) == 0:
                self.root_nodes.append(k)
    
    def get_n_nodes(self):
        return self.n_nodes
    
    def get_parent_nodes(self, k):
        return self.parent_nodes[k]
    
    def get_root_nodes(self):
        return self.root_nodes
    