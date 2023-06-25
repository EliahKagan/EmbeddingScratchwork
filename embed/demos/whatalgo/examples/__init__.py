"""
Examples for the ``whatalgo`` algorithm identification demo.

Each submodule is one example. The examples do not have explanatory docstrings
or comments, because the point is to investigate if embeddings can be used to
infer what algorithm is implemented. The "answer key" is as follows:

- ``components_a``: Depth-first search (DFS).
- ``components_b``: Breadth-first search (BFS).
- ``components_c``: Union-find (disjoint sets).
"""

__all__ = ['components_a', 'components_b', 'components_c']

from . import components_a, components_b, components_c
