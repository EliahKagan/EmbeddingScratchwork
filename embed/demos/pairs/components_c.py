"""Connected components in an undirected graph - Way "C"."""

import collections

import attrs


@attrs.mutable
class _Node:
    """Special-purpose node."""

    _next_node = attrs.field()
    _stability = attrs.field()

    def __init__(self):
        self._next_node = self
        self._stability = 0

    def follow(self):
        if self._next_node is not self:
            self._next_node = self._next_node.follow()
        return self._next_node

    def bridge(self, other):
        self.follow()._bridge_loops(other.follow())

    def _bridge_loops(self, other):
        if self is other:
            return

        if self._stability < other._stability:
            self._next_node = other
        else:
            if self._stability == other._stability:
                self._stability += 1
            other._next_node = self


def get_components(vertices, edges):
    """Compute the components of an undirected graph with the given edges."""
    nodes = {vertex: _Node() for vertex in vertices}

    for vertex1, vertex2 in edges:
        nodes[vertex1].bridge(nodes[vertex2])

    groups = collections.defaultdict(list)  # far node -> vertex

    for vertex, node in nodes.items():
        groups[node.follow()].append(vertex)

    return frozenset(frozenset(component) for component in groups.values())
