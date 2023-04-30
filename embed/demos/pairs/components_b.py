"""Connected components in an undirected graph - Way "B"."""

import collections


def get_components(vertices, edges):
    """Compute the components of an undirected graph with the given edges."""
    adj = {vertex: [] for vertex in vertices}

    for vertex1, vertex2 in edges:
        adj[vertex1].append(vertex2)
        adj[vertex2].append(vertex1)

    unclassified = set(adj)  # Use adj, since vertices can be an iterator.

    def discover(start):
        unclassified.remove(start)
        vis = {start}
        fringe = collections.deque(vis)
        while fringe:
            src = fringe.popleft()
            for dest in adj[src]:
                if dest in vis:
                    continue
                unclassified.remove(dest)
                vis.add(dest)
                fringe.append(dest)

    components = []

    while unclassified:
        components.append(discover(unclassified.pop()))

    return frozenset(components)
