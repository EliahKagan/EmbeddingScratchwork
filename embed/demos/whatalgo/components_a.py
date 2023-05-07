"""Connected components in an undirected graph."""


def get_components(vertices, edges):
    """Compute the components of an undirected graph with the given edges."""
    adj = {vertex: [] for vertex in vertices}

    for vertex1, vertex2 in edges:
        adj[vertex1].append(vertex2)
        adj[vertex2].append(vertex1)

    unclassified = set(adj)  # Use adj, since vertices can be an iterator.

    def discover(src, vis):
        unclassified.remove(src)
        vis.add(src)
        for dest in adj[src]:
            if dest not in vis:
                discover(dest, vis)

    components = []

    while unclassified:
        component = set()
        discover(unclassified.pop(), component)
        components.append(frozenset(component))

    return frozenset(components)
