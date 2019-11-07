"""
Script created for testing and learning graph-tool library.
"""

import graph_tool.all as graph_tool

if __name__ == '__main__':
    g = graph_tool.Graph()
    v1 = g.add_vertex()
    print(v1)
    v1.lemma = "aaa"
    v2 = g.add_vertex()
    print(v2)
    g.add_edge(v1, v2)
    print(g.vertex_index(v1))
