import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt

st.title("Smart Route Finder (A* vs Greedy + Graph Visualization)")

# ---------------------------
# Graph (like your image)
# ---------------------------
graph = {
    'A': {'B': 7, 'D': 3},
    'B': {'F': 6},
    'D': {'E': 4},
    'E': {'F': 4},
    'F': {}
}

# Heuristic values (to goal F)
heuristic = {
    'A': 10,
    'B': 6,
    'D': 7,
    'E': 4,
    'F': 0
}

cities = list(graph.keys())

# ---------------------------
# A* Algorithm
# ---------------------------
def a_star(start, goal):
    open_list = [(0, start)]
    came_from = {}
    g_cost = {node: float('inf') for node in graph}
    g_cost[start] = 0
    nodes_explored = 0

    while open_list:
        _, current = heapq.heappop(open_list)
        nodes_explored += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_cost[goal], nodes_explored

        for neighbor, cost in graph[current].items():
            new_cost = g_cost[current] + cost

            if new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current

    return None, float('inf'), nodes_explored

# ---------------------------
# Greedy Search
# ---------------------------
def greedy(start, goal):
    open_list = [(heuristic[start], start)]
    came_from = {}
    visited = set()
    nodes_explored = 0

    while open_list:
        _, current = heapq.heappop(open_list)
        nodes_explored += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)

            # Calculate cost
            total_cost = 0
            path_reversed = path[::-1]
            for i in range(len(path_reversed) - 1):
                total_cost += graph[path_reversed[i]][path_reversed[i+1]]

            return path_reversed, total_cost, nodes_explored

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                heapq.heappush(open_list, (heuristic[neighbor], neighbor))
                came_from[neighbor] = current

    return None, float('inf'), nodes_explored

# ---------------------------
# Graph Visualization
# ---------------------------
def draw_graph():
    G = nx.DiGraph()

    for node in graph:
        for neighbor, cost in graph[node].items():
            G.add_edge(node, neighbor, weight=cost)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(5,4))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    st.pyplot(plt)

# ---------------------------
# UI
# ---------------------------
st.subheader("Graph Visualization")
draw_graph()

start = st.selectbox("Start City", cities)
goal = st.selectbox("Goal City", cities)

if st.button("Find Route"):

    path_a, cost_a, explored_a = a_star(start, goal)
    path_g, cost_g, explored_g = greedy(start, goal)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### A*")
        st.write("Path:", " → ".join(path_a))
        st.write("Cost:", cost_a)
        st.write("Nodes Explored:", explored_a)

    with col2:
        st.markdown("### Greedy")
        st.write("Path:", " → ".join(path_g))
        st.write("Cost:", cost_g)
        st.write("Nodes Explored:", explored_g)

    st.markdown("### Conclusion")
    st.write("""
    A* gives optimal path using g(n)+h(n).
    Greedy is faster but may not give shortest path.
    """)