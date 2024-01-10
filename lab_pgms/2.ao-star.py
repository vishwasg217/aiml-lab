class Graph():
    def __init__(self,heuristic, graph, start_node):
        self.h = heuristic
        self.g = graph
        self.start_node = start_node
        self.best_path = {}

    def AOstar(self, node):
        print("HEURISTIC VALUES  :", self.h)
        print("SOLUTION GRAPH    :", self.best_path)
        print("PROCESSING NODE   :", node)
        print("-----------------------------------------------------------------------------------------")
        
        if node not in self.g:
            return self.h[node] , node
        
        min_cost = float('inf')
        best_subpath = None

        for children in self.g[node]:
            total_cost = 0
            subpath = [node]

            for child , weight in children:
                child_cost , child_path = self.AOstar(child)
                total_cost += child_cost + weight
                subpath.extend(child_path)

            if total_cost < min_cost:
                min_cost = total_cost
                best_subpath = subpath

        self.h[node] = min_cost
        self.best_path[node] = best_subpath
        return min_cost, best_subpath
    
    def find_solution(self):
        cost, path = self.AOstar(self.start_node)
        return cost , path

h = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 0, 'T': 3}
g = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]   
}
graph = Graph(h, g, 'A')
solution_path, solution_graph = graph.find_solution()
print(solution_graph)