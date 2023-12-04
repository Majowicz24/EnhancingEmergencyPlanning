import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import joblib
from sklearn.metrics import confusion_matrix
from sklearn_extra.cluster import KMedoids
import networkx as nx

# Load the SVC model and data scaler
svc_model = joblib.load('C:\\Users\\andre\\PYTHON\\svc_model.pkl')
scaler = joblib.load('C:\\Users\\andre\\PYTHON\\data_scalar.pkl')

# Load X_test and y_test data
X_test = np.load('C:\\Users\\andre\\PYTHON\\X_test.npy')
y_test = np.load('C:\\Users\\andre\\PYTHON\\Y_test.npy')

# Predict using the SVC model
predictions = svc_model.predict(X_test)

print(predictions.shape)
print(y_test.shape)

cm = confusion_matrix(y_test, predictions)
print(cm)

# Constants for time calculations
SPEED = 300  # meters per minute
STOP_DURATION = 10  # minutes

# Fetch the road network of Hoboken, NJ
G = ox.graph_from_place('Buffalo, New York, USA', network_type='drive')
#places = ['Hoboken, NJ, USA', 'Union City, NJ, USA', 'Jersey City, NJ']
#G = ox.graph_from_place(places, network_type='drive')
np.random.seed(42)
random_nodes = np.random.choice(G.nodes, 101, replace=False)

start_node = random_nodes[0]
homes_nodes = random_nodes[1:]

TP_indices = np.where((y_test == 1) & (predictions == 1))[0]
TN_indices = np.where((y_test == 0) & (predictions == 0))[0]
FP_indices = np.where((y_test == 0) & (predictions == 1))[0]
FN_indices = np.where((y_test == 1) & (predictions == 0))[0]

selected_TP = np.random.choice(TP_indices, 47, replace=False)
selected_TN = np.random.choice(TN_indices, 50, replace=False)
selected_FP = np.random.choice(FP_indices, 1, replace=False)
selected_FN = np.random.choice(FN_indices, 3, replace=False)

selected_indices = np.concatenate([selected_TP, selected_TN, selected_FP, selected_FN])

# Assume homes_nodes is a list of nodes in your graph that represent homes
# Mapping selected indices (which are TP and FP predictions) to the homes_nodes

def compute_shortest_path(start, end):
    return ox.shortest_path(G, start, end, weight='length')

def compute_path_length(path):
    return sum(G.get_edge_data(path[i], path[i + 1])[0]['length'] for i in range(len(path) - 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
n_homes = 100

def calculate_distance_matrix(nodes):
    length = len(nodes)
    distance_matrix = np.zeros((length, length))
    for i, node_origin in enumerate(nodes):
        for j, node_destination in enumerate(nodes):
            if i != j:
                # Use NetworkX function for shortest path length
                path_length = nx.shortest_path_length(G, node_origin, node_destination, weight='length')
                distance_matrix[i][j] = path_length
    return distance_matrix

homes_distance_matrix = calculate_distance_matrix(homes_nodes)

# Use KMedoids instead of KMeans
kmedoids = KMedoids(n_clusters=4, random_state=0, metric="precomputed").fit(homes_distance_matrix)


toolbox.register("indices", np.arange, n_homes)
toolbox.register("assignments", np.vectorize(lambda i: kmedoids.labels_[i]), np.arange(n_homes))
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.indices, toolbox.assignments), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

all_home_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in homes_nodes]
# Mapping selected indices (which are TP and FP predictions) to the homes_nodes
selected_nodes = np.random.choice(homes_nodes, size=min(len(selected_indices), len(homes_nodes)), replace=False)
mapping = dict(zip(selected_nodes, selected_indices))

# Determine which homes_nodes correspond to TP and FP using the mapping
homes_to_visit = [node for node, index in mapping.items() if index in np.concatenate([selected_TP, selected_FP])]



def mtsp_eval(individual):
    individual_distances = [] 
    order, assignments = individual[0], individual[1]
    for traveler_id in range(4):  
        sub_indices = np.where(assignments == traveler_id)[0]
        ordered_sub_indices = order[np.isin(order, sub_indices)]
        # Calculate path distance for the current traveler
        traveler_distance = 0
        start = start_node 
        for home_index in ordered_sub_indices:
            end = homes_nodes[home_index] 
            path = compute_shortest_path(start, end)  # Function to compute shortest path between two nodes
            traveler_distance += compute_path_length(path)  # Function to calculate the length of a path
            start = end  # Set the current endpoint to be the next start point
        path = compute_shortest_path(start, start_node)
        traveler_distance += compute_path_length(path)
        individual_distances.append(traveler_distance)  # Append distance for the traveler
    # Calculate the standard deviation of the distances traveled
    distance_std = np.std(individual_distances)  
    return (distance_std,) 
def mutate(ind):
    if np.random.rand() < 0.2:
        i = np.random.randint(len(ind[1]))
        ind[1][i] = np.random.randint(4)
    tools.mutShuffleIndexes(ind[0], indpb=0.05)
    return ind,

toolbox.register("evaluate", mtsp_eval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=4)

def get_classification_label(node, mapping, selected_TP, selected_TN, selected_FP, selected_FN):
    index = mapping.get(node, -1)  # Using -1 or some invalid default for nodes not found in the mapping
    if index in selected_TP:
        return 'TP'
    elif index in selected_TN:
        return 'TN'
    elif index in selected_FP:
        return 'FP'
    elif index in selected_FN:
        return 'FN'
    else:
        return 'Unclassified'


if __name__ == "__main__":
    population = toolbox.population(n=60)
    NGEN = 160
    CXPB = 0.8
    MUTPB = 0.17

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, stats=stats)
    
    best_ind = tools.selBest(population, 1)[0]

    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=False, edge_color='lightgray', edge_alpha=0.7, bgcolor='white')
    start_node_coord = G.nodes[start_node]['x'], G.nodes[start_node]['y']
    ax.scatter(start_node_coord[0], start_node_coord[1], c='yellow', s=100)

    medoids_nodes = homes_nodes[kmedoids.medoid_indices_]
    for medoid_node in medoids_nodes:
        medoid_coord = G.nodes[medoid_node]['x'], G.nodes[medoid_node]['y']
        ax.scatter(medoid_coord[0], medoid_coord[1], c='black', s=100, marker='x')


    for node in homes_nodes:
        classification_label = get_classification_label(node, mapping, selected_TP, selected_TN, selected_FP, selected_FN)
        node_coord = G.nodes[node]['x'], G.nodes[node]['y']
        ax.scatter(node_coord[0], node_coord[1], c='grey', s=50)  # Use a neutral color like grey
        ax.text(node_coord[0], node_coord[1] + 0.001, classification_label, fontsize=8, ha='center')  # Adjust offset as needed

    
    traveler_colors = ['blue', 'green', 'red', 'purple']
    order, assignments = best_ind[0], best_ind[1]

    total_distance_all_travelers = 0
    individual_distances = []
    total_time_travelers = []
    priority_distances = []
    priority_times = []

for traveler_id in range(4):
    sub_indices = np.where(assignments == traveler_id)[0]
    ordered_sub_indices = order[np.isin(order, sub_indices)]
    
    # Find the position of the last priority index in the route
    last_priority_index_position = max([ordered_sub_indices.tolist().index(i) for i in ordered_sub_indices if homes_nodes[i] in homes_to_visit], default=-1)
    
    # If no priority homes in this traveler's route, set last priority index position to -1 (no priority homes)
    if last_priority_index_position == -1:
        priority_indices = []
        total_time_priority = 0
        priority_distance = 0
    else:
        # Select indices up to and including the last priority index
        priority_indices = ordered_sub_indices[:last_priority_index_position + 1]

        # Calculate distance and time for the priority part
        priority_distance = 0
        time_stops_priority = len(priority_indices) * STOP_DURATION

        start = start_node
        for i in priority_indices:
            end = homes_nodes[i]
            path = compute_shortest_path(start, end)
            distance = compute_path_length(path)
            priority_distance += distance
            start = end

        time_travel_priority = priority_distance / SPEED
        total_time_priority = time_travel_priority + time_stops_priority

    priority_distances.append(priority_distance)
    priority_times.append(total_time_priority)

    # Continue with full route calculation for each traveler
    traveler_distance = 0
    time_stops = (len(ordered_sub_indices) - 1) * STOP_DURATION
    
    start = start_node
    for idx, i in enumerate(ordered_sub_indices):
        end = homes_nodes[i]
        path = compute_shortest_path(start, end)
        
        distance = compute_path_length(path)
        traveler_distance += distance

        ax.text(G.nodes[end]['x'], G.nodes[end]['y'], str(idx + 1), color='white', ha='center', va='center', weight='bold')
        ox.plot.plot_graph_route(G, path, route_color=traveler_colors[traveler_id], route_linewidth=2, orig_dest_size=100, ax=ax, show=False, close=False)
        
        start = end

    path = compute_shortest_path(start, start_node)
    traveler_distance += compute_path_length(path)
    
    time_travel = traveler_distance / SPEED
    total_time = time_travel + time_stops
    
    total_time_travelers.append(total_time)
    individual_distances.append(traveler_distance)
    total_distance_all_travelers += traveler_distance
    
    ox.plot.plot_graph_route(G, path, route_color=traveler_colors[traveler_id], route_linewidth=2, orig_dest_size=100, ax=ax, show=False, close=False)

# Print the total optimal distance and the details for each traveler
print(f"Total optimal distance covered by all travelers: {total_distance_all_travelers:.2f} meters\n")
for color, distance, time in zip(traveler_colors, individual_distances, total_time_travelers):
    print(f"{color.capitalize()} traveler:")
    print(f"Distance covered: {distance:.2f} meters")
    print(f"Total time taken: {time:.2f} minutes\n")

# Print the priority distances and times for each traveler
print("Priority Distances and Times for Each Traveler:")
for traveler_id, (distance, time) in enumerate(zip(priority_distances, priority_times)):
    print(f"Traveler {traveler_id + 1}:")
    print(f"Priority Distance: {distance:.2f} meters")
    print(f"Priority Time: {time:.2f} minutes")
    print()

plt.show()