<a href="https://colab.research.google.com/github/googlecolab/colabtools/Anna-Peng/SOM-exercise/blob/master/SOM.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# -*- coding: utf-8 -*-

import numpy as np # This package deals with matrix and array
from matplotlib import pyplot as plt

data_dim = 12 # the number of dimensions
data_no = 300 # Number of data sample (divisable by 3 groups)
n_iteration=[5000, 8000, 12000] # the script will loop through different no. of training iterations
lrate = 0.07 # learning rate, the value will decrease with iteration
group_diff = [[0, 1], [3], [4, 5]] # these specific dimensions that make a unique group
training_no = data_no


'''
Generate random data sample
'''
raw_data = [np.random.random(data_dim) for i in range(data_no)] # Generate data samples
n_group = int(data_no/len(group_diff)) # number of sample in each group
group_index = [range(0, n_group-1), range(n_group, 2*n_group-1), range(2*n_group, 3*n_group-1)]

for i, ii, iii in zip(group_index[0], group_index[1], group_index[2]): # Coerce group indices
        raw_data[i][group_diff[0]]   = 0.01 #[1 if a < 0.5 else a for a in raw_data[i][group_diff[0]] ]
        raw_data[ii][group_diff[1]]  = 0.01 #[1 if a < 0.5 else a for a in raw_data[ii][group_diff[1]] ]
        raw_data[iii][group_diff[2]] = 0.01 #[1 if a < 0.5 else a for a in raw_data[iii][group_diff[2]] ]   

'''
Construct Network
'''    
n_node=17 # network size (square) rule of thumb M= 5*sqrt(dim)
net = np.random.random((n_node, n_node, data_dim)) # Random weight matrix
init_rad= n_node/2  # initiate neighborhood size to 1/2 of grid space, this will gradually decrease, can be of different values 

'''
Define functions
'''

def radius_update(init_rad, timer, norm_it): # this function reduces the current neighborhood radius
    return init_rad * np.exp( - timer / norm_it)

def eta_update(lrate, timer, iteration): # This function reduces learning rate
    return lrate * np.exp( - timer / iteration)
    
def calculate_influence(distance, curr_rad): # This function determines a larger influence to closer node to bmu
    return np.exp(-distance / (2* (curr_rad**2))) # i.e. smaller influence to distant node

def find_bmu(v, net): # Find best matching unit, v is one training data sample    
    bmu_id = np.array([0, 0]) # initiate empty bmu_id
    # initiate a large number for the greatest possible manhatten distance (distance--diff bt data sample and weight vector), the value is a computer-defined number. 
    # This computer-based number can be used for non-normalised data where the distance can be large  
    min_dist = np.iinfo(np.int).max 
    for i in range(n_node):
        for j in range(n_node):
            W = net[i, j, :] # node weights to all dimensional nodes 
            # Calculate manhatten Distance bt weight vector and input vector
            # Manhattan distance. Definition: The distance between two points measured along axes at right angles. 
            # it is |x1 - x2| + |y1 - y2|
            curr_dist = np.sum(abs(W - v)) 
            if curr_dist < min_dist:
                    min_dist = curr_dist
                    bmu_id = np.array([i, j]) # record which node is the best-matching unit
    bmu_w = net[bmu_id[0], bmu_id[1], :] # Get the weight vector for the bmu
    return (bmu_w, bmu_id) # bmu_w is the weight vector to the bmu unit
    
def move_neighbour(v, bmu_id, net, curr_rad, curr_lrate): # This function finds the neighbors and update the weights
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            W = net[x, y, :] # input-node weight vector
            w_dis = np.sum(abs(np.array([x, y]) - bmu_id)) # w_dist for all neighboring nodes
            if w_dis <= curr_rad:
                influence = calculate_influence(w_dis, curr_rad)
                new_w = W + (curr_lrate * influence * (v - W)) # if unit in v is smaller than unit in W, weight is reduced
                net[x, y, :] = new_w
    return net


curr_rad = init_rad
curr_lrate = lrate

'''
Running the net with for loop in n_iteration
'''
for iteration in n_iteration:
    norm_it = iteration / np.log(init_rad) # this allows neighborhood to decay to 1 with iteration time
    for timer in range(iteration):
            train_v=raw_data[np.random.randint(1,training_no)]
            bmu_w, bmu_id = find_bmu(train_v, net)
            trained_net = move_neighbour(train_v, bmu_id, net, curr_rad, curr_lrate)
            curr_rad = radius_update(init_rad, timer, norm_it)
            curr_lrate = eta_update(lrate, timer, iteration)
            

    
    '''
    Plotting
    '''
    
    def euc_dist(v1, v2):
        return np.linalg.norm(v1 - v2) 
    
    Rows=n_node
    Cols=n_node
    
    print("Constructing U-Matrix from SOM, Data: {3}, Net: {0}, Dim: {1}, It: {2}".format(n_node, data_dim, iteration, data_no))
    u_matrix = np.zeros(shape=(Rows, Cols), dtype=np.float64)
    
    for i in range(Rows):
        for j in range(Cols):
            v = trained_net[i][j]  # matrix to input unit vector
            sum_dists = 0.0; ct = 0 # ct counter
            if i-1 >= 0:    # above
                sum_dists += euc_dist(v, trained_net[i-1][j]); ct += 1
            if i+1 <= Rows-1:   # below
                sum_dists += euc_dist(v, trained_net[i+1][j]); ct += 1
            if j-1 >= 0:   # left
                sum_dists += euc_dist(v, trained_net[i][j-1]); ct += 1
            if j+1 <= Cols-1:   # right
                sum_dists += euc_dist(v, trained_net[i][j+1]); ct += 1
            u_matrix[i][j] = sum_dists / ct
            
    plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
    plt.show()


for i in range(0,data_dim):
    print("Dimension: {}".format(i))
    plt.imshow(trained_net[:,:,i], cmap='gray')
    plt.show()
