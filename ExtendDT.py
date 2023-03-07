# This is a class for extending the decision tree of the sklearn

# %% Import required packages
import numpy as np
import pandas as pd
from sklearn.tree._tree import TREE_LEAF
import copy

# %% Class definition
class ext_dt:
    # ========== Initialize the object
    def __init__(self, base_dt):
        self.base_dt = base_dt # Give the base DT
        self.n_all_nodes = self.base_dt.tree_.node_count # Number of nodes
        
    # ========== Find leafs of the DT
    def findLeafs(self):
        # ----- get children information
        children_left = self.base_dt.tree_.children_left
        children_right = self.base_dt.tree_.children_right
        # ----- get and store all nodes count
        n_nodes = self.n_all_nodes
        is_leaf = np.zeros(shape = n_nodes, dtype = bool)
        # ----- Find the leaf split status (even hidden nodes)
        for idx in range(n_nodes):
            if children_left[idx] != children_right[idx]: # it is split
                is_leaf[idx] = False
            else: # it is leaf 
                is_leaf[idx] = True
        self.is_leaf = is_leaf
        
    # ========== Find each node's parent
    def findParents(self):
        children_left = self.base_dt.tree_.children_left
        children_right = self.base_dt.tree_.children_right
        n_nodes = self.n_all_nodes
        node_parent = np.zeros(shape = n_nodes, dtype = np.int64) # Parent node ID
        node_parent += -1 # this is to differentiate parents for the hidden (removed nodes)
        # ----- Initialize the stack
        stack = [(0, -1)]  # [node id, parent id] root node has no parent => -1
        # ----- Go downward to identify child/parent and leaf status
        while len(stack) > 0:    
            # ----- check each node only once, using pop
            node_id, parent  = stack.pop()
            node_parent[node_id] = parent
            # ----- add data of split point to the stack to go through
            if children_left[node_id] != children_right[node_id]:
                child_l_id = children_left[node_id]
                child_r_id = children_right[node_id]
                stack.append((child_l_id, node_id))
                stack.append((child_r_id, node_id))
        self.parent = node_parent
        
    # ========== Find each node's children
    def getChildren(self):
        n_nodes = self.n_all_nodes
        children_left = self.base_dt.tree_.children_left
        children_right = self.base_dt.tree_.children_right
        children = [[]] * n_nodes
        for idx in range(n_nodes):
            children[idx] = [children_left[idx], children_right[idx]]
        self.children = children
    
    # ========== Make informatoin of nodes' depth
    def getNodeDepth(self):
        n_nodes = self.n_all_nodes
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        #* Simplify access to children
        children_left = self.base_dt.tree_.children_left
        children_right = self.base_dt.tree_.children_right
        #* What is the maximum depth of tree
        mx_depth = self.base_dt.tree_.max_depth
        #* Empty list to store depth of each node
        node_depth = np.zeros(shape = n_nodes, dtype = np.int64)
        #* Build an empty list (depth is +1 due to the way sklearn measures it)
        list_nodes_in_depth = [[] for _ in range(mx_depth + 1)]
        #* Initialize the stack
        stack = [(0, 0)] #* (node id , Depth)
        #* Go through the stack
        while len(stack) > 0:
            #* Get a single node
            node_id, depth  = stack.pop()
            #* Store its depth
            node_depth[node_id] = depth
            #* Append it to the related list
            list_nodes_in_depth[depth].append(node_id)
            #* If this node is not a leaf
            if not is_leaf[node_id]:
                #* Add its children to the stack
                child_l_id = children_left[node_id]
                child_r_id = children_right[node_id]
                stack.append((child_l_id, depth + 1))
                stack.append((child_r_id, depth + 1))
        #* Store the data in the class
        self.node_depth = node_depth
        self.nodes_in_depth = list_nodes_in_depth
        
    # ========== Find depth of each leaf node
    def getLeafDepth(self):
        n_nodes = self.n_all_nodes
        leaf_depth = np.zeros(shape = n_nodes, dtype = np.int64)
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        # ----- Are node depth already known?
        if not hasattr(self, "node_depth"):
            self.getNodeDepth()
        node_depth = self.node_depth
        for node_ind in range (n_nodes):         
            if is_leaf[node_ind] == True:
                leaf_depth[node_ind] = node_depth[node_ind]
        self.leaf_depth = leaf_depth
    
    # ========== Find lists of index for all leafs and nodes
    def getIndexLists(self):
        n_nodes = self.n_all_nodes
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        # @ a list for normal nodes
        list_nodes = []
        # @ a list for leafs
        list_leafs = []
        for node_ind in range (n_nodes):         
            if is_leaf[node_ind] == True:
                list_leafs.append(node_ind)
            else:
                list_nodes.append(node_ind)
        self.list_leafs = list_leafs
        self.list_nodes = list_nodes

    # ========== Calculate statistical information of the DT's depth
    def getDepthStat(self):
        if not hasattr(self, "leaf_depth"):
            self.getLeafDepth()
        leaf_depth_raw = self.leaf_depth
        # ----- remove data about the non leaf nodes
        clean_leaf_depth = leaf_depth_raw[leaf_depth_raw != 0]
        # ----- find values
        if len(clean_leaf_depth) != 0: 
            min_depth = np.min(clean_leaf_depth)
            mean_depth = np.mean(clean_leaf_depth)
            median_depth = np.median(clean_leaf_depth)
            max_depth = np.max(clean_leaf_depth)
        else: # only root is remained
            min_depth = 0
            mean_depth = 0
            median_depth = 0
            max_depth = 0
        return min_depth, mean_depth, median_depth, max_depth
    
    # ========== Calculate Resource Cost (RC) factors
    def getRCFactor(self):
        n_nodes = self.n_all_nodes
        samples = self.base_dt.tree_.n_node_samples
        # RC_node = np.zeros(shape = n_nodes, dtype = np.int64)
        RC_node = copy.copy(samples)
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        # ----- Are parents already known?
        if not hasattr(self, "parent"):
            self.findParents()
        parent = self.parent
        # ----- From leaves go upward and calculate the PC
        for node_ind in range (n_nodes):         
            if not is_leaf[node_ind]:
                current_parent = parent[node_ind]
                while current_parent != -1:
                    RC_node[current_parent] += samples[node_ind]
                    current_parent = parent[current_parent] # get the next parent node id
        RC_branch = RC_node / samples
        self.RC_node = RC_node
        self.RC_branch = RC_branch
        self.DT_APD = RC_branch[0]
        
    def get_APD(self):
        if not hasattr(self, "DT_APD"):
            self.getRCFactor()
        return self.DT_APD

    def get_branchSamples(self, node_id):
        if not hasattr(self, "RC_node"):
            self.getRCFactor()
        return self.RC_node[node_id]

    def get_branchAPD(self, node_id):
        if not hasattr(self, "RC_branch"):
            self.getRCFactor()
        return self.RC_branch[node_id]
    # # ========== Calculate Resource Cost (RC) factors
    # def getRCFactor(self):
    #     n_nodes = self.n_all_nodes
    #     samples = self.base_dt.tree_.n_node_samples
    #     RC_node = np.zeros(shape = n_nodes, dtype = np.float64)
    #     # ----- Are leafs already detected?
    #     if not hasattr(self, "is_leaf"):
    #         self.findLeafs()
    #     is_leaf = self.is_leaf
    #     # ----- Are parents already known?
    #     if not hasattr(self, "parent"):
    #         self.findParents()
    #     parent = self.parent
    #     # ----- From leaves go upward and calculate the PC
    #     for node_ind in range (n_nodes):         
    #         if is_leaf[node_ind]:
    #             current_parent = parent[node_ind]
    #             upward_length = 1
    #             while current_parent != -1:
    #                 RC_node[current_parent] = RC_node[current_parent] + upward_length * samples[node_ind]/samples[current_parent]
    #                 current_parent = parent[current_parent] # get the next parent node id
    #                 upward_length = upward_length + 1
    #         RC_branch = RC_node * samples / samples[0]
    #     self.RC_node = RC_node
    #     self.RC_branch = RC_branch
    #     self.DT_APD = RC_branch[0]
    
    # ========== Calculate Absolute Cost factors
    def getAbsCostFactor(self):
        n_nodes = self.n_all_nodes
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        # ----- Are parents already known?
        if not hasattr(self, "parent"):
            self.findParents()
        parent = self.parent
        # * all class' samples are integer values
        tr_values = self.base_dt.tree_.value.astype(int)
        # * get the sum of samples in each node
        samples = np.sum(tr_values, 2)
        # * In case of pruning all samples from non-max class are missclassified
        miss_class = samples - np.amax(tr_values, 2)
        # * Flatten out the arrays
        samples = [item for sublist in samples for item in sublist]
        miss_class = [item for sublist in miss_class for item in sublist]
        # * A FIFO list is needed -> use queue
        from queue import Queue
        queue = Queue()
        # * Initialization of parameters
        # @ a node cost is summed when all children are checked
        child_checked = np.zeros(shape = n_nodes, dtype = np.int8)
        abs_cost = copy.copy(samples)
        # ** Handling the leafs
        for node in range(n_nodes):
            if is_leaf[node]:
                # * Leaves have no child -> add to queue and already checked
                queue.put(node)
                child_checked[node] = 2
                # * leafs have no cost
                abs_cost[node] = 0
        # ** Go through all nodes in the queue
        while not queue.empty():
            current_node = queue.get()
            current_samples = abs_cost[current_node]
            current_parent = parent[current_node]
            child_checked[current_parent] += 1
            # * add to the queue if both children checked and not the root
            if child_checked[current_parent] == 2 and current_parent != 0:
                queue.put(current_parent)
            # * Update the cost of the node
            abs_cost[current_parent] = abs_cost[current_parent] + current_samples
        self.samples = samples
        self.miss_class = miss_class
        self.abs_cost = abs_cost
    # ========== Calculate impurity factors
    def getImpFactor(self):
        n_nodes = self.n_all_nodes
        impurity = self.base_dt.tree_.impurity   
        weighted_n_node_samples = self.base_dt.tree_.weighted_n_node_samples
        total_sum_weights = self.base_dt.tree_.weighted_n_node_samples[0]
        # ----- Initialise empty variables
        r_node = np.zeros(shape = n_nodes, dtype = np.float64)
        r_branch = np.zeros(shape = n_nodes, dtype = np.float64)
        # ----- Calculate the relative r_node
        for node_ind in range (n_nodes): # for all nodes
            r_node[node_ind] = (weighted_n_node_samples[node_ind] * impurity[node_ind] / total_sum_weights)
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        # ----- Are parents already known?
        if not hasattr(self, "parent"):
            self.findParents()
        parent = self.parent
        # ----- Calculate the r_branch
        for ind in range (n_nodes):
            if is_leaf[ind]:
                current_r = r_node[ind]
                parent_ind = parent[ind]
                while parent_ind != -1:
                    r_branch[parent_ind] += current_r
                    # leaves_branch[parent_ind] += 1 # No. of leaves in the branch, if needed (for CCP)
                    parent_ind = parent[parent_ind]
                if ind == 0: # only root node is remained in the DT (root is a leaf)
                    r_branch[0] += current_r
        self.Imp_node = r_node
        self.Imp_branch = r_branch
        self.DT_Imp = r_branch[0]

    # ========== Remove a single branch from the DT
    def removeBranch(self, PruneBranchID):
        # ----- Make a copy using deepcopy (not related versions)
        dt_pruned = copy.deepcopy(self.base_dt)
        # ----- read data from the tree
        children_left = dt_pruned.tree_.children_left
        children_right = dt_pruned.tree_.children_right
        n_nodes = dt_pruned.tree_.node_count   
        # ----- Initialise empty variables
        nodes_to_remove = np.zeros(shape = n_nodes, dtype = np.int64)
        # ----- Are leafs already detected?
        if not hasattr(self, "is_leaf"):
            self.findLeafs()
        is_leaf = self.is_leaf
        # ----- Check that the node is split and not a leaf
        if is_leaf[PruneBranchID]:
            raise ValueError("Prune node cannot be a leaf! \n Solution: Select index of a split node.")
        # ----- Use a stack to list the id of all nodes which have to be removed
        stack = [children_left[PruneBranchID]]
        stack.append(children_right[PruneBranchID])
        while len(stack) > 0: 
            stack_ind  = stack.pop()
            nodes_to_remove[stack_ind] = True # This node is removed
            if not is_leaf[stack_ind]: # it is a split node (has children)
                # add children to the stack
                stack.append((children_left[stack_ind]))
                stack.append((children_right[stack_ind]))
            children_left[stack_ind] = -1
            children_right[stack_ind] = -1
        # ----- detach children
        children_left[PruneBranchID] = TREE_LEAF
        children_right[PruneBranchID] = TREE_LEAF
        # ----- Number of removed nodes
        no_removed_nodes = np.sum(nodes_to_remove)
        return dt_pruned
    
    # ========== Analyze the performance of the DT and find its Resource Cost (RC) for test data
    def getPerformance(self, X_data, Y_Data):
        data_size = len(X_data)
        # ----- Get the score for the whole data
        test_score = self.base_dt.score(X_data, Y_Data)
        # ----- find the leaf id for each data sample
        leaf_node = self.base_dt.apply(X_data) 
        # ----- find the depth of each lead node
        if not hasattr(self, "leaf_depth"):
            self.getLeafDepth()
        leaf_depth = self.leaf_depth
        # ----- Store depth for each test data
        sample_RC = np.zeros(shape = data_size, dtype = np.int64)
        for idx in range(data_size):
            sample_RC[idx] = leaf_depth[leaf_node[idx]]
        # ----- Calculate the average (APD)
        test_APD = np.sum(sample_RC)/data_size
        # ------ Check if the prediction is correct
        test_prediction = self.base_dt.predict(X_data)
        test_correct = test_prediction == Y_Data
        # ----- make a dataframe
        perf_data = pd.DataFrame({"Cost" : sample_RC,
                                  "Correct" : test_correct})

        return test_score, test_APD, perf_data
    
    # ========== Prepare all data required for selecting the pruning node
    def getPruneData(self):
        n_nodes = self.n_all_nodes
        if not hasattr(self, "RC_node"):
            self.getRCFactor()
        if not hasattr(self, "Imp_node"):
            self.getImpFactor()
        # ----- Build the dataframe
        all_prune_data = pd.DataFrame({"Node" : np.arange(n_nodes),
                                   "is_leaf" : self.is_leaf,
                                   "Imp_node" : self.Imp_node,
                                   "Imp_branch" : self.Imp_branch,
                                   "Imp_Gini" : self.base_dt.tree_.impurity, 
                                   "RC_node" : self.RC_node,
                                   "RC_branch" : self.RC_branch
                                   })
        # ----- Remove leafs
        prune_data = all_prune_data.drop(all_prune_data[all_prune_data.is_leaf == True].index)
        # ----- Clean up the output
        # Remove the leaf data column
        prune_data.drop('is_leaf', axis=1, inplace=True)
        # Reset the index for easier access
        prune_data.reset_index(drop=True, inplace=True)
        return prune_data
    # TODO: Add a function to find number of hidden nodes