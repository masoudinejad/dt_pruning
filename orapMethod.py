from ExtendDT import ext_dt
from Sequence import *

#> Find the optimal sequence for a single node when its children are optimized
def PruneOptimalSingle(xdt, main_seq, root_id):
    current_seq = copy.copy(main_seq[root_id])
    #* if optimized is not available, make it
    if  not current_seq.optimized:
        #* Begin of optimization procedure ==============================
        print(f"[{root_id} (d:{xdt.node_depth[root_id]})] | Opting ===>")
        # * check this is a leaf
        if xdt.is_leaf[root_id]:
            combined_seq = PruneSequence()
            combined_seq.OptFinished()
            current_seq = combined_seq
        else:
            #* Find children IDs
            left_root = xdt.children[root_id][0]
            right_root = xdt.children[root_id][1]
            #* Get optimal sequence of children
            lef_seq, node_id  = PruneOptimalSingle(xdt, main_seq, left_root)
            right_seq, node_id  = PruneOptimalSingle(xdt, main_seq, right_root)
            #* Combine children sequences with each other
            combined_seq = CombineSequence(lef_seq, right_seq, root_id)
            # * Append combined seq to the sequence of root
            current_seq.AppendSequence(combined_seq)
            #+ Tuple of changing root into a leaf
            current_tuple = [xdt.abs_cost[root_id], xdt.miss_class[root_id], [root_id]]
            #* Append this to the sequence
            current_seq.AppendTuple(current_tuple)
            #* Set register to finished
            current_seq.OptFinished()
        #* End of optimization procedure ==============================
        print(f"[{root_id}] ===> Ready |")
    return current_seq, root_id

#> Function for finding the optimal sequence of a node recursively
def PruneOptimal(xdt, goal_node):
    #* Get depth list
    xdt.getNodeDepth()
    depth_list = xdt.nodes_in_depth
    #* get the cost factors
    xdt.getAbsCostFactor()
    #* Find the children relations
    xdt.getChildren()
    #* Make the empty sequence for all
    main_seq = [PruneSequence()] * xdt.n_all_nodes
    #* Go through all depth from bottom to top
    for depth_row in range(len(depth_list)-1, -1, -1):
        row_list = depth_list[depth_row]
        #* Go through nodes in this depth
        for node_in_list in row_list:
            current_seq, node_id = PruneOptimalSingle(xdt, main_seq, node_in_list)
            main_seq[node_in_list] = current_seq
            if node_in_list == goal_node:
                break
        if node_in_list == goal_node:
            break
    return main_seq[goal_node]

#> Function for finding the optimal sequence of a node recursively
def PruneOptimalParallel(xdt, goal_node):
    #* Load multi-processor pckg
    import multiprocessing as mp
    #* Get depth list
    xdt.getNodeDepth()
    depth_list = xdt.nodes_in_depth
    #* get the cost factors
    xdt.getAbsCostFactor()
    #* Find the children relations
    xdt.getChildren()
    #* Make the empty sequence for all
    main_seq = [PruneSequence()] * xdt.n_all_nodes
    #+ Is goal reached
    goal_reached = False
    #* Start the pool
    num_processor = mp.cpu_count()
    pool = mp.Pool(processes = int(num_processor/2))
    #* Go through all depth from bottom to top
    for depth_row in range(len(depth_list)-1, -1, -1):
        #* Go through nodes in this depth
        row_list = depth_list[depth_row]   
        #* Run the pool in this depth
        result_async = [pool.apply_async(PruneOptimalSingle, args = (xdt, main_seq, node_in_list)) for node_in_list in row_list]
        #* Get the pool results
        for worker_idx in result_async:
            current_seq = worker_idx.get()[0]
            current_node = worker_idx.get()[1]
            main_seq[current_node] = current_seq
            if current_node == goal_node:
                goal_reached = True
        if goal_reached:
            break
    pool.close()
    # pool.join()
    return main_seq[goal_node]

#> A function to get the 
def get_ORAP_sequence(dt_base, **kwargs):
    # parse optional input
    use_parallel = False # by default use a single core
    for key, value in kwargs.items():
        # if use_parallel passed in set its value
        if key == "use_parallel":
            use_parallel = value
    #* Build the extended tree object
    xdt = ext_dt(dt_base)
    #* Calculate the optimal sequence for the root of the whole DT
    goal_node = 0
    #* Decide about number of processors
    if use_parallel:
        goal_seq = PruneOptimalParallel(xdt, goal_node)
    else:
        goal_seq = PruneOptimal(xdt, goal_node)
    #* Get the sequence for root (complete sequence)
    ORAP_base_sequence = copy.copy(goal_seq)
    #* Sort sequence according to the cost
    sort_factor = "cost"
    ORAP_base_sequence.SortSequence(sort_factor)
    #* Add the original tree to the sequence
    ORAP_base_sequence.cost.insert(0, 0)
    ORAP_base_sequence.miss.insert(0, 0)
    ORAP_base_sequence.removed.insert(0, [])

#>A function to remove a series of nodes from a base DT
def series_prune(xdt, node_series):
    pruned_xdt = copy.deepcopy(xdt)
    #* Step by step remove branches
    for prune_node in node_series:
        current_dt = pruned_xdt.removeBranch(prune_node)
        pruned_xdt = ext_dt(current_dt)
    return pruned_xdt