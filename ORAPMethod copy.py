# %% Load required modules
from GeneralFncs import *
from ExtendDT import ext_dt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import time
import copy
from Sequence import *
import os

# #> Find the optimal sequence for a single node when its children are optimized
# def PruneOptimalSingle(xdt, main_seq, root_id, **kwargs):
#     store_data = False
#     for key, value in kwargs.items():
#         #* Store the data into the memory
#         if key == "store_data":
#             store_data = value
#         #* Name of the dataset for proper data storage
#         if key == "data_name":
#             data_name = value
#     # if store_data:
#     #     #* folder for storing test results
#     #     store_path_test = ("./", "Sessions", data_name, "Temp") 
#     #     Result_file_path = makeAdd(store_path_test)
#     #* folder for storing test results
#     store_path_test = ("./", "Sessions", data_name, "Temp") 
#     Result_file_path = makeAdd(store_path_test)
#     desired_file = f"{Result_file_path}/Temp_{data_name}_{root_id}"
    
#     current_seq = copy.copy(main_seq[root_id])
#     #* if optimized is not available, make it
#     if  not current_seq.optimized:
#         #* Begin of optimization procedure ==============================
#         print(f"[{root_id} (d:{xdt.node_depth[root_id]})] | Opting ===>")
#         # * check this is a leaf
#         if xdt.is_leaf[root_id]:
#             combined_seq = PruneSequence()
#             combined_seq.OptFinished()
#             current_seq = combined_seq
#         else:
#             #* Find children IDs
#             left_root = xdt.children[root_id][0]
#             right_root = xdt.children[root_id][1]
#             #* Get optimal sequence of children
#             lef_seq, node_id  = PruneOptimalSingle(xdt, main_seq, left_root)
#             right_seq, node_id  = PruneOptimalSingle(xdt, main_seq, right_root)
#             #* Combine children sequences with each other
#             combined_seq = CombineSequence(lef_seq, right_seq, root_id)
#             # * Append combined seq to the sequence of root
#             current_seq.AppendSequence(combined_seq)
#             #+ Tuple of changing root into a leaf
#             current_tuple = [xdt.abs_cost[root_id], xdt.miss_class[root_id], [root_id]]
#             #* Append this to the sequence
#             current_seq.AppendTuple(current_tuple)
#             #* Set register to finished
#             current_seq.OptFinished()
#         #* End of optimization procedure ==============================
#         print(f"[{root_id}] ===> Ready |")
#         #* Save the sequence as JSON 
#         if store_data:
#             # desired_file = f"{Result_file_path}/Temp_{data_name}_{root_id}"
#             current_seq.SaveJSON(desired_file)
#     return current_seq, root_id

#> Find the optimal sequence for a single node when its children are optimized
def PruneOptimalSingle(xdt, main_seq, root_id, **kwargs):
    store_data = False
    for key, value in kwargs.items():
        #* Store the data into the memory
        if key == "store_data":
            store_data = value
        #* Name of the dataset for proper data storage
        if key == "data_name":
            data_name = value
    if store_data:
        #* folder for storing test results
        store_path_test = ("./", "Sessions", data_name, "Temp") 
        Result_file_path = makeAdd(store_path_test)
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
        #* Save the sequence as JSON 
        if store_data:
            desired_file = f"{Result_file_path}/Temp_{data_name}_{root_id}"
            current_seq.SaveJSON(desired_file)
    return current_seq, root_id

#> Function for finding the optimal sequence of a node recursively
def PruneOptimal(xdt, goal_node, **kwargs):
    store_data = False
    for key, value in kwargs.items():
        #* Store the data into the memory
        if key == "store_data":
            store_data = value
        #* Name of the dataset for proper data storage
        if key == "data_name":
            data_name = value
    if store_data:
        #* folder for storing test results
        store_path_test = ("./", "Sessions", data_name, "Temp") 
        Result_file_path = makeAdd(store_path_test)
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
            current_seq, node_id = PruneOptimalSingle(xdt, main_seq, node_in_list, data_name = data_name, store_data = store_data)
            main_seq[node_in_list] = current_seq
            if node_in_list == goal_node:
                break
        if node_in_list == goal_node:
            break
    return main_seq[goal_node]

#> Function for finding the optimal sequence of a node recursively
def PruneOptimalParallel(xdt, goal_node, **kwargs):
    store_data = False
    for key, value in kwargs.items():
        #* Store the data into the memory
        if key == "store_data":
            store_data = value
        #* Name of the dataset for proper data storage
        if key == "data_name":
            data_name = value
    if store_data:
        #* folder for storing test results
        store_path_test = ("./", "Sessions", data_name, "Temp") 
        Result_file_path = makeAdd(store_path_test)
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
        result_async = [pool.apply_async(PruneOptimalSingle, args = (xdt, main_seq, node_in_list), kwds = {"store_data" : store_data, "data_name" : data_name}) for node_in_list in row_list]
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

#>A function to remove a series of nodes from a base DT
def series_prune(xdt, node_series):
    pruned_xdt = copy.deepcopy(xdt)
    #* Step by step remove branches
    for prune_node in node_series:
        current_dt = pruned_xdt.removeBranch(prune_node)
        pruned_xdt = ext_dt(current_dt)
    return pruned_xdt

def optPruneEval(xdt, current_seq, X_train, y_train, X_test, y_test):
    sequence_length = len(current_seq.cost)
    current_seq.APD = [0] * sequence_length
    current_seq.impurity = [0] * sequence_length
    current_seq.trn_accuracy = [0] * sequence_length
    current_seq.tst_accuracy = [0] * sequence_length
    pruning_data = []
    for idx in range(sequence_length):
        Pruned_xdt = series_prune(xdt, current_seq.removed[idx])
        Pruned_xdt.getRCFactor()
        Pruned_xdt.getImpFactor()
        current_seq.APD[idx] = Pruned_xdt.DT_APD
        current_seq.impurity[idx] = Pruned_xdt.DT_Imp

        # plot_name = data_name + "_" + methodName + "_" + f"{idx:04}"
        # make_DT_pdf(Pruned_xdt.base_dt, file_path, plot_name)

        #* Get depth statistical data
        min_depth, mean_depth, median_depth, max_depth = Pruned_xdt.getDepthStat()
        #* Evaluate the performance using TRAINING data
        train_score, train_APD, perf_data = Pruned_xdt.getPerformance(X_train, y_train)
        current_seq.trn_accuracy[idx] = train_score
        #* Evaluate the performance using TEST data
        test_score, test_APD, perf_data = Pruned_xdt.getPerformance(X_test, y_test)
        current_seq.tst_accuracy[idx] = test_score
        #* Append data to the matrix
        check_result = [idx, Pruned_xdt.DT_Imp, min_depth, mean_depth, median_depth, max_depth, train_score, train_APD, test_score, test_APD]
        pruning_data.append(check_result)
    #* Make a dataframe
    pruning_data_df = pd.DataFrame(pruning_data, columns = ["Step", "Impurity", "MinDepth", "MeanDepth", "MedianDepth", "MaxDepth", "Score_train", "APD_train", "Score_test", "APD_test"])
    return current_seq, pruning_data_df

#> Remove prune steps which are not optimal
def SequenceCleaning(init_seq):
    red_opt_length = len(init_seq.cost)
    useles_prune_list = []
    for i in range(red_opt_length):
        j = copy.copy(i) + 1
        while j < red_opt_length:
            # if init_seq.impurity[i] >= init_seq.impurity[j]:
            if init_seq.trn_accuracy[i] <= init_seq.trn_accuracy[j]:
                useles_prune_list.append(i)
                break
            else:
                j += 1

    #* remove elements
    init_seq.cost = [v for i, v in enumerate(init_seq.cost) if i not in useles_prune_list]
    init_seq.miss = [v for i, v in enumerate(init_seq.miss) if i not in useles_prune_list]
    init_seq.removed = [v for i, v in enumerate(init_seq.removed) if i not in useles_prune_list]
    init_seq.APD = [v for i, v in enumerate(init_seq.APD) if i not in useles_prune_list]
    init_seq.impurity = [v for i, v in enumerate(init_seq.impurity) if i not in useles_prune_list]
    init_seq.trn_accuracy = [v for i, v in enumerate(init_seq.trn_accuracy) if i not in useles_prune_list]
    init_seq.tst_accuracy = [v for i, v in enumerate(init_seq.tst_accuracy) if i not in useles_prune_list]
    return init_seq

#> To do the ORAP analysis
def ORAP_Analysis(dt_base, X_train, y_train, X_test, y_test, **kwargs):
    #* Name of this method for data logging
    methodName = "ORAP"
    print ("*******************************************")
    #+ In normal case data_name is not available
    data_name_available = False
    #+ In normal condition do not restore data
    restore_data = False
    #* ----- Parse input data
    for key, value in kwargs.items():
        #* Running in single or multi core mode
        if key == "use_parallel": 
            use_parallel = value
        #* Make PDF graph of the DT after each prune
        if key == "make_pdf":
            make_pdf = value
        #* Store the data into the memory
        if key == "store_data":
            store_data = value
        #* Name of the dataset for proper data storage
        if key == "data_name":
            data_name_available = True
            data_name = value
            store_path_performance = ("./", "Sessions", data_name, "Results") # folder for performance storage
            file_path = makeAdd(store_path_performance)
            store_name_performance = data_name + "_" + methodName  + "_Performance"
        if key == "restore_data":
            restore_data = value
    #* If data should be restored and available
    if restore_data and data_name_available and os.path.exists(os.path.join(*store_path_performance) + "/" + store_name_performance + ".csv") :
        print ("[" + time.strftime("%H:%M:%S") + "] " + methodName + " performance data RESTORED.")
        #* restore the performance data
        base_seq_Performance = RestoreData_csv(store_path_performance, store_name_performance)
        ORAP_seq_tested = 0
    else:
        print ("[" + time.strftime("%H:%M:%S") + "] " + methodName + " analysis STARTED.")
        #* ----- To avoide makedir issue in parallel mode
        if make_pdf:
            #* folder for pdf graphs
            store_path_pdf = ("./", "Sessions", data_name, "Graphs", methodName) 
            pdf_file_path = makeAdd(store_path_pdf)
        if store_data:
            #* folder for storing test results
            store_path_test = ("./", "Sessions", data_name, "Results") 
            Result_file_path = makeAdd(store_path_test)
        #* Build the extended tree object
        xdt = ext_dt(dt_base)
        #* Calculate the optimal sequence for the root of the whole DT
        goal_node = 0
        #* Decide about number of processors
        if use_parallel:
            goal_seq = PruneOptimalParallel(xdt, goal_node, data_name = data_name, store_data = store_data)
        else:
            goal_seq = PruneOptimal(xdt, goal_node, data_name = data_name, store_data = store_data)
        #* Get the sequence for root (complete sequence)
        ORAP_base_sequence = copy.copy(goal_seq)
        #* Sort sequence according to the cost
        sort_factor = "cost"
        ORAP_base_sequence.SortSequence(sort_factor)
        #* Add the original tree to the sequence
        ORAP_base_sequence.cost.insert(0, 0)
        ORAP_base_sequence.miss.insert(0, 0)
        ORAP_base_sequence.removed.insert(0, [])
        #* Get the evaluation data for this sequence
        ORAP_seq_tested, base_seq_Performance = optPruneEval(xdt, ORAP_base_sequence, X_train, y_train, X_test, y_test)

        if store_data:
            #* Save the sequence as JSON
            desired_name = data_name + "_" + methodName 
            desired_file = Result_file_path + "/" + desired_name + "_Sequence"
            ORAP_seq_tested.SaveJSON(desired_file)
            print(f"ORAP sequence saved as '{desired_name}_Sequence.json' to the memory.")
            #* Remove temp folder
            import shutil
            shutil.rmtree(f"./Sessions/{data_name}/Temp")
            #* Save performance data as csv
            store_name_performance = desired_name + "_Performance"
            storeData_csv(base_seq_Performance, store_path_test, store_name_performance)
            print(f"ORAP performance saved as '{desired_name}_Performance.csv' to the memory.")
        print ("[" + time.strftime("%H:%M:%S") + "] " + methodName + "analysis FINISHED.")
    print ("*******************************************")
    #* Give results back
    return base_seq_Performance, ORAP_seq_tested