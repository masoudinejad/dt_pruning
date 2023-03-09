from cmath import nan
import numpy as np
import pandas as pd
import copy
from progress.bar import Bar

# %% Class definition
class PruneSequence:
    # > Initialize the sequence
    def __init__(self):
        self.cost = []
        self.miss = []
        self.removed = []
        self.APD = []
        self.impurity = []
        self.trn_accuracy = []
        self.tst_accuracy = []
        self.report = False # + to report calculations
        self.optimized = False # + to know if optimization is finished
        self.sorted = "None" # + to know if sorted
        
    # > Set a register to provide operational info printed
    def getReport(self, stat):
        self.report = stat
    
    # > A register to know that this is a complete sequence for a node (branch/tree)
    def OptFinished(self):
        self.optimized = True
        
    # * Print the massages if report is requested
    def ReportPrint(self, msg):
        if self.report:
            print(msg)

    # > show the content of the sequence
    def ShowSequence(self):
        print(f"--------------------")
        print(f"D(APD):\t{self.cost}")
        print(f"Miss.:\t{self.miss}")
        print(f"Rmvd.:\t{self.removed}")
        # print(f"APD:\t{self.APD}")
        # print(f"Imp.:\t{self.impurity}")
        # print(f"Stat:\t{self.optimized}")
        # print(f"Sort:\t{self.sorted}")
        print(f"--------------------")

    #> To check that former elements in the list are not worse than last added one
    def cleanFormerElements(self, tuple2add):
        current_cost = tuple2add[0]
        current_miss = tuple2add[1]
        #* Is there any former element with less cost change but higher miss? 
        former_check = np.logical_and(np.array(self.cost) < current_cost, np.array(self.miss) >= current_miss)
        if np.sum(former_check) == 1: # There is a single non-optimal element
            #* get the index of non-optimal point
            non_opt_idx = np.where(former_check == True)[0][0]
            #* Remove that specific point
            del self.cost[non_opt_idx]
            del self.miss[non_opt_idx]
            del self.removed[non_opt_idx]
            self.ReportPrint("Single non-optimal former element removed!")
        elif np.sum(former_check) > 1:
            #* Remove them
            self.cost = [self.cost[idx] for idx in range(len(self.cost)) if former_check[idx]==False]
            self.miss = [self.miss[idx] for idx in range(len(self.miss)) if former_check[idx]==False]
            self.removed = [self.removed[idx] for idx in range(len(self.removed)) if former_check[idx]==False]
            self.ReportPrint("Multiple non-optimal former elements removed!")
    
    # > add a tuple to the sequence
    def AppendTuple(self, tuple2add):
        # * check that the passed tuple is not empty
        if not tuple2add:
            self.ReportPrint("Passed tuple is empty!")
        else:
            # * if the sequence is empty, simply initiate lists
            if not self.cost:
                self.ReportPrint("Initial addition")
                self.cost = [tuple2add[0]]
                self.miss = [tuple2add[1]]
                self.removed = [tuple2add[2]]
            else: #* This is a non-empty tuple
                #* Is there any better element
                current_cost = tuple2add[0]
                current_miss = tuple2add[1]
                # * If cost not there
                if not current_cost in self.cost:
                    optimality_check1 = np.logical_and(np.array(self.cost) > current_cost, np.array(self.miss) <= current_miss)
                    #* There is no element with higher cost and less miss (better performance)
                    if np.sum(optimality_check1) == 0:
                        self.ReportPrint("[Append]")
                        self.cost.append(tuple2add[0])
                        self.miss.append(tuple2add[1])
                        self.removed.append(tuple2add[2])
                        #* Is there any former element with less cost change but higher miss? 
                        self.cleanFormerElements(tuple2add)
                else: # * If same cost exists
                    # * Find the index of that element
                    same_cost_idx = self.cost.index(current_cost)
                    # * check if its error is different
                    current_miss = tuple2add[1]
                    if current_miss < self.miss[same_cost_idx]:
                        # * replace it in case of less missclassification
                        self.ReportPrint("[Replace] based on the miss rate")
                        self.miss[same_cost_idx] = current_miss
                        self.removed[same_cost_idx] = tuple2add[2]
                        #* Is there any former element with less cost change but higher miss? 
                        self.cleanFormerElements(tuple2add)
                    # * if both cost and miss are equal
                    elif current_miss == self.miss[same_cost_idx]:
                        # * replace only if it makes the tree smaller (more nodes are removed)
                        self.ReportPrint("[No Replace] original sequence prunes more nodes")
                        if len(tuple2add[2]) > len(self.removed[same_cost_idx]):
                            self.ReportPrint("[Replace] based on the prune nodes")
                            self.removed[same_cost_idx] = tuple2add[2]
                            #* Is there any former element with less cost change but higher miss? 
                            self.cleanFormerElements(tuple2add)

    # > Appending a sequence to this (not combining!)
    def AppendSequence(self, seq2add):
        # * go through tuples 1by1 and append them to the sequence
        for idx in range(len(seq2add.cost)):
            tuple2add = [seq2add.cost[idx], seq2add.miss[idx], seq2add.removed[idx]]
            self.AppendTuple(tuple2add)

    #> Sort the sequence according to any desired factor
    def SortSequence(self, factor):
        import numpy as np
        #* Select the proper array according to the input
        if factor == "cost":
            base_list = np.array(self.cost)
        elif factor == "miss":
            base_list = np.array(self.miss)
        elif factor == "no_node":
            #* find number of pruned node in each step
            rmv_length = [len(ele) for ele in(self.removed)]
            base_list = np.array(rmv_length)
        else:
            raise TypeError("Provided sort factor is not available. Use: cost, miss, no_node")
        #* Find the sorted index list for the array
        sort_index = np.argsort(base_list)
        #* get the size of sorted list
        srt_length = len(sort_index)
        #* Make empty list to enter sorted values
        new_cost = [None] * srt_length
        new_miss = [None] * srt_length
        new_rmvd = [None] * srt_length
        #* Replace sorted values into the class
        for idx in range(srt_length):
            new_cost[idx] = self.cost[sort_index[idx]]
            new_miss[idx] = self.miss[sort_index[idx]]
            new_rmvd[idx] = self.removed[sort_index[idx]]
        #* Replace the parameters in the class with the sorted one
        self.cost = new_cost
        self.miss = new_miss
        self.removed = new_rmvd
        #* Update the sorted register
        self.sorted = factor
    
    #> Saving sequence to JSON
    def SaveJSON(self, desired_name):
        import json
        seq_length = len(self.cost)
        seq_entry = {}
        for seq_iter in range(seq_length):
            seq_name = F"Seq_{seq_iter}"
            seq_entry[seq_name] = {"Iteration": seq_iter, 
                                "Cost": int(self.cost[seq_iter]),
                                "Miss": int(self.miss[seq_iter]),
                                "NodeRemoved": [int(ele) for ele in self.removed[seq_iter]]
                                }
        save_name = desired_name + ".json"
        with open(save_name, 'w') as f:
            json.dump(seq_entry, f, indent = 4)

# > Combine two sequences to build a new sequence
def CombineSequence(seq1, seq2, parent):
    # + make a new sequence
    combined_seq = PruneSequence()
    # * Append both sequences if non-empty
    if seq1.cost:
        combined_seq.AppendSequence(seq1)
    if seq2.cost:
        combined_seq.AppendSequence(seq2)
    # * if both non-empty, built combinations
    if seq1.cost and seq2.cost:
        #* Status bar
        show_range = len(seq1.cost)
        status_msg = f"Combining for node {parent}\t"
        bar = Bar(status_msg, max=show_range)
        # * Build combinations of both
        for idx_1 in range(len(seq1.cost)):
            for idx_2 in range(len(seq2.cost)):
                # * calculate parameters
                combined_cost = seq1.cost[idx_1] + seq2.cost[idx_2]
                combined_miss = seq1.miss[idx_1] + seq2.miss[idx_2]
                combined_removed = seq1.removed[idx_1] + seq2.removed[idx_2]
                # + build the new tuple
                combined_tuple = [combined_cost, combined_miss, combined_removed]
                # * Append this tuple to the combined sequence
                combined_seq.AppendTuple(combined_tuple)
            #* Update status
            bar.next()
    # print("\n")
    return combined_seq

def getSeqPerformance(APD, Score):
    #* Fining the delta APD
    APD_diff = APD.diff()
    #* Find the performance between two subsequent steps
    partial_perf = abs(APD_diff) * Score
    #* Find the range of APD
    delta_APD = APD.max() - APD.min()
    return partial_perf, delta_APD

#> Function to remove a single step (element) from final sequence
def RemoveOneStep(input_perf_df, experiment_type):
    if experiment_type == "Train":
        APD_array = input_perf_df.APD_train
        Acc_array = input_perf_df.Score_train
    elif experiment_type == "Test":
        APD_array = input_perf_df.APD_test
        Acc_array = input_perf_df.Score_test
    else:
        raise TypeError("Only 'Train' or 'Test' is allowed!")
    #* Get the performance of each element
    partial_perf, delta_APD = getSeqPerformance(APD_array, Acc_array)
    #* Avoid last one (APD=0) to be removed
    partial_perf.iat[-1] = nan
    # print(partial_perf)
    #* Find minimum effect
    step2remove = partial_perf.idxmin()
    # print(F"to remove: {step2remove}")
    # print(input_perf_df)
    #* Remove the related step
    new_df = input_perf_df.drop(step2remove)
    new_df = new_df.reset_index(drop=True)
    return new_df, step2remove

#> To calculate the overall performance for the whole range
def getPerformance(APD, Score):
    partial_perf, delta_APD = getSeqPerformance(APD, Score)
    #* Find the sum of all steps
    sum_perf = partial_perf.sum()
    mean_performance = 100 * sum_perf / delta_APD
    return mean_performance, sum_perf

#> Resize the sequence into a new desired size
def seqResize(perf, desired_size, experiment_type):
    perf_resized = copy.deepcopy(perf)
    if experiment_type == "Train":
        sort_col = "APD_train"
    elif experiment_type == "Test":
        sort_col = "APD_test"
    perf_resized = perf_resized.sort_values(by=[sort_col], ascending=False, ignore_index=True)
    #+ Storing the steps removed
    step_removed = []
    #* Remove steps till reaching the goal
    while perf_resized.shape[0] > desired_size:
        perf_resized, step2remove = RemoveOneStep(perf_resized, experiment_type)
        step_removed.append(step2remove)
    # perf_resized.sort_values(by=[sort_col], inplace=True, ascending=False)
    removal_data = {'Remove Order':range(len(step_removed)), 
            'Seq2Remove':step_removed}
    removal_data_df = pd.DataFrame(removal_data)
    return perf_resized, removal_data_df