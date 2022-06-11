import pandas as pd
import numpy as np
import re
from Levenshtein import distance as lev
import math
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from collections import deque
import argparse
import scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Line Balancer Ripik')
    parser.add_argument('--unsequenced_schedule',type=str,required=True,help="Path to the  unsequenced schedule excel file")
    parser.add_argument('--priority_present' , type = bool, required= False, default=False,help="if false adds a priority column with all tasks having equal priority")
    parser.add_argument('--file_name',type=str,required=False, default = 'output.xlsx' ,help="name for the sequenced excel file")
    parser.add_argument('--k' , type=int,required=False, default=10,help="Number of best schedules that need to be present in the output file")
    parser.add_argument('--max_trials', type = int, required = False, default= 5, help = "Max Swaps for the SA optimiser")
    parser.add_argument('--shuffle', type = bool, required = False, default = False, help = "To shuffle the imported unscheduled file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    xls = pd.ExcelFile(args.unsequenced_schedule) # input file
    file_name = args.file_name
    df1 = pd.read_excel(xls, xls.sheet_names[0])
    df1['DATE'] = pd.to_datetime(df1["DATE"], format='%Y-%m-%d', errors='coerce')
    if (args.priority_present == False):
        df1['PRIORITY'] = 1
    if (args.shuffle == True):
        shuffled = df1.sample(frac=1).reset_index(drop=True)
    else:
        shuffled = (df1.copy()).reset_index(drop = True)
    
    final = scheduler.priority_based_seperator_2(shuffled, args.k, args.max_trials)
    with pd.ExcelWriter(file_name) as writer:
        tc = len(final[0])
        for i in range(0,tc):
            ((final[0])[tc - i - 1]).to_excel(writer, sheet_name = 'S' + str(i) + 'CId- ' + str((final[1])[tc - i - 1]) , index=False)
