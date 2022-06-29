import pandas as pd
import numpy as np
import re
from Levenshtein import distance as lev
import math
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from collections import deque
import scheduler

def starter_ex(unsequenced_schedule, file_name = 'output.xlsx' , k = 10, max_trials = 10000 ,shuffle = False):

    #'--unsequenced_schedule' => "Path to the  unsequenced schedule excel file")
    #'--priority_present'  => "if false adds a priority column with all tasks having equal priority" i.e. if priority column present in input then true else false
    #'--file_name' => 'output.xlsx' ,help="name for the sequenced excel file")
    #'--k' => "Number of best schedules that need to be present in the output file")
    #'--max_trials' => "Max Swaps for the SA optimiser")
    #'--shuffle' => "To shuffle the imported unscheduled file")
    xls = pd.ExcelFile(unsequenced_schedule) # input file
    df1 = pd.read_excel(xls, xls.sheet_names[0])
    df1['DATE'] = pd.to_datetime(df1["DATE"], format='%Y-%m-%d', errors='coerce')
    u_dates = (df1['DATE']).unique()
    u_dates.sort()
    u_dict =  dict(zip(u_dates, range(0,len(u_dates))))
    if ('PRIORITY' not in df1.columns):
        df1['PRIORITY'] = (df1['DATE']).apply(lambda x : u_dict.get(x))
    if (shuffle == 'True'):
        shuffled = df1.sample(frac=1).reset_index(drop=True)
    else:
        shuffled = (df1.copy()).reset_index(drop = True)   
    final = scheduler.priority_based_seperator_2(shuffled, k, max_trials)
    tc = len(final[0])
    for i in range(0,tc):
        (final[0])[i] = (final[0])[i].fillna(" ")
        (final[0])[i] = (final[0])[i].loc[:, ~((final[0])[i]).columns.str.contains('^Unnamed')]
    return final

def output_writer(final,file_name ='output.xlsx'):
    with pd.ExcelWriter(file_name) as writer:
        tc = len(final[0])
        for i in range(0,tc):
            ((final[0])[tc - i - 1]).to_excel(writer, sheet_name = 'S' + str(i) + 'CId- ' + str((final[1])[tc - i - 1]) , index=False)
    return None                                                
