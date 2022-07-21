import pandas as pd
import numpy as np
import re
from Levenshtein import distance as lev
import math
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from collections import deque
import scheduler

def backlog_reader(source):
    back_df = pd.read_csv(source)
    back_idx = -1
    for i in range(0,len(back_df)):
        if (back_df.iloc[i])["DESCRIPTION"] == "BackLog":
            back_idx = i
            break
    if(back_idx == -1):
        return None
    else:
        back_df = back_df.iloc[back_idx+1:]
        back_df = back_df[back_df.columns[:-2]]
        back_df['DATE'] = pd.to_datetime(back_df["DATE"], format='%Y-%m-%d', errors='coerce')
        if(back_df.iloc[0]["PRIORITY"] == 1):
            back_df["PRIORITY"] = 0
        elif ((back_df.iloc[0]["PRIORITY"] == 2)) :
            back_df["PRIORITY"] = 2
        return back_df
def starter_ex(unsequenced_schedule, file_name = 'output.xlsx' , k = 1, max_trials = 10000 ,shuffle = False, backlog1 = None, backlog2 = None):

    #'--unsequenced_schedule' => "Path to the  unsequenced schedule excel file")
    #'--priority_present'  => "if false adds a priority column with all tasks having equal priority" i.e. if priority column present in input then true else false
    #'--file_name' => 'output.xlsx' ,help="name for the sequenced excel file")
    #'--k' => "Number of best schedules that need to be present in the output file")
    #'--max_trials' => "Max Swaps for the SA optimiser")
    #'--shuffle' => "To shuffle the imported unscheduled file")
    #xls = pd.ExcelFile(unsequenced_schedule) # input file
    df1 = unsequenced_schedule#pd.read_csv(unsequenced_schedule, index_col = 0)
    df1['DATE'] = pd.to_datetime(df1["DATE"], format='%d-%b-%y', errors='coerce') #'%d-%b-%Y' '%Y-%m-%d'
    u_dates = (df1['DATE']).unique()
    u_dates.sort()
    u_dict =  dict(zip(u_dates, range(0,len(u_dates))))
    if ('PRIORITY' not in df1.columns):
        df1['PRIORITY'] = (df1['DATE']).apply(lambda x : u_dict.get(x.to_datetime64()))
    if (shuffle == 'True'):
        shuffled = df1.sample(frac=1).reset_index(drop=True)
    else:
        shuffled = (df1.copy()).reset_index(drop = True)   
    if (backlog1 != None):
        blog1 = backlog_reader(backlog1)
        blog1.columns = shuffled.columns
        print(blog1)
        shuffled = pd.concat([shuffled,blog1], axis=0)
    if (backlog2 != None):
        blog2 = backlog_reader(backlog2)
        blog2.columns = shuffled.columns
        print(blog2)
        shuffled = pd.concat([shuffled,blog2], axis=0)
    print(shuffled)
    final = scheduler.priority_based_seperator_2(shuffled, k, max_trials)
    tc = len(final[0])
    for i in range(0,tc):
        (final[0])[i] = (final[0])[i].fillna(" ")
        (final[0])[i] = (final[0])[i].loc[:, ~((final[0])[i]).columns.str.contains('^Unnamed')]
        (final[0])[i].reset_index(drop = True, inplace = True)
    return final

def sns_apc_adder(sequenced):
    sns_pkts_apc_col = pd.read_csv("SNS packets and apc colours - Sheet1.csv")
    sns_pkts_spc_seq = pd.read_csv("SNS_apc_list - Sheet1.csv")
    sns_pkts_apc_col_dict = dict(zip(sns_pkts_apc_col["APC ItemCode"],sns_pkts_apc_col["Colour"]))
    sns_pkts_spc_seq.fillna("Missing", inplace = True)
    sns_msku_ic = sns_pkts_spc_seq["Main SKU IC"]
    sns_apc_pkt_lst = []
    for x in range (0, len(sns_pkts_spc_seq)):
        st = 3
        out_arr = []
        while (st < len(sns_pkts_spc_seq.loc[x])):
            out_arr.append([(sns_pkts_spc_seq.loc[x])[st],(sns_pkts_spc_seq.loc[x])[st+1] ])
            st = st + 2
        sns_apc_pkt_lst.append([out_arr])
    sns_pkt_apc_dict = dict(zip(sns_msku_ic,sns_apc_pkt_lst))
    for x in range(0,len(sequenced)):
        if (sequenced.loc[x])[2] in  sns_pkt_apc_dict.keys():
            date = (sequenced.loc[x])[0]
            qty = (sequenced.loc[x])[5]
            prior = (sequenced.loc[x])[6]
            cur_insert = sns_pkt_apc_dict.get((sequenced.loc[x])[2])
            cur_insert = cur_insert[0]
            for c in range(0, len(cur_insert)):
                index = x + (0.1*(c+1))
                sequenced.loc[index + 0.01] = [date, " ", cur_insert[c][0], cur_insert[c][1],sns_pkts_apc_col_dict.get( cur_insert[c][0]), qty, prior]
    sequenced = sequenced.sort_index(ascending=True)
    sequenced.reset_index(drop = True,inplace = True)
    return sequenced


    
def output_writer(final,file_name ='output.xlsx'):
    with pd.ExcelWriter(file_name) as writer:
        tc = len(final[0])
        for i in range(0,tc):
            ((final[0])[tc - i - 1]).to_excel(writer, sheet_name = 'S' + str(i) + 'CId- ' + str((final[1])[tc - i - 1]) , index=False)
    return None                                                
