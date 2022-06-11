# -*- coding: utf-8 -*-
"""reader_aproxtp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xMfot8xR5nPkVlpBzxDwOgO473JLH9B8
"""

import pandas as pd
import numpy as np
import re
!pip install python-Levenshtein
from Levenshtein import distance as lev
import math
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from collections import deque

def priority_based_seperator(shuffled, k = 10):
  a = shuffled['PRIORITY'].unique()
  a.sort()
  dfo = [pd.DataFrame(columns = shuffled.columns)]*k
  cost_arr = [0]*k
  for x in a:
    cur = (shuffled[shuffled['PRIORITY'] == x])
    clean_shuffle_pair = clean_and_format(cur)
    shuffled_pair = dark_light_seperator(cur,k)
    temp1 = TSP_SA(switch_over_cost_matrix(clean_shuffle_pair[0]), k)
    temp2 = TSP_SA(switch_over_cost_matrix(clean_shuffle_pair[1]),k)
    sequence_pair = temp1[1] + temp2[1] 
    cost_l = temp1[0]
    cost_d = temp2[0]
    curo = output(shuffled_pair, sequence_pair) 
    for i in range(0,k):
      dfo[i] = pd.concat([dfo[i],curo[i]])
      cost_arr[i] = cost_arr[i] + cost_l[i] + cost_d[i]
  return (dfo,cost_arr)

def priority_based_seperator_2(shuffled, k = 10):
  a = shuffled['PRIORITY'].unique()
  a.sort()
  dfo = [pd.DataFrame(columns = shuffled.columns)]*k
  cost_arr = [0]*k
  for x in a:
    cur = shuffled[shuffled['PRIORITY'] == x]
    clean_shuffle_pair = clean_and_format_2(cur)
    shuffled_pair = dark_light_seperator_2(cur,k)
    temp1 = TSP_SA(switch_over_cost_matrix(clean_shuffle_pair), k)
    sequence_pair = temp1[1]
    cost_l = temp1[0]
    curo = output_2(shuffled_pair, sequence_pair) 
    for i in range(0,k):
      dfo[i] = pd.concat([dfo[i],curo[i]])
      cost_arr[i] = cost_arr[i] + cost_l[i]
  return (dfo,cost_arr)

def dark_light_seperator(shuffled, k = 10):
  shuffled['COLOUR_TYPE'] = shuffled['COLOUR'].apply(lambda x: check_shade(re.sub("[^a-z0-9]+", '', x.lower()))%2) #colour type changes
  shuffled_d = (shuffled[shuffled['COLOUR_TYPE'] == 0]).copy()
  shuffled_l = (shuffled[shuffled['COLOUR_TYPE'] == 1]).copy()
  shuffled_d.reset_index(drop = True,inplace = True)
  shuffled_l.reset_index(drop = True, inplace = True)
  out = [shuffled_d]
  for x in range(0,k-1):
    out = out + [shuffled_d.copy()]
  for x in range(0,k):
    out = out + [shuffled_l.copy()]
  return out
def dark_light_seperator_2(shuffled, k = 10):
  shuffled['COLOUR_TYPE'] = shuffled['COLOUR'].apply(lambda x: check_shade(re.sub("[^a-z0-9]+", '', x.lower()))%2) #colour type changes
  shuffled.reset_index(drop = True,inplace = True)
  out = [shuffled.copy()]
  for x in range(0,k-1):
    out = out + [shuffled.copy()]
  return out

def clean_and_format(shuffled):
  df1c = shuffled.copy()
  df1c.drop(df1c.columns[6:], axis = 1, inplace = True)
  df1c.drop(df1c.columns[0:3], axis = 1, inplace = True)
  df1c.drop(df1c.columns[2:3], axis = 1, inplace = True)
  df1c['DESCRIPTION'] = df1c['DESCRIPTION'].apply(lambda x: x.lower())
  df1c['COLOUR'] = df1c['COLOUR'].apply(lambda x: x.lower())
  df1c['DESCRIPTION'] = df1c['DESCRIPTION'].apply(lambda x: re.sub("[^a-z0-9]+", ' ', x))  
  df1c['COLOUR'] = df1c['COLOUR'].apply(lambda x: re.sub("[^a-z0-9]+", '', x))
  df1c['COLOUR_TYPE'] = df1c['COLOUR'].apply(lambda x: check_shade(x))
  dfd = df1c[df1c['COLOUR_TYPE'] == 0]
  dfl = df1c[df1c['COLOUR_TYPE'] == 1]
  dfd.reset_index(drop = True, inplace = True)
  dfl.reset_index(drop = True, inplace = True)
  return (dfd,dfl)

def clean_and_format_2(shuffled):
  df1c = shuffled.copy()
  df1c.drop(df1c.columns[6:], axis = 1, inplace = True)
  df1c.drop(df1c.columns[0:3], axis = 1, inplace = True)
  df1c.drop(df1c.columns[2:3], axis = 1, inplace = True)
  df1c['DESCRIPTION'] = df1c['DESCRIPTION'].apply(lambda x: x.lower())
  df1c['COLOUR'] = df1c['COLOUR'].apply(lambda x: x.lower())
  df1c['DESCRIPTION'] = df1c['DESCRIPTION'].apply(lambda x: re.sub("[^a-z0-9]+", ' ', x))  
  df1c['COLOUR'] = df1c['COLOUR'].apply(lambda x: re.sub("[^a-z0-9]+", '', x))
  df1c['COLOUR_TYPE'] = df1c['COLOUR'].apply(lambda x: check_shade(x))
  df1c.reset_index(drop = True, inplace = True)
  return df1c

def output(shuffled_set, sequence_set):
  s = len(shuffled_set)
  dfos = []
  l = int (s/2)
  for i in range(0,s):
    shuffled_set[i] = shuffled_set[i].reindex(sequence_set[i])
  for i in range(0,l):
    temp = pd.concat([shuffled_set[i],shuffled_set[i+ l]])
    temp.reset_index(drop = True, inplace = True)
    temp.drop(["COLOUR_TYPE"], axis = 1, inplace = True)
    temp['DATE'] = temp['DATE'].dt.date
    dfos = dfos + [temp]
  return dfos
def output_2(shuffled_set, sequence_set):
  s = len(shuffled_set)
  for i in range(0,s):
    shuffled_set[i] = shuffled_set[i].reindex(sequence_set[i])
    shuffled_set[i].reset_index(drop = True, inplace = True)
    shuffled_set[i].drop(["COLOUR_TYPE"], axis = 1, inplace = True)
    shuffled_set[i]['DATE'] = shuffled_set[i]['DATE'].dt.date
  return shuffled_set
#file_name = 'output.xlsx'
#dfo.to_excel(file_name, index=False)

def jaccard_distance(w1, w2):
    list1 = w1.split()
    list2 = w2.split()
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return (1 - (float(intersection) / union))
def edit_distance(w1,w2):
  return  (float (lev(w1,w2))/ max(len(w1),len(w2)))
def cosine_distance(w1, w2):
    vec1 = Counter(w1.split())
    vec2 = Counter(w2.split())
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def dice_distance(w1, w2):
    list1 = w1.split()
    list2 = w2.split()
    intersection = len(list(set(list1).intersection(list2)))
    sum = (len(set(list1)) + len(set(list2))) 
    return (1 - (float(2* intersection) / sum))

def distance(u,v, scale = 1):
  if (u[1] == v[1]):
    return (1 + 3*edit_distance(u[0],v[0])) * scale #np.random.randint(3,7)
  else:
    if (check_shade(u[1]) == check_shade(v[1])):
      return (6 + 3*edit_distance(u[0],v[0])) * scale
    else : 
      return(11 + 3*edit_distance(u[0],v[0]))* scale

def switch_over_cost_matrix(df):
  dmd = pdist(df, distance) #assuming symmetric distance is present
  sqd = squareform(dmd)
  return sqd

def cost(G, s):
   l = 0
   for i in range(len(s)-1):
      l += G[s[i]][s[i+1]]
   #l += G[s[len(s)-1]][s[0]] 
   return l
def swap(s, m, n):
   i, j = min(m, n), max(m, n)
   s1 = s.copy()
   while i < j:
      s1[i], s1[j] = s1[j], s1[i]
      i += 1
      j -= 1
   return s1
def TSP_SA(G, k = 10):
   s = list(range(len(G)))
   c = cost(G, s)
   c_d = deque([c])
   s_d = deque([s])
   print(c) #only for illustration purposes
   ntrial = 1
   T = 30
   alpha = 0.99
   while ntrial <= 10000:
      n = np.random.randint(0, len(G))
      while True:
         m = np.random.randint(0, len(G))
         if n != m:
            break
      s1 = swap(s, m, n)
      c1 = cost(G, s1)
      if c1 < c:
         s, c = s1, c1
         if (c <= c_d[-1]):
           c_d.append(c)
           s_d.append(s)
         if (len(s_d) > k):
           s_d.popleft()
         if (len(c_d) > k):
           c_d.popleft()
      else:
         if np.random.rand() < np.exp(-(c1 - c)/T):
            s, c = s1, c1
      T = alpha*T
      ntrial += 1
   s_arr = list(s_d)
   c_arr = list(c_d)
   while(len(c_arr) < k):
     c_arr.append(c_arr[-1])
   while(len(s_arr) < k):
     s_arr.append(s_arr[-1])
   return (c_arr,s_arr)

def setup_colour():
  colourfile = pd.ExcelFile('/content/drive/MyDrive/Light & Dark Color Shade List in MES.xlsx')
  colourdf = pd.read_excel(colourfile, colourfile.sheet_names[0] )
  colourdf.drop(colourdf.columns[0:2], axis = 1, inplace = True)
  colourdf= colourdf.iloc[5:]
  colourdf.columns = colourdf.iloc[0]
  colourdf = colourdf.iloc[1:]
  colourdf.reset_index(drop = True, inplace = True)
  colourdf.fillna("bad", inplace = True)
  colourdf["LIGHT"] = colourdf["LIGHT"].apply(lambda x: re.sub("[^a-z0-9]+", '', x.lower()))
  colourdf["DARK"] = colourdf["DARK"].apply(lambda x: re.sub("[^a-z0-9]+", '', x.lower()))
  light = colourdf["LIGHT"].unique()
  dark = colourdf["DARK"].unique()
  return (dark, light)
dark , light = setup_colour()
def check_shade(colour):
  if (colour in light):
    return 1
  elif  (colour in dark):
    return 0
  else :
    print("missing colour" + " " + colour)
    return 1

"""here is output stuff"""

xls = pd.ExcelFile('/content/drive/MyDrive/UNSEQUENCE PLAN 26 & 27 may priority.xlsx') # input file
file_name = 'output.xlsx'

df1 = pd.read_excel(xls, xls.sheet_names[0] )
df1['DATE'] = pd.to_datetime(df1["DATE"], format='%Y-%m-%d', errors='coerce')
shuffled = df1.sample(frac=1).reset_index(drop=True)
final = priority_based_seperator_2(shuffled)
with pd.ExcelWriter('output.xlsx') as writer:
  tc = len(final[0])
  for i in range(0,tc):
    ((final[0])[tc - i - 1]).to_excel(writer, sheet_name = 'S' + str(i) + 'CId- ' + str((final[1])[tc - i - 1]) , index=False)

final[1]

