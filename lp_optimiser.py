

import pulp
import numpy as np
import pandas as pd
import random

def family_corrector(family):
    if (family == 'SL Precoated'):
        return 'X2 Precoated'
    elif (family == 'SLD-SCREENPRINTING NEW'):
        return 'SL Door'
    elif (family == 'X2 Dresser'):
        return 'X2 Body'
    elif (family == 'X2-SCREENPRINTING'):
        return 'X2 Door'
    elif (family == 'SL Blend'):
        return 'SL Door'
    elif (family == 'SL Dual'):
        return 'SL Door'
    elif (family == 'SLIMLINE DUAL PRECOATED'):
        return 'X2 Precoated'
    else:
        return family

def TP_setter(colour, TP, ytp,gtp,wtp):
    if (colour == "Yellow"):
        return ytp
    if (colour == "Green"):
        return gtp
    if (colour == "White"):
        return wtp
    if (colour == "Red"):
        return TP
    else:
        return 0

def mask_gen(family):
    present_families = ['Slide & Store' , 'SL Body', 'SL Door', 'X2 Body', 'X2 Door', 'X2 Precoated', 'Platina' ]
    if family in present_families:
        return True
    else :
        return False
col_dic_df = pd.read_csv('bpr_colour_mas.csv', index_col=0)
item_col_dict = dict(zip(col_dic_df.index,col_dic_df["Powder Colour"]))
family_line = pd.read_excel("Family_line.xlsx")
families = family_line["Families"]
lines = family_line["Line-Family"]
family_line_dict = dict(zip(families,lines))
ictpf_df = pd.read_csv("ItemCode_to_Production_family.csv", index_col = 0)
ictpf_dic = dict(zip(ictpf_df.index, ictpf_df['Product Family']))

def backlog_reader(source):
    back_df = pd.read_csv(source, index_col=0)
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
        back_df["Family"] = back_df["ITEMCODE"].apply(lambda x : ictpf_dic.get(x, "No Fam"))
        back_df["Family"] = back_df["Family"].apply(lambda x : family_corrector(x))
        return back_df

vect_mask = np.vectorize(mask_gen)
xs_arr = [0,0,0,0,0,0,0]
max_arr = [0,0,0,0,0,0,0]
sns_max = 50
slb_max = 200
sld_max = 600
x2b_max = 100
x2d_max = 200
x2p_max = 200
plt_max = 50
pri1 = [0,0,0,0,0,0,0]
sns_pri1 = 0
slb_pri1 = 0
sld_pri1 = 0
x2b_pri1 = 0
x2d_pri1 = 0
x2p_pri1 = 0
plt_pri1 = 0
mkact_arr = [0,0,0,0,0,0,0]
mk_arr = [0,0,0,0,0,0,0]
sns_make = min(sns_max,sns_pri1)
slb_make = min(slb_max,slb_pri1)
sld_make = min(sld_max,sld_pri1)
x2b_make = min(x2b_max,x2b_pri1)
x2d_make = min(x2d_max,x2d_pri1)
x2p_make = min(x2p_max,x2p_pri1)
plt_make = min(plt_max,plt_pri1)
sns_xs = sns_max - sns_make
slb_xs = slb_max - slb_make
sld_xs = sld_max - sld_make
x2b_xs = x2b_max - x2b_make
x2d_xs = x2d_max - x2d_make
x2p_xs = x2p_max - x2p_make
plt_xs = plt_max - plt_make

def qty_p1(qty, family,colour, buffer):
        out = 0
        if (family =="Slide & Store" ):
            out += int(qty*mk_arr[0]/pri1[0])
            mkact_arr[0] += out
            if (colour == "Black" and xs_arr[0] > 0):
                out += min( int (buffer * 0.05), xs_arr[0] )
                xs_arr[0] -= min( int (buffer * 0.05), xs_arr[0] )
        if (family =='SL Body' ):
            out += int(qty*mk_arr[1]/pri1[1])
            mkact_arr[1] += out
            if (colour == "Black" and xs_arr[1] > 0):
                out += min( int (buffer * 0.05), xs_arr[1] )
                xs_arr[1] -= min( int (buffer * 0.05), xs_arr[1] )
        if (family =='SL Door' ):
            out += int(qty*mk_arr[2]/pri1[2])
            mkact_arr[2] += out
            if (colour == "Black" and xs_arr[2] > 0):
                out += min( int (buffer * 0.05), xs_arr[2] )
                xs_arr[2] -= min( int (buffer * 0.05), xs_arr[2] )
        if (family =='X2 Body' ):
            out += int(qty*mk_arr[3]/pri1[3])
            mkact_arr[3] += out
            if (colour == "Black" and xs_arr[3] > 0):
                out += min( int (buffer * 0.05), xs_arr[3] )
                xs_arr[3] -= min( int (buffer * 0.05), xs_arr[3] )
        if (family =='X2 Door' ):
            out += int(qty*mk_arr[4]/pri1[4])
            mkact_arr[4] += out
            if (colour == "Black" and xs_arr[4] > 0):
                out += min( int (buffer * 0.05), xs_arr[4] )
                xs_arr[4] -= min( int (buffer * 0.05),xs_arr[4] )
        if (family =='X2 Precoated' ):
            out += int(qty*mk_arr[5]/pri1[5])
            mkact_arr[5] += out
            if (colour == "Black" and xs_arr[5] > 0):
                out += min( int (buffer * 0.05), xs_arr[5] )
                xs_arr[5] -= min( int (buffer * 0.05), xs_arr[5] )
        if (family =='Platina' ):
            out += int(qty*mk_arr[6]/pri1[6])
            mkact_arr[6] += out
            if (colour == "Black" and xs_arr[6] > 0):
                out += min( int (buffer * 0.05), xs_arr[6] )
                xs_arr[6] -= min( int (buffer * 0.05), xs_arr[6] )
        return out

def qty_p2(qty, colour, buffer, family):
    out = qty
    if (colour == "Black" and qty == 0):
        if (family =="Slide & Store" ):
            if (max_arr[0] > 0):
                out += min( int (buffer * 0.05), max_arr[0] )
                max_arr[0] -= min( int (buffer * 0.05), max_arr[0] )
        if (family =='SL Body' ):
            if (max_arr[1] > 0):
                out += min( int (buffer * 0.05), max_arr[1] )
                max_arr[1] -= min( int (buffer * 0.05), max_arr[1])
        if (family =='SL Door' ):
            if (max_arr[2] > 0):
                out += min( int (buffer * 0.05), max_arr[2] )
                max_arr[2] -= min( int (buffer * 0.05), max_arr[2] )
        if (family =='X2 Body' ):
            if (max_arr[3] > 0):
                out += min( int (buffer * 0.05), max_arr[3] )
                max_arr[3] -= min( int (buffer * 0.05), max_arr[3] )
        if (family =='X2 Door' ):
            if (max_arr[4] > 0):
                out += min( int (buffer * 0.05), max_arr[4] )
                max_arr[4] -= min( int (buffer * 0.05), max_arr[4] )
        if (family =='X2 Precoated' ):
            if (max_arr[5] > 0):
                out += min( int (buffer * 0.05), max_arr[5])
                max_arr[5] -= min( int (buffer * 0.05), max_arr[5] )
        if (family =='Platina' ):
            if (max_arr[6] > 0):
                out += min( int (buffer * 0.05), max_arr[6] )
                max_arr[6] -= min( int (buffer * 0.05), max_arr[6] )
    return out

def d_scheduler(source, backlogl1 = None, backlogl2 = None):
    xls = pd.ExcelFile(source)
    bpr = pd.read_excel(xls, xls.sheet_names[0])

    date = "-".join((((bpr.iloc[1])["Unnamed: 1"]).split(" ")[-1]).split("-")[0:3])
    bpr = bpr.iloc[8:]
    bpr.columns = bpr.iloc[0]
    bpr = bpr.iloc[1:]
    bpr.reset_index(inplace= True, drop = True)
    bpr = bpr[bpr["Location"] == "Khalapur K2"]
    bpr["Product Family"] = bpr["Product Family"].apply(lambda x :family_corrector(x) )
    bpr_reg_norm =  bpr[bpr['Norm Category'] != "Ecom"] #316
    bpr_reg_norm.drop([bpr_reg_norm.columns[0]], axis = 1,inplace = True)
    bpr_reg_norm.reset_index(inplace= True, drop = True)
    bpr_reg_norm.set_index("Item Code", inplace = True)

    bpr_reg_norm['TP'] = 0
    bpr_reg_norm['CB'] = 0
    bpr_reg_norm['ytp'] = 0
    bpr_reg_norm['gtp'] = 0
    bpr_reg_norm['wtp'] = 0

    tp = pd.read_csv('probability data/continious_black.csv', index_col=0)

    cb = pd.read_csv('probability data/continious_black.csv', index_col=0)

    yp = pd.read_csv('probability data/yt_prob.csv', index_col=0)
    gp = pd.read_csv('probability data/gt_prob.csv', index_col=0)
    wp = pd.read_csv('probability data/wt_prob.csv', index_col=0)

    for x in (bpr_reg_norm.index):
        if x in tp.index:
            c_row = bpr_reg_norm.loc[x, bpr_reg_norm.columns]
            count_row = (tp.loc[x, tp.columns])
            count_row_2 = (cb.loc[x, cb.columns])
            count_row_3 = (yp.loc[x, yp.columns])
            count_row_4 = (gp.loc[x, gp.columns])
            count_row_5 = (wp.loc[x, wp.columns])
            c_row['TP'] = count_row['0']
            c_row['CB'] = count_row_2['0']
            c_row['ytp'] = count_row_3['0']
            c_row['gtp'] = count_row_4['0']
            c_row['wtp'] = count_row_5['0']
            bpr_reg_norm.loc[x, bpr_reg_norm.columns] = c_row
            if ((c_row != bpr_reg_norm.loc[x, bpr_reg_norm.columns]).all()):
                print("error")

    bpr_reg_norm["TP"] = bpr_reg_norm.apply(lambda row : TP_setter(row["Colour Status"][1],row["TP"], row["ytp"], row["gtp"], row["wtp"] ), axis = 1)

    bpr_reg_norm.drop(bpr_reg_norm.columns[-3:], inplace = True, axis = 1)

    #present_families = ['Slide & Store' , 'SL Body', 'SL Door', 'X2 Body', 'X2 Door', 'X2 Precoated', 'Platina']


    fam_mask = vect_mask(np.array(bpr_reg_norm["Product Family"].reset_index(drop = True)))

    bpr_reg_norm = bpr_reg_norm[fam_mask]

    bpr_reg_norm['QTY'] = 0

    bpr_reg_norm["Pri1"] = bpr_reg_norm.apply(lambda row : max(row['Pending Orders'] + row['Branch TOG'] - row['Stock Qty'] - row['Intransit Stock'] - row['Stock at Plant'],0), axis = 1)
    df_make = (bpr_reg_norm[(bpr_reg_norm["Pri1"]>0)]).copy()
    sns_max = 50
    slb_max = 200
    sld_max = 600
    x2b_max = 100
    x2d_max = 200
    x2p_max = 200
    plt_max = 50
    present_families = ['Slide & Store' , 'SL Body', 'SL Door', 'X2 Body', 'X2 Door', 'X2 Precoated', 'Platina' ]
    if (backlogl1 != None):
        blog1 = backlog_reader(backlogl1)
        for x in range(0,len(blog1)):
            if ((blog1.iloc[x])["Family"] in present_families):
                family = (blog1.iloc[x])["Family"]
                qty =  int((blog1.iloc[x])["QTY"])
                if (family =="Slide & Store" ):
                    sns_max = max(0,sns_max - qty)
                if (family =='SL Body' ):
                    slb_max = max(0, slb_max - qty)
                if (family =='SL Door' ):
                    sld_max = max(0, sld_max - qty)
                if (family =='X2 Body' ):
                    x2b_max = max(0, x2b_max - qty)
                if (family =='X2 Door' ):
                    x2d_max = max(0,x2d_max - qty)
                if (family =='X2 Precoated' ):
                    x2p_max = max(0, x2p_max - qty)
                if (family =='Platina' ):
                    plt_max = max(0, plt_max - qty)
        


    if (backlogl2 != None):
        blog2 = backlog_reader(backlogl2)
        for x in range(0,len(blog2)):
            if ((blog2.iloc[x])["Family"] in present_families):
                family = (blog2.iloc[x])["Family"]
                qty =  int((blog2.iloc[x])["QTY"])
                if (family =="Slide & Store" ):
                    sns_max = max(0,sns_max - qty)
                if (family =='SL Body' ):
                    slb_max = max(0, slb_max - qty)
                if (family =='SL Door' ):
                    sld_max = max(0, sld_max - qty)
                if (family =='X2 Body' ):
                    x2b_max = max(0, x2b_max - qty)
                if (family =='X2 Door' ):
                    x2d_max = max(0,x2d_max - qty)
                if (family =='X2 Precoated' ):
                    x2p_max = max(0, x2p_max - qty)
                if (family =='Platina' ):
                    plt_max = max(0, plt_max - qty)
    
    sns_pri1 = sum((df_make[df_make["Product Family"] == "Slide & Store"])["Pri1"])
    slb_pri1 = sum((df_make[df_make["Product Family"] == 'SL Body'])["Pri1"])
    sld_pri1 = sum((df_make[df_make["Product Family"] == 'SL Door'])["Pri1"])
    x2b_pri1 = sum((df_make[df_make["Product Family"] == 'X2 Body'])["Pri1"])
    x2d_pri1 = sum((df_make[df_make["Product Family"] == 'X2 Door'])["Pri1"])
    x2p_pri1 = sum((df_make[df_make["Product Family"] == 'X2 Precoated'])["Pri1"])
    plt_pri1 = sum((df_make[df_make["Product Family"] == 'Platina'])["Pri1"])
    pri1_t = [sns_pri1,slb_pri1,sld_pri1,x2b_pri1,x2d_pri1,x2p_pri1,plt_pri1]
    print(pri1_t)
    sns_make = min(sns_max,sns_pri1)
    slb_make = min(slb_max,slb_pri1)
    sld_make = min(sld_max,sld_pri1)
    x2b_make = min(x2b_max,x2b_pri1)
    x2d_make = min(x2d_max,x2d_pri1)
    x2p_make = min(x2p_max,x2p_pri1)
    plt_make = min(plt_max,plt_pri1)
    mk_arr_t = [sns_make,slb_make,sld_make,x2b_make,x2d_make,x2p_make,plt_make]
    print(mk_arr_t)
    sns_xs = sns_max - sns_make
    slb_xs = slb_max - slb_make
    sld_xs = sld_max - sld_make
    x2b_xs = x2b_max - x2b_make
    x2d_xs = x2d_max - x2d_make
    x2p_xs = x2p_max - x2p_make
    plt_xs = plt_max - plt_make
    xs_arr_t = [sns_xs, slb_xs, sld_xs, x2b_xs, x2d_xs, x2p_xs, plt_xs]
    for i in range(0,7):
        pri1[i] = pri1_t[i]
        mk_arr[i] = mk_arr_t[i]
        xs_arr[i] = xs_arr_t[i]

    df_make["QTY"] = df_make.apply(lambda row : qty_p1(row["Pri1"], row["Product Family"] ,row["Colour Status"][1] , row["Buffer"]), axis = 1)

    for i in range(0,7):
        if (mk_arr[i] != mkact_arr[i]):
            xs_arr[i] += (mk_arr[i] - mkact_arr[i])
    print(mkact_arr)
    print(xs_arr)
    sns_max = xs_arr[0]
    slb_max = xs_arr[1]
    sld_max = xs_arr[2]
    x2b_max = xs_arr[3]
    x2d_max = xs_arr[4]
    x2p_max = xs_arr[5]
    plt_max = xs_arr[6]
    max_arr_t = [sns_max,slb_max,sld_max,x2b_max,x2d_max,x2p_max,plt_max]
    for i in range(0,7):
        max_arr[i] = max_arr_t[i]
    for x in df_make.index:
        if x in bpr_reg_norm.index:
            c_row = bpr_reg_norm.loc[x, bpr_reg_norm.columns]
            count_row = (df_make.loc[x, df_make.columns])
            c_row['QTY'] = count_row['QTY']
            bpr_reg_norm.loc[x, bpr_reg_norm.columns] = c_row
            if ((c_row != bpr_reg_norm.loc[x, bpr_reg_norm.columns]).all()):
                print("error")

    bpr_reg_norm["QTY"] = bpr_reg_norm.apply(lambda row : qty_p2(row["QTY"],row["Colour Status"][1] , row["Buffer"],row["Product Family"] ), axis = 1)
    range_mask = (np.arange(0,len(bpr_reg_norm)))
    sns_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] == 'Slide & Store').reset_index(drop = True))]
    slb_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] == 'SL Body').reset_index(drop = True))]
    sld_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] ==  'SL Door').reset_index(drop = True))]
    x2b_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] == 'X2 Body').reset_index(drop = True))]
    x2d_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] == 'X2 Door').reset_index(drop = True))]
    x2p_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] == 'X2 Precoated').reset_index(drop = True))]
    plt_mask = (np.arange(0,len(bpr_reg_norm)))[np.array((bpr_reg_norm["Product Family"] == 'Platina').reset_index(drop = True))]
    sku_tp = np.array((bpr_reg_norm.iloc[:, 20]).reset_index(drop = True))
    sku_bc = np.array((bpr_reg_norm.iloc[:, 21]).reset_index(drop = True))
    sku_names = np.array(bpr_reg_norm.index)
    sku_colour = np.array((bpr_reg_norm.iloc[:, 12]).reset_index(drop = True))
    n = len(sku_colour)
    sku_cost = [0]*n
    for i in range(0,n):
        col = sku_colour[i]
        if (col == "Black"):
            sku_cost[i] = 15 + random.uniform(0,0.9) + sku_bc[i]
        elif (col == "Red"):
            sku_cost[i] = 10 + 2*(sku_tp[i]) + random.uniform(0,0.001)
        elif (col == "Yellow"):
            sku_cost[i] = 6 + 2*(sku_tp[i]) + random.uniform(0,0.001)
        elif (col == "Green") :
            sku_cost[i] = 3 + 2*(sku_tp[i]) + random.uniform(0,0.001)
        else:
            sku_cost[i] = 2*(sku_tp[i]) + random.uniform(0,0.9)

    sku_max = np.array((bpr_reg_norm.iloc[:, 15] - bpr_reg_norm.iloc[:, 22]).reset_index(drop = True))

    problem = pulp.LpProblem('Production_Scheduler', pulp.LpMaximize)
    p = [pulp.LpVariable(sku_names[i], lowBound = 0, upBound = sku_max[i], cat = "Integer") for i in range(0,len(sku_names)) ]
    f= pulp.LpAffineExpression( [(p[i],sku_cost[i]) for i in range(0,len(sku_names))])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in sns_mask]) ,sense = -1 , rhs = max_arr[0])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in slb_mask]) ,sense = -1 , rhs = max_arr[1])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in sld_mask]) ,sense = -1 , rhs = max_arr[2])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in x2b_mask]) ,sense = -1 , rhs = max_arr[3])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in x2d_mask]) ,sense = -1 , rhs = max_arr[4])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in x2p_mask]) ,sense = -1 , rhs = max_arr[5])
    problem += pulp.LpConstraint(e = pulp.LpAffineExpression([ (p[i],1) for i in plt_mask]) ,sense = -1 , rhs = max_arr[6])
    problem += f

    status = problem.solve()

    for var in problem.variables():
        if(var.value() != 0):
            #print(f"{var.name}: {var.value()}")
            c_row = bpr_reg_norm.loc[var.name, bpr_reg_norm.columns]
            c_row['QTY'] += var.value()
            bpr_reg_norm.loc[var.name, bpr_reg_norm.columns] = c_row

    #bpr_reg_norm.to_excel('quantity.xlsx')

    out_df = bpr_reg_norm[bpr_reg_norm["QTY"] > 0]

    out_df = out_df[["Item Desc","Product Family","QTY"]]

    out_df.reset_index(inplace = True)

    out_df["Date"] = date

    out_df = out_df[["Date","Item Code","Item Desc", "Product Family","QTY"]]
    out_df["Product Family"] = out_df["Product Family"].apply(lambda x: family_line_dict.get(x))
    out_df["PRODUCTION NO"] = " "
    out_df["COLOUR"] = out_df["Item Code"].apply(lambda x: item_col_dict.get(x))
    out_df = out_df[out_df["QTY"] >= 15]
    out_df.reset_index(inplace = True)
    out_df = out_df[["Date","PRODUCTION NO","Item Code","Item Desc","COLOUR","QTY","Product Family"]]
    out_df["Product Family"] = out_df["Product Family"].map({2 : 3, '2' : 3, 1 : 1, '1' : 1})
    #(out_df[out_df["Product Family"] == 2])["Product Family"] = 3
    out_df.columns = ["DATE","PRODUCTION NO","ITEMCODE","DESCRIPTION","COLOUR" ,"QTY", "PRIORITY"]
    return out_df
