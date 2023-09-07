from PyQt5 import QtCore, QtGui, QtWidgets
import json
import re
import pandas as pd
import numpy as np
import itertools

### PYQT IMPORTS
from PyQt5 import QtCore, QtGui, QtWidgets, QtSql
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QRegExpValidator



from PyQt5.QtCore import QRegExp

### Modules
from Status_Dialog import * 
from RSE_BalancedHalfSample import *


from PyQt5.QtWidgets import QMessageBox
from RSE_TOOL import *


### Balanced Half Sample Method
### ----------------------------

def balancedHalfSample(data, df):
    final_parsed_query = []

    for i in list(TRANSFORM_(data))[2]:
        # print(i)
        if i:
            parsed_query = parse_query_string(i)
            final_parsed_query.append(parsed_query)
        else:
            final_parsed_query.append(i)

    ### Calling the final_parsed_query function
    ### ---------------------------------------
    print(final_parsed_query)

    ###
    Arr1 = list(TRANSFORM_(data))[0]
    Arr2 = list(TRANSFORM_(data))[1]

    # print(Arr2)
    Arr2 = [i.strip('()\'\"') for i in Arr2]
    Arr3 = final_parsed_query

    Arr2 = [float(elem) if elem.replace('.', '', 1).isdigit() else elem for elem in Arr2]
    print(Arr2)

    ###
    # Check for name mismatches and create new columns
    mismatched_names = set(Arr1) - set(df.columns)

    new_columns = {name: [0] * df.shape[0] for name in mismatched_names}
    df = df.assign(**new_columns)

    Arr3 = [i.replace("(['", "([").replace("'])", "])") if i is not None else 'True' if Arr3[j] == 'Otherwise' else True for j, i in enumerate(Arr3)]

    ###
    new_arr3 = []
    for i, value in enumerate(Arr3):
        if value == 'Otherwise':
            other_values = set(['~(' + Arr3[j] + ')' for j in range(len(Arr3)) if Arr1[j] == Arr1[i] and Arr3[j] != 'Otherwise'])
            new_value = ' & '.join(other_values)
            print(new_value)
            new_arr3.append(new_value)
        else:
            new_arr3.append(value)
            print(value)

    Arr3 = new_arr3

    for i in range(len(Arr2)):
        if type(Arr3[i]) == str:
            print(Arr3[i])

    ###
    ### Apply the queries in Arr3 to filter the rows of the dataframe
    for i, query_str in enumerate(Arr3):
        
        if query_str is True:
            if type(Arr2[i]) == str:
                if '+' in Arr2[i]:
                    df[Arr1[i]] = df[Arr2[i].split('+')].apply(lambda x: x.sum(), axis=1)
                    print('*** First if', Arr2[i], Arr1[i], Arr3[i], i)

                else:
                    df[Arr1[i]] = df[[Arr2[i]]].apply(lambda x: x, axis=1)
                    print('*** First else', Arr2[i], Arr1[i], Arr3[i], i)
            # if type(Arr3[i]):
            #     print('^^^ Second if', Arr2[i], Arr1[i], Arr3[i], i)
            #     df.loc[df_query.index, Arr1[i]] = Arr2[i]
        
        else:
            # try:
            # query_str = re.sub(r'0?\s?(\d+)', r'\1', query_str)
            print(query_str)
            df_query = df.query(query_str)
            # except:
            #     pass

            if type(Arr2[i]) == float:
                df.loc[df_query.index, Arr1[i]] = Arr2[i]
                # print('### Third if', Arr2[i], Arr1[i], Arr3[i])
            else:
                df.loc[df_query.index, Arr1[i]] = Arr2[i]
                # print('@@@ First else', Arr2[i], Arr1[i], Arr3[i])


            if type(Arr2[i]) == str:
                if Arr2[i] in list(df.columns):
                    df[Arr1[i]] = df[[Arr2[i]]].apply(lambda x: x, axis=1)
                    # print('$$$ Fourth if', Arr2[i], Arr1[i], Arr3[i])

    ###
    df.to_csv('WorkFile_Balanced_Half_Sample_Method_Example1.csv')


    ''' ****************** STEP -1  *********************** '''
    ''' *************************************************** '''
    ### Read the data from the CSV file
    ### Get the current column names
    old_col_names = df.columns.tolist()

    # Create a dictionary with the new column names
    new_col_names = {col: col.strip() for col in old_col_names}

    # Rename the columns using the dictionary
    df.rename(columns=new_col_names, inplace=True)

    df['MULT'] = pd.to_numeric(df['MULT'])

    # Calculate the SS_MULT and CMULT fields
    df['SS_MULT'] = df['MULT'] / 100
    df['CMULT'] = df.apply(lambda row: row['MULT'] / 200 if row['NSC'] > row['NSS'] else row['MULT'] / 100, axis=1)

    lst_df = list(df.columns)

    groupby_cols = GROUP_(data, df)

    # Create a list of all possible subsets of groupby_cols
    groupby_cols_subsets = [list(subset) for i in range(len(groupby_cols) + 1)
                            for subset in itertools.combinations(groupby_cols, i)]

    # Select columns that are not in groupby_cols
    cols = [col for col in df.columns if col in groupby_cols]

    agg_dict = {col: sum for col in cols if (df[col].dtype == int or df[col].dtype == float) and col in groupby_cols}



    ################
    Arr_W1 = []


    ### Loop over each subset of columns and perform the aggregation
    ### ------------------------------------------------------------
    for i in range(len(groupby_cols_subsets)):

        print(groupby_cols_subsets[i])

        # Define the prefix to use for the new columns
        prefix = 'Z'

        ## For All Combinations
        if i == 0:

            # # If empty list is selected, perform aggregation without grouping
            # w1 = df.agg(agg_dict).to_frame().T

            # print(new_df)

            test_lst = groupby_cols

            df['POP'] = pd.to_numeric(df['POP']) * df['CMULT']
            df['LF'] = pd.to_numeric(df['LF']) * df['CMULT']
            df['WRK'] = pd.to_numeric(df['WRK']) * df['CMULT']
            df['no_sam'] = df.shape[0]

            agg_dict = {'POP': 'sum', 'LF': 'sum', 'WRK': 'sum', 'no_sam': 'size'}

            # Group the data by SEC and ST_GR and calculate the aggregates
            w1 = df.agg(agg_dict).to_frame().T

            w1 = w1.rename(columns={'POP': 'pophat', 'LF': 'lfhat', 'WRK': 'wrkhat' }).reset_index()

            w1['groupby_cols'] = 'All columns'

            new_col_val_ = []

            for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w1.columns)))):
                # Create a new column with the appropriate number of "Z"s and the prefix
                new_col_val_.append(prefix * (i+1))

            for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w1.columns)))):
                # Add the new column to the DataFrame
                w1[test_lst[i]] = new_col_val_[i]

            ################
            Arr_W1.append(w1)
            
        else:
            # Create a new dataframe by grouping by the selected columns and aggregating
            # Note: here I'm just calculating the sum of columns C and D, but you can use any aggregation function
            # new_df = df.groupby(list(groupby_cols_subset)).agg(agg_dict).reset_index()

            w11 = df.groupby(groupby_cols_subsets[i]).agg(
                                                            pophat=('POP', 'sum'),
                                                                lfhat=('LF', 'sum'),
                                                                    wrkhat=('WRK', 'sum'),
                                                                        no_sam= ('POP', 'size')
                                                            ).reset_index()

            print(w11)

            # Rename the columns and reset the index
            # w1 = w1.rename(columns={'POP': 'pophat', 'LF': 'lfhat', 'WRK': 'wrkhat'}).reset_index()

            test_lst = groupby_cols_subsets[i]

            # Add a new column to indicate which groupby columns are selected
            w11['groupby_cols'] = ', '.join(groupby_cols_subsets[i])

            new_col_val = []
            for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w11.columns)))):
                # Create a new column with the appropriate number of "Z"s and the prefix
                new_col_val.append(prefix * (i+1))

            for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w11.columns)))):
                # Add the new column to the DataFrame
                w11[list(set(groupby_cols_subsets[-1]) - set(w11.columns))[i]] = new_col_val[i]


            Arr_W1.append(w11)
            
            print(Arr_W1)

    '''*********************** STEP - 2*************************'''
    '''*********************************************************'''
    ### Filter dataframe to include only rows where SS is 1 or 2
    df = df.query("SS in [1, 2]")
    # print(f"Number of rows after filtering: {len(df)}")

    Arr_W2 = []

    ### Loop over subset of columns to perform aggregation
    for i in range(len(groupby_cols_subsets)):

        new_groupby_cols_subset = ['STRMID']

        # Define the prefix to use for the new columns
        prefix = 'Z'

        if i == 0:

            df['popstr'] = pd.to_numeric(df['POP']) * df['CMULT']
            df['s1pop'] = np.where(df['SS'] == 1, pd.to_numeric(df['POP']) * df['SS_MULT'], 0)
            df['s2pop'] = np.where(df['SS'] == 2, pd.to_numeric(df['POP']) * df['SS_MULT'], 0)
            df['lfstr'] = pd.to_numeric(df['LF']) * df['CMULT']
            df['s1lf'] = np.where(df['SS'] == 1, pd.to_numeric(df['LF']) * df['SS_MULT'], 0)
            df['s2lf'] = np.where(df['SS'] == 2, pd.to_numeric(df['LF']) * df['SS_MULT'], 0)
            df['wrkstr'] = pd.to_numeric(df['WRK']) * df['CMULT']
            df['s1wrk'] = np.where(df['SS'] == 1, pd.to_numeric(df['WRK']) * df['SS_MULT'], 0)
            df['s2wrk'] = np.where(df['SS'] == 2, pd.to_numeric(df['WRK']) * df['SS_MULT'], 0)

            agg_dict = {'popstr': 'sum', 's1pop': 'sum', 's2pop': 'sum', 'lfstr': 'sum', 's1lf': 'sum', \
                        's2lf': 'sum', 'wrkstr': 'sum', 's1wrk': 'sum', 's2wrk': 'sum'}

            # # If empty list is selected, perform aggregation without grouping
            w2 = df.agg(agg_dict).to_frame().T

            w2['STRMID'] = 'ALL'

            new_col_val_ = []

            for i in range(len(list(set(groupby_cols) - set(w2.columns)))):
                # Create a new column with the appropriate number of "Z"s and the prefix
                new_col_val_.append(prefix * (i+1))

            for i in range(len(list(set(groupby_cols) - set(w2.columns)))):
                # Add the new column to the DataFrame
                w2[groupby_cols[i]] = new_col_val_[i]

            Arr_W2.append(w2)

            continue

        print(groupby_cols_subsets[i])

        new_groupby_cols_subset.extend(groupby_cols_subsets[i])

        print(new_groupby_cols_subset)

        w2 = df.groupby(groupby_cols_subsets[i]).agg(
                popstr=('POP', lambda x: np.sum(x * df['CMULT'])),
                s1pop=('POP', lambda x: np.nansum(np.where(df['SS'] == 1, x * df['SS_MULT'], 0))),
                    s2pop=('POP', lambda x: np.nansum(np.where(df['SS'] == 2, x * df['SS_MULT'], 0))),
                    lfstr=('LF', lambda x: np.nansum(x * df['CMULT'])),
                        s1lf=('LF', lambda x: np.nansum(np.where(df['SS'] == 1, x * df['SS_MULT'], 0))),
                        s2lf=('LF', lambda x: np.nansum(np.where(df['SS'] == 2, x * df['SS_MULT'], 0))),
                            wrkstr=('WRK', lambda x: np.nansum(x * df['CMULT'])),
                            s1wrk=('WRK', lambda x: np.nansum(np.where(df['SS'] == 1, x * df['SS_MULT'], 0))),
                            s2wrk=('WRK', lambda x: np.nansum(np.where(df['SS'] == 2, x * df['SS_MULT'], 0)))
                                ).reset_index()

        new_col_val = []
        for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w2.columns)))):
            # Create a new column with the appropriate number of "Z"s and the prefix
            new_col_val.append(prefix * (i+1))

        for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w2.columns)))):
            # Add the new column to the DataFrame
            w2[list(set(groupby_cols_subsets[-1]) - set(w2.columns))[i]] = new_col_val[i]

        Arr_W2.append(w2)      


    '''************************** STEP - 3 *****************************'''
    '''*****************************************************************'''
    Arr_W3 = []

    # Define the prefix to use for the new columns
    prefix = 'Z'

    ### Loop over subset of columns to perform aggregation
    for i in range(len(Arr_W2)):

        # Define the prefix to use for the new columns
        prefix = 'Z'

        # Join w2 and w1 on SEC and ST_GR columns
        # w3 = Arr_W2[i].merge(Arr_W1[i], on=['SEC', 'AGE_GR'])
        w3 = pd.concat([Arr_W2[i], Arr_W1[i]], axis=1)

        # if i==0:
        #     w3['no_sam'] = df.shape[0]
        # elif i>0:
        #     w3['no_sam'] = df.shape[0]
        
        # Add new columns to w3
        w3['r1hat'] = w3['lfhat'] / w3['pophat']
        w3['r2hat'] = w3['wrkhat'] / w3['pophat']

        # Equate s1pop and s2pop
        w3.loc[w3['popstr'] == w3['s1pop'], 's1pop'] = w3.loc[w3['popstr'] == w3['s1pop'], 's2pop']
        w3.loc[w3['popstr'] == w3['s2pop'], 's2pop'] = w3.loc[w3['popstr'] == w3['s2pop'], 's1pop']

        # Equate s1lf and s2lf
        w3.loc[w3['lfstr'] == w3['s1lf'], 's1lf'] = w3.loc[w3['lfstr'] == w3['s1lf'], 's2lf']
        w3.loc[w3['lfstr'] == w3['s2lf'], 's2lf'] = w3.loc[w3['lfstr'] == w3['s2lf'], 's1lf']

        # Equate s1wrk and s2wrk
        w3.loc[w3['wrkstr'] == w3['s1wrk'], 's1wrk'] = w3.loc[w3['wrkstr'] == w3['s1wrk'], 's2wrk']
        w3.loc[w3['wrkstr'] == w3['s2wrk'], 's2wrk'] = w3.loc[w3['wrkstr'] == w3['s2wrk'], 's1wrk']
        
        new_col_val = []
        for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w3.columns)))):
            # Create a new column with the appropriate number of "Z"s and the prefix
            new_col_val.append(prefix * (i+1))

        for i in range(len(list(set(groupby_cols_subsets[-1]) - set(w3.columns)))):
            # Add the new column to the DataFrame
            w3[list(set(groupby_cols_subsets[-1]) - set(w3.columns))[i]] = new_col_val[i]

        Arr_W3.append(w3)


    ### Drop duplicate columns
    for i in range(len(Arr_W3)):
        Arr_W3[i] = Arr_W3[i].loc[:, ~Arr_W3[i].columns.duplicated()]
        print(Arr_W3[i].columns)


    '''**************************** STEP - 4 *********************************** '''
    '''**************************************************************************'''
    Arr_W4 = []

    ### Loop over subset of columns to perform aggregation
    for i in range(len(Arr_W3)):
        mask = Arr_W3[i].columns.duplicated()
        Arr_W3[i] = Arr_W3[i].loc[:, ~mask]

        gby_list = ['no_sam', 'pophat', 'lfhat', 'wrkhat', 'r1hat', 'r2hat'] + groupby_cols

        print(gby_list)

        w4 = Arr_W3[i].groupby(gby_list).agg(

            var_pop = ('s1pop', lambda x: np.sum((x - Arr_W3[i]['s2pop']) ** 2)),

                mse_R1 = ('s1lf', lambda x: np.nansum(
                            ((x - Arr_W3[i]['s2lf']) ** 2) +
                                ((Arr_W3[i]['r1hat'] ** 2) * ((Arr_W3[i]['s1pop'] - Arr_W3[i]['s2pop']) ** 2) -
                                    (2 * Arr_W3[i]['r1hat'] * (x - Arr_W3[i]['s2lf']) * (Arr_W3[i]['s1pop'] - Arr_W3[i]['s2pop'])))
                        )),

                    mse_R2 = ('s1wrk', lambda x: np.nansum(
                                ((x - Arr_W3[i]['s2wrk']) ** 2) +
                                    ((Arr_W3[i]['r2hat'] ** 2) * ((Arr_W3[i]['s1pop'] - Arr_W3[i]['s2pop']) ** 2) -
                                        (2 * Arr_W3[i]['r2hat'] * (x - Arr_W3[i]['s2wrk']) * (Arr_W3[i]['s1pop'] - Arr_W3[i]['s2pop'])))
                            ),
            
        )).reset_index()

        Arr_W4.append(w4)

        print(Arr_W4[i])


    '''*************************** STEP - 5 *****************************'''
    '''******************************************************************'''
    Arr_W5 = []
    for i in range(len(Arr_W4)):
        # Calculate RSEs for each variable in T1
        w5 = Arr_W4[i].assign(
            RSE_POP=lambda x: 100 * (x['var_pop'] ** 0.5) / (2 * x['pophat']),
                RSE_R1=lambda x: 100 * (x['mse_R1'] ** 0.5) / (2 * x['lfhat']),
                    RSE_R2=lambda x: 100 * (x['mse_R2'] ** 0.5) / (2 * x['wrkhat'])
                        )

        Arr_W5.append(w5)

        print(Arr_W5)

    
    '''******************************* STEP - 6 *********************************'''
    '''**************************************************************************'''
    Tables = []

    for i in range(len(Arr_W5)):
        gby_list = ['no_sam', 'pophat', 'lfhat', 'wrkhat', 'r1hat', 'r2hat', 'RSE_POP', 'RSE_R1', 'RSE_R2'] + groupby_cols
        
        # assuming the data is loaded into a DataFrame called w5
        T1 = Arr_W5[i][gby_list].copy()

        # add empty columns for sec_desc and st_gr_desc
        T1['sec_desc'] = ''
        T1['st_gr_desc'] = ''

        # apply formatting to columns
        T1['R1'] = 100 * T1['r1hat']
        T1['R2'] = 100 * T1['r2hat']

        gby_list = ['sec_desc', 'st_gr_desc', 'no_sam', 'pophat','lfhat', 'wrkhat', 'RSE_POP', 'R1', 'RSE_R1', 'R2', 'RSE_R2'] + groupby_cols

        # reorder columns
        T1 = T1[gby_list]

        gby_list = ['no_sam', 'pophat','lfhat', 'wrkhat', 'r1hat', 'r2hat', 'RSE_POP', 'RSE_R1', 'RSE_R2'] + groupby_cols

        T1 = Arr_W5[i][gby_list].copy()

        T1['sec_desc'] = ' '
        T1['st_gr_desc'] = ' '

        T1['R1'] = 100 * T1['r1hat']
        T1['R2'] = 100 * T1['r2hat']

        gby_list = ['no_sam', 'pophat','lfhat', 'wrkhat', 'RSE_POP', 'R1', 'RSE_R1', 'R2', 'RSE_R2'] + groupby_cols

        T1 = T1[gby_list]

        Tables.append(T1)
        
    print(Tables)
    print('\n')


    '''***************************** STEP - 7 **********************************'''
    '''*************************************************************************'''
    def merge_dataframes(tables):
        merged_df = tables[0]
        for df in tables[1:]:
            merged_df = pd.concat([merged_df, df], axis=0)
        return merged_df

    df_Tables = merge_dataframes(Tables)

    try:
        settings = QtCore.QSettings()
        path = settings.value("Paths/csvfile")
        
        ### Opening file dialog to export multiplier file
        filename_design_template, _  = QtWidgets.QFileDialog.getSaveFileName(None, path, 'Relative Standard Error', filter='*.xls;;*.xlsx;;*.csv')

        if filename_design_template:
            finfo = QtCore.QFileInfo(filename_design_template)
            settings.setValue("Paths/csvfile", finfo.absoluteDir().absolutePath())

            ###! File selection of xls, xlsx or csv  
            if filename_design_template.endswith(".csv"):
                df_Tables.to_csv(filename_design_template)

            if filename_design_template.endswith(".xls"):
                df_Tables.to_excel(filename_design_template)

            if filename_design_template.endswith(".xlsx"):
                df_Tables.to_excel(filename_design_template)

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("RSE file successfully exported.")
            msgBox.setWindowTitle("Success")
            msgBox.setStyleSheet(
                "QMessageBox { width:800px; height:800px; } QPushButton{ width:140px; font-size: 20px; }"
            )
            msgBox.setStyleSheet("background-color: #A5D6A7;")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()

            ### To verify the return value (yes/no) and close the tool on clicking 'Yes'
            if returnValue == QMessageBox.Ok:
                pass

    except:
        pass



### SAMPLING
############
def SAMPLING_(data):
    for key, value in data.items():
        # To read the sampling method used
        if str('SAMPLING').lower() in str(key).lower():
            
            if str('SUBSAMPLE2').lower() in str(key).lower():
                print('SUBSAMPLE2')
                return str('SUBSAMPLE2')
            elif (str('PPSWR-SSS').lower() or str('PPSWR').lower()) in str(key).lower():
                return str('PPSWR-SSS')
            elif (str('SRSWR-SSS').lower() or str('SRSWR').lower()) in str(key).lower():
                return str('SRSWR-SSS')
            
### ---------------------------------------------------------------------------------------------------------
def GROUP_(data, df):
    for key, value in data.items():    
        if str('GROUP').lower() in str(key).lower():
            value = [s.split()[0] for s in value]
            for name in value:
                if name in df.columns:
                    pass
                elif name not in df.columns:
                    df[name] = 0

            return value

    for key, value in data.items():
        print(key, value)
        ### To read the group columns
        if str('GROUP').lower() in str(key).lower():
            # Define regular expression to match three words separated by whitespace
            regex = r'\w+\s+\w+\s+\w+'

            for i in range(len(value)):
                # Find the first match in the string
                match = re.search(regex, value[i])

                # Print the match
                lst = match.group().split()

                print(lst[0], lst[1])

                if lst[0] in df.columns:
                    if lst[1] == 'C':
                        df[lst[0]] = df[lst[0]].astype(str)
                elif lst[0] not in df.columns:
                    # if lst[1] == 'C':
                    #     df[lst[0]] = ''
                    pass

        else:
            pass

### ---------------------------------------------------------------------------------------------------------
def FILTER_(data, df):
    lst_queries = []
    for key, value in data.items():

        if str('FILTER').lower() in str(key).lower():
            # Define regular expression to match three words separated by whitespace
            regex = r'\([^()]+\)|\w+[<>!=]+[0-9]+(?:\s*(?:&&|\|\|)\s*(?:\([^()]+\)|\w+[<>!=]+[0-9]+))*'

            for i in range(len(value)):
                # Find the first match in the string
                match = re.search(regex, value[i])

                # Print the match
                lst_queries.append(match.group())

        else:
            pass

    filter_query = ' & '.join(['({})'.format(item) for item in lst_queries])
   
    try:
        df = df.query(filter_query)
        return df
    except:
        return df


### ---------------------------------------------------------------------------------------------------------
def FILE_(data):
    for key, value in data.items():
        ### To read the file used
        if str('FILE').lower() in str(key).lower():
            if key.strip().endswith('.xlsx') or key.strip().endswith('.csv'):
                match = re.search(r'[A-Za-z0-9]+\.xlsx|[A-Za-z0-9]+\.csv', key)   
                if match:
                    file_name = match.group(0)
                    return file_name
                else:
                    file_name = None
                    return file_name
                print(file_name)

            # return file_name

### ---------------------------------------------------------------------------------------------------------
def RENAME_(data, df):
    # create an empty dictionary to store the column names
    column_dict = {}

    for key, value in data.items():
        ### To read the file used
        if str('RENAME').lower() in str(key).lower():
            # value = [elem for elem in value[0].split(' ') if elem.strip()]
            # return value

            for col in value:
                col_split = col.split()

                # print(col_split[0])

                # extract the last element as the column name
                col_name = col_split[-1]

                new_col_name = col_split[0]

                ### if there is a "+" sign in the last element, concatenate the columns
                if "+" in col_name:
                    col_concat = col_name.split("+")
                    col_concat = [i.upper() for i in col_concat]

                    ### concatenate the columns
                    # df[new_col_name] = df.loc[:, col_concat].apply(lambda x: ''.join(['0' + str(i) if int(i) < 10 else str(i) for i in x]), axis=1)
                    if isinstance(df, pd.DataFrame):
                        # Perform DataFrame operations on 'df' here
                        df[new_col_name] = df.loc[:, col_concat].apply(lambda x: ''.join(['0' + str(i) if int(i) < 10 else str(i) for i in x]), axis=1)
                    else:
                        # Handle the case when 'df' is not a DataFrame
                        # Print an error message or perform any necessary actions
                        print("Error: 'df' is not a DataFrame")

                else:
                    col_name = col_name.upper()

### ---------------------------------------------------------------------------------------------------------
def NEWVARIABLE_(data, df):
    for key, value in dict(data).items():
        ### To read the file used
        if str('VARIABLE').lower() in str(key).lower():
            value = [s.split()[0] for s in value]
            
            for name in value:
                df[name] = 1

### ---------------------------------------------------------------------------------------------------------
def TRANSFORM_(data):
    global condition

    # Initialize empty arrays to store the values
    Arr1 = []
    Arr2 = []
    Arr3 = []

    for key, value in data.items():
        ### To read the file used
        if str('TRANSFORM').lower() in str(key).lower():
            lines = '\n'.join(value)
            lines = lines.split("\n")

            # Loop through the lines and extract the values
            for line in lines:
                line_values = line.split('=')  # split the line into values
                print(line_values)
                var_name = line_values[0].strip()  # extract the variable name
                
                var_value = line_values[1].strip().split()[0]  # extract the variable value

                            
                # Regular expression pattern to extract the condition
                pattern = r'^\w+=\([^\)]*\)(?:\s+(.*?)(?:\n|$))?$'

                # Split the string into lines and extract the condition from each line
                conditions = []

                for line in line.split('\n'):
                    match = re.match(pattern, line)
                 
                    if match:
                        condition = match.group(1)
                        
                        if condition is not None:
                            conditions.append(condition.strip())
                        else:
                            conditions.append(None)

                # Append the values to the arrays
                Arr1.append(var_name)
                Arr2.append(var_value)

                Arr3.append(condition)

    return Arr1, Arr2, Arr3
        
### ---------------------------------------------------------------------------------------------------------
def EST_RSE_(data):
    for key, value in data.items():
        pass

###
def parse_query_string(query_string):
    # Replace curly quotes with straight quotes
    query_string = query_string.replace("”", "\"")
    
    if 'in' in query_string:
        if ('NOT in' or 'not in') in query_string:
            query_string = re.sub(r'(\b\w+\b)\s+NOT in\s*\(([^)]+)\)', r'~\1.isin([\2])', query_string)
            query_string = re.sub(r'(\b\w+\b)\s+in\s*\(([^)]+)\)', r'\1.isin([\2])', query_string)
        elif 'in' in query_string:
            # Replace 'in' operators with 'isin' and add list brackets
            query_string = re.sub(r'(\b\w+\b)\s+in\s*\(([^)]+)\)', r'\1.isin([\2])', query_string)
        else:
            pass
            
    else:
        query_string = re.sub(r'''(\b\w+\b)(\s*[<>=!]+\s*)('[^']*'|"[^"]*")|(\b\w+\b)(\s*[<>=!]+\s*)("[^]*'|"[^"]*")''', r'\1\2\3', query_string)
        query_string = re.sub(r'''(\b\w+\b)(\s*[<>=!]+\s*)(\d+)''', r'\1\2\3', query_string)

    # Replace double quotes with single quotes
    query_string = query_string.replace("\"", "'")
    
    # Replace 'AND' with '&', 'OR' with '|', and add parentheses
    query_string = query_string.replace(" AND ", " & ")\
                        .replace(" OR ", " | ").\
                            replace("&&", '&').\
                                replace('||', '|').\
                                    replace('<>', '!=').\
                                        replace('NOT', '~')
    
    # Replace column names with df[column] syntax
    query_string = re.sub(r'(\b\w+\b)=', r'\1==', query_string)


    # # Define a function to replace variable names with df["<variable_name>"]
    # def replace_var(match):
    #     return f'df["{match.group(0)}"]'

    # # Use regex to replace variable names with df["<variable_name>"]
    # pattern = re.compile(r'''\b\w+\b(?=\s*[<>=!]=*\s*[\'"\d])''')
    # new_string = pattern.sub(replace_var, query_string)

    query_string = re.sub(r"'(\d+)'|\"(\d+)\"", lambda match: str(int(match.group(1) or match.group(2))), query_string)
    query_string = re.sub(r'\[(\d+,\s*)+\d+\]', lambda match: '[' + match.group(0).replace("\'", "'") + ']', query_string)
    
    return query_string



###! Main UI Class
#### -------------
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1186, 620)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        # Set the background gradient using a stylesheet
        style_sheet = "QWidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00467f, stop:1 #4286f4); }"
        
        self.centralwidget.setStyleSheet(style_sheet)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("MoSPI_1.png"))
        # C:\Users\gaurav\Desktop\a 'RSE TOOL'\RSE_Deliverable\MoSPI_1.png
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 3, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(20)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);\n"
"")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 3)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 0, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)

        ###
        self.pushButton_3.clicked.connect(self.generateRSE)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: rgb(236, 250, 244);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_2.addWidget(self.pushButton_3, 3, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 2, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)

        ###
        self.pushButton.clicked.connect(self.onClickUploadRSEFileBtn)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(236, 250, 244);")
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 1, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem2, 4, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)

        ###
        self.pushButton_2.clicked.connect(self.onClickUploadDataFileBtn)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(236, 250, 244);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_2.addWidget(self.pushButton_2, 2, 1, 1, 1)

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("background-color: rgb(220, 20, 60);\n"
"color:rgb(229, 232, 232);")
        self.pushButton_4.setObjectName("pushButton_4")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("background-color: rgb(46, 204, 113);\n"
"color:rgb(229, 232, 232);")
        self.gridLayout_2.addWidget(self.pushButton_5, 0, 2, 1, 1)

        self.gridLayout_2.addWidget(self.pushButton_4, 4, 2, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 2)
        self.gridLayout_2.setColumnStretch(1, 5)
        self.gridLayout_2.setColumnStretch(2, 2)
        self.gridLayout_2.setRowStretch(0, 3)
        self.gridLayout_2.setRowStretch(1, 5)
        self.gridLayout_2.setRowStretch(2, 5)
        self.gridLayout_2.setRowStretch(3, 5)
        self.gridLayout_2.setRowStretch(4, 2)
        self.gridLayout.addLayout(self.gridLayout_2, 1, 0, 1, 5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 653, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "RSE for NSS Two- Stage Samples"))
        self.pushButton_3.setText(_translate("MainWindow", "Generate Relative Standard Error"))
        self.pushButton.setText(_translate("MainWindow", "Upload RSE File"))
        self.pushButton_2.setText(_translate("MainWindow", "Upload Data File"))
        self.pushButton_4.setText(_translate("MainWindow", "Exit"))
        self.pushButton_5.setText(_translate("MainWindow", "User Manual"))

    ###
    def onClickUploadDataFileBtn(self):
        global df

        try:
            ### setting the path 
            settings = QtCore.QSettings()
            path = settings.value("Paths/csvfile")
            design_temp = QtWidgets.QFileDialog.getOpenFileName(
                    None, "Upload file", path, "Select csv or xlsx or xls file (*.csv *.xlsx *.xls)"
                )[0]
            self.df = pd.read_csv(design_temp)
            # remove blank spaces from front and back of column names
            self.df.columns = self.df.columns.str.strip()
            # return df

        except:
            pass


    ###
    def onClickUploadRSEFileBtn(self):
        global data

        try:
            ### setting the path 
            settings = QtCore.QSettings()
            path = settings.value("Paths/csvfile")
            fname = QtWidgets.QFileDialog.getOpenFileName(
                    None, "Upload RSE file", path, "Select RSE file (*.RSE)"
                )[0]
            
            with open(fname, "r") as f:
                ## Data is the dictionary
                data = {}

                ## Keys
                key = ''

                ## values
                values = []

                # print(f.readlines())

                # for line in f:
                #     line = line.strip()
                #     print(line)

                for line in f:
                    print(line)
                    line = line.strip()
                    if line.startswith('#'):
                        if key:
                            key = key.split("'")[0]
                            key = key.split("\t")[0]
                            data[key] = [v for v in values if v and not v.startswith("'")]
                        key = line.split("'")[0]
                        key = key.split("\t")[0]
                        values = []
                    elif line.startswith("'"):
                        pass
                    else:
                        line = line.split("'")[0]
                        values.append(line.strip())

            # add last group
            if key:
                data[key] = [v for v in values if v and not v.startswith("'")]

            self.data = data
            # return data
        except:
            pass


    ###
    def generateRSE(self):
        if data:
            ### Calling all the functions
            SAMPLING_(self.data)
            GROUP_(self.data, self.df)
            self.df = FILTER_(self.data, self.df)
            TRANSFORM_(self.data)

            RENAME_(self.data, self.df)
            NEWVARIABLE_(self.data, self.df)
            FILE_(self.data)
            print('kaam kar raha hai ...')
            balancedHalfSample(self.data, self.df)

            

                



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
