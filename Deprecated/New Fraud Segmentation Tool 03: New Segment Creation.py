# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fraud Segmentation Tool Historical Rule Performance Demo
# MAGIC ## Date: 9/5/2024
# MAGIC ## Author: Joby George
# MAGIC
# MAGIC # Background
# MAGIC
# MAGIC This script goes over the last step in the Fraud Segmentation Tool, identifying high P2D0 or NTL segments using Decision Trees. 
# MAGIC
# MAGIC Previous scripts have completed the feature engineering, which creates the dataset used in this script.
# MAGIC
# MAGIC The main steps in this script are:
# MAGIC     
# MAGIC     A. Pull feature data from previously created feature table
# MAGIC     
# MAGIC     B. Prepare dataset for decision tree
# MAGIC
# MAGIC     C. Train and Tune Decision Tree
# MAGIC
# MAGIC     D. Identify High Performing Rules
# MAGIC
# MAGIC     E. Export sample rule 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Imports and Connection to SF (skipped for Demo)

# COMMAND ----------

!pip install tensorflow  


# COMMAND ----------

!pip install tensorflow_decision_forests


# COMMAND ----------

!pip install flatten-dict

# COMMAND ----------

dbutils.library.restartPython()




# COMMAND ----------

#connect to SF
from block_cloud_auth.authenticators import SnowflakeAuthenticator, SnowflakeProvider

snowflake_credentials = SnowflakeAuthenticator(SnowflakeProvider()).get_credentials()

#provide options
options = {
    "sfUrl": "https://square.snowflakecomputing.com/",
    "sfUser": snowflake_credentials.user,
    "sfPassword": snowflake_credentials.password,
    "sfDatabase": snowflake_credentials.database,
}

#standard inputs and formatting
from pysnowflake import Session
import pandas as pd
from pandasql import sqldf
import numpy as np
from datetime import datetime

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)

pd.options.display.max_colwidth = 200
from flatten_dict import flatten

#tensor flow imports for decision trees
import tensorflow_decision_forests as tfdf
import tensorflow as tf


#pandas sql function
run_query = lambda query: sqldf(query, globals())



# COMMAND ----------

#replace below with your ldap name
USER_NAME = 'jobyg'

#establish snowflake session
sess = Session(
   connection_override_args={
       'autocommit': True,
       'authenticator': 'externalbrowser',
       'account': 'square',
       'database': f'PERSONAL_{USER_NAME.upper()}',
       'user': f'{USER_NAME}@squareup.com'
   }
)
#click the url, copy and paste the url and then hit enter to finalize authentication to snowflake 
conn = sess.open()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #Step A: Pull feature data from previously created feature table
# MAGIC
# MAGIC We need to grab the control group transactions from the AU new user base. We can specify whether we want the decision tree to optimize for tokens that result in **NTL** or **P2_D0**
# MAGIC
# MAGIC Given the analytical window i have chosen, i am going to use P2D0

# COMMAND ----------

#provide input rule: 
# target_rule = 'gb_fraud_online_email_age_amt_v2_migrated'
target_rule = 'anz_fraud_online_network_address_phone_checks_v2_migrated'

#specify par region and checkpoint
checkpoint = 'CHECKOUT_CONFIRM'#capitalization is important
# par_region = 'GB'
par_region = 'AU' #capitalization is important
target_column = 'p2_d0' #target column accepts values of p2_d0 or ntl

# COMMAND ----------

#basic query to create feature driver tool

# COMMAND ----------



# COMMAND ----------

from functions import pull_decision_tree_driver
dtree_driver = pull_decision_tree_driver(
    feat_driver_table='fraud_segmentation_tool_feat_driver',
    control_token_table= 'fraud_segmentation_tool_control_declines',
    target_column='p2_d0',
    conn=conn
)

# COMMAND ----------

dtree_driver.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Step B: Eva

# COMMAND ----------

from functions import grab_features_and_logic_using_rule_name

# COMMAND ----------

rule_features, rule_content = grab_features_and_logic_using_rule_name(target_rule=target_rule,
                           conn=conn)
#returns a dataframe that has rule name, last modified date and rule content
print(rule_content)

# COMMAND ----------

from functions import format_code
#used to lint the python function used in rule editor to be able to identify order tokens hit by the rule in the control group
rule_content = format_code(rule_content)
print(rule_content)

# COMMAND ----------

from functions import format_rule
executable_rule = format_rule(rule_content, rule_features)
print(executable_rule)

#make function executable
from functions import string_to_function
executable_rule = string_to_function(executable_rule)

# COMMAND ----------

#convert datatypes to appropriate values, otherwise we get TypeErrors
from functions import get_datatypes

dtree_driver, dtype_dict = get_datatypes(decision_tree_driver=dtree_driver)

# COMMAND ----------

#execute the rule to identify which tokens are flagged by the rule, we'll use this to determine the rule's performance in the training sample
dtree_driver['target_rule_flag'] = dtree_driver[rule_features].apply(executable_rule, axis=1)

# COMMAND ----------

target_df = run_query('select * from dtree_driver where days_since_first_order_date < 15 and whitepages_identity_network_score > .88 and whitepages_identity_check_score >= 480 and whitepages_primary_phone_checks_match_to_name in ("No match", "No name found") and whitepages_primary_address_checks_match_to_name = "No match"')

# COMMAND ----------

target_df[['days_since_first_order_date','whitepages_identity_network_score','whitepages_identity_check_score','whitepages_primary_phone_checks_match_to_name','whitepages_primary_address_checks_match_to_name','target_rule_flag']].head()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from functions import prep_for_training
# MAGIC
# MAGIC
# MAGIC
# MAGIC X, dtree_driver   = prep_for_training(decision_tree_driver = dtree_driver,
# MAGIC                                       exclude_features=['days_since_first_order_date'],
# MAGIC                                       test_ratio=.25) 
# MAGIC
# MAGIC from functions import split_dataset
# MAGIC
# MAGIC #split train data
# MAGIC
# MAGIC train, val = split_dataset(X)
# MAGIC
# MAGIC #specify column to be used as target
# MAGIC label = 'target'
# MAGIC #specify target values
# MAGIC classes = [0,1]
# MAGIC train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label=label)
# MAGIC val_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(val, label=label)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

#hyperparameter tuning 
bootstrap_size_ratio = [.25,.5,.75,.9] 
#bootstrapping ratio determines what percentage of RECORDS is used for training the decision trees,
#decision trees that are built on different sets of records, are more likely to be different

num_candidate_attributes_ratio = [.4,.55,.7,.85,.925]
#num candidate attributes ratio determines what percentage of FEATURES is used for training the decision trees,
#decision trees that are built on different sets of features, are more likely to be different

train_rows = train.shape[0]
min_examples_param_list = [train_rows//200, train_rows//100, train_rows//50,train_rows//25]
#min_examples_param_list determines the minimum number of records that must be present when creating segments
#for example a min_examples hyper parameter of 100 means no final segment will contain less than 100 records
#a higher value of this parameter results in less over-fit segments
 
max_depth = [3,5,7] 
#max depth determines how many times the tree can split, a higher value of this parameter can result in over-fitting
#as the tree will be able to form precise segments that only perform well on the training data through use of excessive splitting

num_trees = [100,101,102,103]
#num trees determines how many trees are built as part of the random forest, a larger number of trees results in a better performing model, but takes longer to train


# Create a Random Search tuner with 60 trials and specified hp configuration.
tuner = tfdf.tuner.RandomSearch(num_trials=10)
tuner.choice("min_examples", min_examples_param_list)
tuner.choice("num_candidate_attributes_ratio", bootstrap_size_ratio)
tuner.choice("num_trees", num_trees)
tuner.choice("max_depth", max_depth)
# tuner.choice("allow_na_conditions", allow_na_conditions)


#optional create weights for class labels due to imbalanced nature of p2 d0 transactions
total = len(train)
pos = np.sum(train['target'] == 1)
neg = np.sum(train['target'] == 0)
 
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
weight_ratio = weight_for_1/weight_for_0
class_weight = {0: weight_for_0, 1: weight_for_1}
 
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
print(f'ratio of weights is {weight_ratio}')
# Define and train the model.
tuned_model = tfdf.keras.RandomForestModel(tuner=tuner)
tuned_model.fit(train_ds,
                #verbose=2 #if you want logs to print, slows down the processing as log printing is slow
                class_weight=class_weight)


# COMMAND ----------

logs = tuned_model.make_inspector().tuning_logs() #assess tuning logs
num_trees =logs[logs.best==True]['num_trees'].iloc[0]
logs


# COMMAND ----------

#plot decision tree one
tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=8)


# COMMAND ----------

#convert decision tree into case when statement
def dfs_all_paths(node, path, all_paths):
    from collections import deque
    from tensorflow_decision_forests.component.py_tree import tree as tree_lib
    from tensorflow_decision_forests.component.py_tree.node import LeafNode as Leaf
    from tensorflow_decision_forests.component.py_tree.node import NonLeafNode as NonLeaf



    def negate_condition(condition):
        """
        Negate the given condition for the 'else' branch.
        """
        if '>=' in condition:
            return condition.replace('>=', '<')
        elif '>' in condition:
            return condition.replace('>', '<=')
        elif '<=' in condition:
            return condition.replace('<=', '>')
        elif '<' in condition:
            return condition.replace('<', '>=')
        elif '==' in condition:
            return condition.replace('==', '!=')
        elif '!=' in condition:
            return condition.replace('!=', '==')
        elif 'not in' in condition:
            return condition.replace('not in', 'in')
        elif ('in' in condition) and ('rating' not in condition):
            return condition.replace('in', 'not in')
       
        else:
            return condition
            
    if node is None:
        return
    # Append current node to the path
    try:
        node.condition = str(node.condition).split(';')[0]
        if node.condition[-1]!= ')':
            node.condition+=')'
        path.append(node.condition)
    except AttributeError:
        path.append(node.value)
   
    # If it's a leaf node, record the path
    if isinstance(node, Leaf):
        all_paths.append(path.copy())
    else:
        # Traverse left and right children
        if (node.pos_child) and isinstance(node.pos_child, NonLeaf):
            
            node.pos_child.condition = str(node.pos_child.condition).split(';')[0]
            if str(node.pos_child.condition)[-1]!= ')':
                node.pos_child.condition+=')'
            dfs_all_paths(node.pos_child, path, all_paths)
        else:
            dfs_all_paths(node.pos_child, path, all_paths)
        if (node.neg_child) and isinstance(node.neg_child, NonLeaf):
            path[-1] = negate_condition(path[-1]) 
            # print(path)   
            node.neg_child.condition = str(node.neg_child.condition).split(';')[0]
            if str(node.neg_child.condition[-1])!= ')':
                node.neg_child.condition+=')'
            dfs_all_paths(node.neg_child, path, all_paths)
        else:
            path[-1] = negate_condition(path[-1])            
            dfs_all_paths(node.neg_child, path, all_paths)

    
    # Backtrack
    path.pop()


# COMMAND ----------

inspector = tuned_model.make_inspector()
tree = inspector.extract_tree(tree_idx=0)
all_paths = []
dfs_all_paths(tree.root, [], all_paths)
        

# COMMAND ----------

threshold = .625
def get_case_when_statement(tuned_model,threshold):
    from tqdm import tqdm
    inspector = tuned_model.make_inspector()
    logs = tuned_model.make_inspector().tuning_logs() #assess tuning logs
    num_trees =logs[logs.best==True]['num_trees'].iloc[0]
    dict_df = {}
    for i in tqdm(range(num_trees)): 
        # print(i)
        dict_df[i] = {}
        tree = inspector.extract_tree(tree_idx=i)
        all_paths = []
        dfs_all_paths(tree.root, [], all_paths)
        for j in range(len(all_paths)):
            path = all_paths[j]
            loss = float(repr(path[-1]).split(',')[1].split(']')[0].strip())
            if loss >= threshold:
                # print('found a promising segment')
                path = all_paths[j]
                path_str = path[:-1]
                path_str=' and '.join(path_str)
                new_path_str = 'CASE WHEN ' + path_str
                new_path_str += f'THEN 1 ELSE 0 END as s_{i}_{j}'
                new_path_str = new_path_str.replace('[', '(')
                new_path_str = new_path_str.replace(']', ')')
                # print(path[-1])
                n = repr(path[-1]).split('n=')[1].split(')')[0]
                loss = repr(path[-1]).split(',')[1].split(']')[0].strip()
                dict_df[i][j] = [new_path_str,n,loss]
    return(dict_df)

# COMMAND ----------

outputs_dict = get_case_when_statement(tuned_model, threshold=threshold) 
#this will take up to 15-20 minutes depending on the threshold specified and num trees.

#The function iterates over all paths in a decision tree, for all trees in the random forest, logic is set to try to focus on high performing trees with the threshold 


# COMMAND ----------



# COMMAND ----------

final_dict = flatten(outputs_dict)
for k,v in final_dict.items():
    v[0]= v[0].replace('[', '(')
    v[0] = v[0].replace(']', ')')
                   

# COMMAND ----------

def evaluate_rules(dtree_driver, rule_dict,threshold):
    from tqdm import tqdm
    target_rule_ctrl_trxn_ct = dtree_driver.loc[dtree_driver['target_rule_flag']==1].target.value_counts().sum()
    target_rule_p2_d0 = dtree_driver.loc[dtree_driver['target_rule_flag']==1].p2_overdue_d0_local.sum()/dtree_driver.loc[dtree_driver['target_rule_flag']==1].p2_due_local.sum()
    print(f'target rule ctrl transaction count is {target_rule_ctrl_trxn_ct}, target rule p2_d0 is {target_rule_p2_d0}')
    performance_dict = {}

    query = 'select *'
    key_list = list(rule_dict.keys())

    first_third = len(key_list)//3
    second_third = 2*len(key_list)//3
    last_third = len(key_list)

    for i in range(0,first_third):
        key = key_list[i]
        val = rule_dict[key]
        query+=f',{val[0]}'
            
    query += ' from dtree_driver'
    test_df = run_query(query)
    print('evaluating first third of the new rules as sql can only display a limited number of columns')
    print('there will be three progress bars as a part of this function, this is progress bar 1')
    for i in tqdm(range(0, first_third)):
        key = key_list[i]
        v = rule_dict[key]
        segment_name = v[0].split(' as ')[1]
        new_rule_p2_d0 = test_df.loc[test_df[segment_name]==1].p2_overdue_d0_local.sum()/test_df.loc[test_df[segment_name]==1].p2_due_local.sum()
        if new_rule_p2_d0 - target_rule_p2_d0 > threshold:
            new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
            performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]

    query = 'select *'
    for i in range(first_third, second_third):
        key = key_list[i]
        val = rule_dict[key]
        query+=f',{val[0]}'
            
    query += ' from dtree_driver'
    test_df = run_query(query)
    print('evaluating second third of the new rules as sql can only display a limited number of columns')
    print('there will be three progress bars as a part of this function, this is progress bar 2')
    for i in tqdm(range(first_third, second_third)):
        key = key_list[i]
        v = rule_dict[key]
        segment_name = v[0].split(' as ')[1]
        new_rule_p2_d0 = test_df.loc[test_df[segment_name]==1].p2_overdue_d0_local.sum()/test_df.loc[test_df[segment_name]==1].p2_due_local.sum()
        if new_rule_p2_d0 - target_rule_p2_d0 > threshold:
            new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
            performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]
    query = 'select *'
    for i in range(second_third, last_third):
        key = key_list[i]
        val = rule_dict[key]
        query+=f',{val[0]}'
            
    query += ' from dtree_driver'
    test_df = run_query(query)
    print('evaluating last third of the new rules as sql can only display a limited number of columns')
    print('there will be three progress bars as a part of this function, this is progress bar 3')
    for i in tqdm(range(second_third, last_third)):
        key = key_list[i]
        v = rule_dict[key]
        segment_name = v[0].split(' as ')[1]
        new_rule_p2_d0 = test_df.loc[test_df[segment_name]==1].p2_overdue_d0_local.sum()/test_df.loc[test_df[segment_name]==1].p2_due_local.sum()
        if new_rule_p2_d0 - target_rule_p2_d0 > threshold:
            new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
            performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]


    
    return(performance_dict)


# COMMAND ----------

#this code will also take a couple minutes to run as we are evaluating the control group p2d0 of hundreds / thousands of rules
performance_dict = evaluate_rules(dtree_driver, final_dict, .1)

# COMMAND ----------

performance_df = pd.DataFrame(performance_dict.values(), columns =['new_rule_ctrl_trxn_ct','new_rule_p2_d0','rule'])

# COMMAND ----------

performance_df.sort_values(by='new_rule_p2_d0', ascending=False).head(50)

# COMMAND ----------

#sample rule
rule_to_author = performance_df.loc[6].rule


# COMMAND ----------

 def create_py_rule(rule,rule_name,path_name):
        from functions import format_code
        import re

        intro = '''def execute_rule():
            actions = []
            ### BEGIN RULE CONTENT ###'
            '''
        rule = rule.split('END as')[0]
        rule = rule.replace('CASE WHEN','if')
        rule = rule.replace("THEN 1 ELSE 0 ",":")
        # intro+='\n'
        intro+= f'{rule}'
        # intro+='\n\t\t'
        intro+="\n\t\tactions.append({'action_name': 'is_rejected_assign'})"
        intro+="""\n\t   ### END RULE CONTENT ###\n"""
        intro+='\t   return actions'
        # intro+='\n\treturn actions'
        print(intro)
        pattern = r'\b(in)\s*\(\s*((?:\'[^\']*\'|\d+)(?:\s*,\s*(?:\'[^\']*\'|\d+))*)\s*\)'
        replacement = r'\1 {\2}' 
        intro = re.sub(pattern, replacement, intro)
        intro = format_code(intro)
        #replace in parenthesis logic with braces 
 
        print(intro)
        with open(f'{path_name}{rule_name}', 'w') as f:
            f.write(intro)
        return(intro)

# COMMAND ----------

rule_to_author = create_py_rule(rule_to_author,  rule_name='test_rule2.py', path_name ='/Workspace/Users/jobyg@squareup.com/Fraud Segmentation Tool /')



# COMMAND ----------

dtree_driver.columns

# COMMAND ----------

performance_df.loc[(performance_df.rule.str.contains('de_merchant_category_code')==False)]

# COMMAND ----------

test_analysis = run_query(f'''select *, {test_case_when} from dtree_driver
                          ''')



# COMMAND ----------

control_target_rule_p2_do_rate = test_analysis.loc[test_analysis['target_rule_flag']==1].p2_overdue_d0_local.sum()/test_analysis.loc[test_analysis['target_rule_flag']==1].p2_due_local.sum()

# COMMAND ----------

control_target_rule_impacted_transactions = test_analysis.loc[test_analysis['target_rule_flag']==1].target.value_counts().sum()
control_target_rule_transaction_ct_accuracy = test_analysis.loc[test_analysis['target_rule_flag']==1].target.value_counts(normalize=True)[1]
control_target_rule_p2_do_rate = test_analysis.loc[test_analysis['target_rule_flag']==1].p2_overdue_d0_local.sum()/test_analysis.loc[test_analysis['target_rule_flag']==1].p2_due_local.sum()

print(control_target_rule_impacted_transactions, control_target_rule_transaction_ct_accuracy, control_target_rule_p2_do_rate)


# COMMAND ----------

sample_rule_control_impacted_transactions = test_analysis.loc[test_analysis['segment_0_0']==1].target.value_counts().sum()
control_sample_rule_transaction_ct_accuracy = test_analysis.loc[test_analysis['segment_0_0']==1].target.value_counts(normalize=True)[1]
control_sample_rule_p2_do_rate = test_analysis.loc[test_analysis['segment_0_0']==1].p2_overdue_d0_local.sum()/test_analysis.loc[test_analysis['segment_0_0']==1].p2_due_local.sum()


print(sample_rule_control_impacted_transactions, control_sample_rule_transaction_ct_accuracy, control_sample_rule_p2_do_rate)


# COMMAND ----------

test_analysis.loc[test_analysis['segment_0_0']==1].target.value_counts()

# COMMAND ----------

test_analysis.loc['segment_0_0'].target.value_counts()

# COMMAND ----------

first_path = all_paths[0]
second_path = all_paths[1]
third_path = all_paths[2]
fourth_path = all_paths[3]



fifth_path = all_paths[4]
sixth_path = all_paths[5]
seventh_path = all_paths[6]
eighth_path = all_paths[7]

ninth_path = all_paths[8]
tenth_path = all_paths[9]
eleventh_path = all_paths[10]
twelvth_path = all_paths[11]
thirteenth_path = all_paths[12]

fourteenth_path = all_paths[13]
fifteenth_path = all_paths[14]
sixteenth_path = all_paths[15]

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=8)


# COMMAND ----------

thirteenth_path

# COMMAND ----------

len(all_paths)

# COMMAND ----------



# COMMAND ----------

for i in range(len(all_paths)):
    if i in [0,1]:
        first_split_condition = all_paths[i][1] #all_paths[i][1] != first_split condition we need to negate the first entry
        continue #first two leaf nodes are correct
    elif i <=:
        if all_paths[i][-3] == all_paths[i-1][-3]:
            all_paths[i][-3] = negate_condition(all_paths[i][-3])


# COMMAND ----------



# COMMAND ----------

print(tree.pretty())

# COMMAND ----------

all_paths

# COMMAND ----------

inspector = tuned_model.make_inspector()
tree = inspector.extract_tree(tree_idx=0)
 

# COMMAND ----------

test_all_paths = deepcopy(all_paths)
first_path = all_paths[0]
path_len = len(first_path)-1 #exclude the last value as it's probability

# split_var_dict = {}
# for i in range(path_len):
#     split_var_dict[i] = first_path[i]

def make_proper_conditions(all_paths, depth):

    def negate_condition(condition):
        """
        Negate the given condition for the 'else' branch.
        """
        if '>=' in condition:
            return condition.replace('>=', '<')
        elif '>' in condition:
            return condition.replace('>', '<=')
        elif '<=' in condition:
            return condition.replace('<=', '>')
        elif '<' in condition:
            return condition.replace('<', '>=')
        elif '==' in condition:
            return condition.replace('==', '!=')
        elif '!=' in condition:
            return condition.replace('!=', '==')
        elif 'not in' in condition:
            return condition.replace('not in', 'in')
        elif 'in' in condition:
            return condition.replace('in', 'not in')
       
        else:
            return condition
        
    first_path = all_paths[0]

    for i in range(len(all_paths)):
        if i in [0,1]:
            print(all_paths[i])
                # print(test_all_paths[i][index],first_path[index])
        if (all_paths[i][depth-2] != first_path[depth-2]) & ( (all_paths[i][depth-3] == first_path[depth-3])): #check if the last variable is the same as the first two trees, since it won't be we need to negate the condition prior to it
                    all_paths[i][depth-2] = negate_condition(all_paths[i][depth-2])
                    print(all_paths[i])
                # elif (test_all_paths[i][index] != first_path[-3]) & (test_all_paths[i][-4] == first_path[-4]):
                #     test_all_paths[i][-4] = negate_condition(test_all_paths[i][-4])
            
make_proper_conditions(test_all_paths, 6)

    #     continue
    # while  all_paths[i][1] == first_split_condition: #check to see if the first split condition is the same, 
    #     if all_paths[i][-3] == second_to_last_split_condition:
    #         #we need to negate this condition
    #         third_to_last_split_condition = path[-4]
    #         all_paths[i][-3] = negate_condition(all_paths[i][-3])
            
       


# COMMAND ----------

split_var_dict = {}
first_path = all_paths[0]
last_path = all_paths[-1]
for i in range(path_len):
    split_var_dict[i] = [first_path[i],last_path[i]]
    




# COMMAND ----------

split_var_dict

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=8)


# COMMAND ----------

split_var_dict

# COMMAND ----------

all_paths[4]

# COMMAND ----------

for path in all_paths:
    print(path)
    

# COMMAND ----------

inspector = tuned_model.make_inspector()
tree = inspector.extract_tree(tree_idx=0)
 

# COMMAND ----------

print(all_paths[0][-3])

# COMMAND ----------

from copy import deepcopy
test_root = deepcopy(tree.root)
all_paths = []
seen_set = set()
dfs_all_paths(test_root, [], all_paths, seen_set=seen_set)


# COMMAND ----------

return_dict, label_dict = bfs(test_root,6)

# COMMAND ----------

return_dict

# COMMAND ----------

print(return_dict)

# COMMAND ----------

print(label_dict)

# COMMAND ----------

all_paths[0]

# COMMAND ----------

test_root = deepcopy(tree)

# COMMAND ----------

test_root.root.pos_child.condition = str(test_root.root.pos_child.condition).split(';')[0]+')'

# COMMAND ----------

test_root.root.pos_child.pos_child

# COMMAND ----------

from copy import deepcopy

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=4)


# COMMAND ----------

all_paths = []
dfs_all_paths(tree.root, [], all_paths)
for path in all_paths:
    print(path)


# COMMAND ----------

for i in range(len(all_paths)):
    if i == 0:
        continue
    elif

# COMMAND ----------

tree.root.pos_child.pos_child.pos_child

# COMMAND ----------

bfs(tree.root,4)

# COMMAND ----------

d# conditions_list = [
#     d[1]['left']['condition'] + d[2]['left']['condition'] + d[4]['left']['condition'],
#     d[1]['left']['condition'] + d[2]['left']['condition'] + d[4]['right']['condition'],
#     d[1]['left']['condition'] + d[2]['right']['condition'] + d[5]['left']['condition'],
#     d[1]['left']['condition'] + d[2]['right']['condition'] + d[5]['right']['condition'],
#     d[1]['right']['condition'] + d[3]['left']['condition'] + d[6]['left']['condition'],
#     d[1]['right']['condition'] + d[3]['left']['condition'] + d[6]['right']['condition'],
#     d[1]['right']['condition'] + d[3]['right']['condition'] + d[7]['left']['condition'],
#     d[1]['right']['condition'] + d[3]['right']['condition'] + d[7]['right']['condition']
# ]
i = 1
while i < max(label_dict):
    print(i)
    left_split = return_dict[i]['left']['condition']
    right_split = return_dict[i]['right']['condition']
    print(i)
    i+=1 
    second_left_split = return_dict[i]['left']['condition']
    third_left_split = 
    
    

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=4)


# COMMAND ----------

def generate_conditions(d):
    # Initialize list to store conditions
    conditions_list = []
    
    # Extract initial conditions from d[1] and d[2]
    initial_conditions = {
        'd1_left': d[1]['left']['condition'],
        'd1_right': d[1]['right']['condition'],
        'd2_left': d[2]['left']['condition'],
        'd2_right': d[2]['right']['condition']
    }
    
    # Generate conditions based on the length of the dictionary
    num_entries = len(d)
    
    for key in range(1, num_entries + 1):
        if key == 1 or key == 2:
            continue
        
        # Extract conditions for the current key
        current_left = d[key]['left']['condition'] if 'left' in d[key] else ''
        current_right = d[key]['right']['condition'] if 'right' in d[key] else ''
        
        if key % 2 == 0:  # For even keys
            conditions_list.append(
                initial_conditions['d1_left'] + initial_conditions['d2_left'] + current_left
            )
            conditions_list.append(
                initial_conditions['d1_left'] + initial_conditions['d2_left'] + current_right
            )
        else:  # For odd keys
            conditions_list.append(
                initial_conditions['d1_right'] + d[key]['left']['condition']
            )
            conditions_list.append(
                initial_conditions['d1_right'] + d[key]['right']['condition']
            )
    
    return conditions_list

# Define the dictionary with conditions
d = {
    1: {'left': {'condition': '(a >= 0.003297204617410898) and'},
        'right': {'condition': '(a < 0.003297204617410898) and'}},
    2: {'left': {'condition': '(b >= 1757.0) and'},
        'right': {'condition': '(b < 1757.0) and'}},
    3: {'left': {'condition': '(c >= 258.5) and'},
        'right': {'condition': '(c < 258.5) and'}},
    4: {'left': {'condition': '(d >= 158.5) and'},
        'right': {'condition': '(d < 158.5) and'}},
    5: {'left': {'condition': '(e >= 17.049999237060547) and'},
        'right': {'condition': '(e < 17.049999237060547) and'}},
    6: {'left': {'condition': '(f >= 0.3725000023841858) and'},
        'right': {'condition': '(f < 0.3725000023841858) and'}},
    7: {'left': {'condition': '(g >= 168.96499633789062) and'},
        'right': {'condition': '(g < 168.96499633789062) and'}}
}

# Generate the conditions
conditions = generate_conditions(d)

# Print the results
for condition in conditions:
    print(condition)

# COMMAND ----------


len(conditions)


# COMMAND ----------

len(cases)

# COMMAND ----------

(number_of_segments/2)-1

# COMMAND ----------

return_dict

# COMMAND ----------

number_of_segments = max(label_dict)

case_list = []
i = 1
depth = 0


threshold = (number_of_segments/2)-1
print(threshold)

# COMMAND ----------

case_list= []

while len(case_list) != number_of_segments
    i = 1
    depth = 0
    while depth <= 1:
        if 
        first_str = 'CASE WHEN'
        second_str = return_dict[i]['left']['condition']
        i+=1
        depth+=1
        print(second_str, depth)

        third_str = return_dict[i]['left']['condition']
        depth+=1
        print(third_str,depth)

        i+=2
        fourth_str = return_dict[i]['left']['condition']
        depth+=1
        print(fourth_str,depth)
        print(rule_1)
        case_list.append(rule_1 = first_str+second_str+third_str+fourth_str)


# COMMAND ----------

return_dict

# COMMAND ----------

rule_1

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=4)


# COMMAND ----------

number_of_segments

# COMMAND ----------

case_list = []
for k,v in return_dict.items():
    base_str = 'CASE WHEN'
    if k == 1:
        left_tree, right_tree = return_dict[k]['left']['condition'], return_dict[k]['right']['condition']

    elif k == 2: #this will be the left tree
        for k2,v2 in return_dict[k].items():
            # print(left_tree,v2)
            case_list.append(base_str + left_tree+ v2['condition']) #this grabs yes to the root, and then yes and no to the first split
    elif k == 3: #this will be the right tree
        for k2,v2 in return_dict[k].items():
            case_list.append(base_str + right_tree+ v2['condition'])
print(case_list)
    else if k%2 == 0:
        

    #             if k2

        

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=4)


# COMMAND ----------

return_dict.pop('start')

# COMMAND ----------

def generate_specific_cases(conditions_dict):
    base_condition = 'CASE WHEN'
    cases = []

    # Extract conditions for key 1
    key_1_conditions = conditions_dict[1]
    left_1 = key_1_conditions['left']['condition']
    right_1 = key_1_conditions['right']['condition']

    # Iterate through the dictionary to match with even and odd keys
    for key, value in conditions_dict.items():
        if key == 1:
            continue  # Skip the first key as it's the reference key

        condition = value['left']['condition'] if key % 2 == 0 else value['right']['condition']
        condition_str = f"{left_1} {condition}".strip()[:-4]  # Remove trailing ' and'
        
        case_statement = f"{base_condition} {condition_str} THEN 1 ELSE 0"
        cases.append(case_statement)

    return cases

# Generate the CASE WHEN statements
specific_cases = generate_specific_cases(return_dict)

# Print the results
for case in specific_cases:
    print(case)


# COMMAND ----------

return_dict

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=4)


# COMMAND ----------

return_dict

# COMMAND ----------

dcases

# COMMAND ----------

rule_list = []
for k,v in return_dict.items():
    left_rule = ''
    if k == 'start':
        left_rule+= k
    elif k == 1:
        rule+=return_dict[k]['left']



# COMMAND ----------

return_dict

# COMMAND ----------

label_dict

# COMMAND ----------

def create_historical_decline_table(rule_decline_table_name, 
                                    
                                    rule_list,
                                    user_name, 
                                    start_date, 
                                    end_date,  
                                    par_region,
                                    checkpoint, 
                                    conn,
                                    need_decline_rate_denom=False):
    print(f'identifying historical declines for the following rules:{rule_list}')
    rule_decline_sql = f"""CREATE or replace table ap_cur_frdrisk_g.public.{rule_decline_table_name} AS (SELECT 
        a.order_token, a.key_Event_info_id, a.event_info_event_time, a.rule_id, a.par_region, a.checkpoint,
        FROM  AP_CUR_R_FEATSCI.curated_feature_science_red.TBL_RAW_C_E_FC_DECISION_RECORD_RULES__jobyg_DSL3_SV a
        WHERE 1=1 
        AND (is_rejected = 'True' and is_in_treatment = 'True')
        AND a.par_process_date between '{start_date}' and '{end_date}'
        and rule_id in ( """
    for i in range(len(rule_list)):
        rule = rule_list[i]
        if i != len(rule_list)-1:
            rule_decline_sql+=f"'{rule}', "
        else:
            rule_decline_sql+=f"'{rule}')"
    rule_decline_sql+=");"
    print(f'historical decline query is {rule_decline_sql}')

    conn.execute(rule_decline_sql)
    print(f'finished creating ap_cur_frdrisk_g.public.{rule_decline_table_name}')



# COMMAND ----------

rule_list = ['anz_fraud_online_network_address_phone_checks_v2_migrated']
start_date, end_date ='2024-01-01', '2024-07-31'

create_historical_decline_table(rule_decline_table_name='fraud_segmentation_tool_demo_historical_declines',
 rule_list=rule_list, 
 user_name=USER_NAME,
 start_date=start_date,
 end_date=end_date, 
 par_region=par_region,
 checkpoint=checkpoint,
 conn=conn
 )
#this will grab the number of declines, now we need to find the unique declines as well as the total number of tokesn to get decline rate 

# COMMAND ----------

def get_decline_rate_denom(start_date, end_date, par_region, checkpoint, user_name, conn):
    decline_rate_denom_q = f'''
        select checkpoint,
               par_region, 
               count(distinct(order_token)) as token_ct,
               sum(consumer_order_amount) as order_amount
               from AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.TBL_RAW_C_E_FC_DECISION_RECORD__{user_name}_DSL3_SV 
               where checkpoint ilike '%{checkpoint}%' and
               par_region ilike '%{par_region}%'
               and par_process_date between '{start_date}' and '{end_date}' 
               and days_since_first_order_date < 15
               group by 1,2
               order by 1,2
               '''
    print(decline_rate_denom_q)
    decline_rate = conn.download(decline_rate_denom_q)
    print(f'decline rate denom is {decline_rate}')
    return(decline_rate)


# COMMAND ----------

decline_rate_denom = get_decline_rate_denom(start_date, end_date, par_region, checkpoint,USER_NAME, conn)

# COMMAND ----------

decline_rate_denom.token_ct.iloc[0]

ANZ_token_denominator = 1023484
ANZ_dollar_denominator = 174827807.2

# COMMAND ----------

def create_unique_decline_table(unique_decline_table_name, start_date, end_date, par_region, checkpoint, user_name, conn): 

    unique_decline_driver = f"""CREATE or replace table ap_cur_frdrisk_g.public.{unique_decline_table_name} as (
     SELECT order_token,EVENT_INFO_EVENT_TIME
    ,count(distinct rule_id) as N_decline
     FROM  AP_CUR_R_FEATSCI.curated_feature_science_red.TBL_RAW_C_E_FC_DECISION_RECORD_RULES__{user_name}_DSL3_SV
     WHERE checkpoint ilike '{checkpoint}'
    AND ((is_rejected = 'True' and is_in_treatment = 'True') or (actions like '%is_rejected_assign%' and is_in_treatment = 'True'))
    AND par_process_date between '{start_date}' and '{end_date}'
    and par_region ilike '{par_region}'
    group by 1,2
    );
    """
    conn.execute(unique_decline_driver)
    print(f'finished creating the unique decline driver')

# COMMAND ----------

create_unique_decline_table('fraud_segmentation_tool_demo_unique_declines',
 start_date,
 end_date,
 par_region,
 checkpoint,
 USER_NAME,
 conn)

# COMMAND ----------

def create_control_table(control_table_name, start_date, end_date, checkpoint, conn):
    control_query = f"""CREATE or replace table AP_CUR_FRDRISK_G.public.{control_table_name} as (
    select g.*,case when (lower(g.control_group) like '%attempt_online_control_group%'
        or lower(g.control_group) like '%attempt_in_store_control_group%'
        or lower(g.control_group) like '%global_control_group%') then 1 else 0 end as is_in_control_group
        , case when is_in_control_group = 1 and g.control_group in ('["global_control_group"]'
        , '["in_store_global_control_group"]'
        , '["in_store_global_control_group", "account_control_group"]'
        , '["global_control_group", "account_control_group"]'
        , '["attempt_online_control_group"]'
        , '["attempt_in_store_control_group"]'
        , '["attempt_online_control_group", "account_online_control_group"]'
        , '["attempt_in_store_control_group", "account_in_store_control_group"]') then 0
        when is_in_control_group = 1 then 1 else -1 end as is_control_declined_historical,

         case when is_in_control_group = 1 and g.control_group in ('["global_control_group"]'
        , '["in_store_global_control_group"]'
        , '["in_store_global_control_group", "account_control_group"]'
        , '["global_control_group", "account_control_group"]'
        , '["attempt_in_store_control_group"]'
        , '["attempt_online_control_group", "account_online_control_group"]'
        , '["attempt_in_store_control_group", "account_in_store_control_group"]') then 0
        when is_in_control_group = 1 then 1 else -1 end as is_control_declined_new_rules_online,
        d.p2_overdue_d0_local,
        d.p2_due_local,
        d.ntl_forecast_local,
        from ap_cur_frdrisk_g.public.HYANG_SUMO_CONTROL_GROUP_BASE_COMBINED g
    left join  AP_CUR_RISKBI_G.CURATED_RISK_BI_GREEN.DWM_ORDER_LOSS_TAGGING  d
    on g.order_id = d.order_token
    and to_varchar(g.asctime, 'YYYY-MM-DD')  = d.order_date   
         where g.checkpoint = '{checkpoint}'
        and g.asctime between '{start_date}' and '{end_date}'
    );
    """
    conn.execute(control_query)

# COMMAND ----------

create_control_table('fraud_segmentation_tool_control_Declines', start_date, end_date, checkpoint, conn)

# COMMAND ----------

hist_declines = conn.download(f"select rule_id, count(distinct(order_token)) as fc_decline_ct from ap_cur_frdrisk_g.public.fraud_segmentation_tool_demo_historical_declines group by 1")

# COMMAND ----------

hist_declines['decline_rt'] = hist_declines['fc_decline_ct'] / ANZ_token_denominator

# COMMAND ----------

hist_declines.T

# COMMAND ----------

def bfs(root,depth):
    from collections import deque
    from tensorflow_decision_forests.component.py_tree import tree as tree_lib
    from tensorflow_decision_forests.component.py_tree.node import LeafNode as Leaf
    from tensorflow_decision_forests.component.py_tree.node import NonLeafNode as NonLeaf


    def negate_condition(condition):
        """
        Negate the given condition for the 'else' branch.
        """
        if '>=' in condition:
            return condition.replace('>=', '<')
        elif '>' in condition:
            return condition.replace('>', '<=')
        elif '<=' in condition:
            return condition.replace('<=', '>')
        elif '<' in condition:
            return condition.replace('<', '>=')
        elif '==' in condition:
            return condition.replace('==', '!=')
        elif '!=' in condition:
            return condition.replace('!=', '==')
        elif 'not in' in condition:
            return condition.replace('not in', 'in')
        elif 'in' in condition:
            return condition.replace('in', 'not in')
       
        else:
            return condition

    if root is None:
        return []

    result_dict = {"start": "CASE WHEN"}
    label_dict = {}
    depth = 0
    i=0
    queue = deque([root])  # Initialize the queue with the root node

    while queue:
        node = queue.popleft()  # Dequeue the front node
        
        if isinstance(node, Leaf):
            label_dict[i] = {}
            if i%2 == 0:
                label_dict[i]['left'] = {}
                label_dict[i]['left']['label_percentage'] = node.value.probability
                label_dict[i]['left']['num_examples'] = node.value.num_examples
                i+=1
            else:
                label_dict[i]['right'] = {}
                label_dict[i]['right']['label_percentage'] = node.value.probability
                label_dict[i]['right']['num_examples'] = node.value.num_examples
                i+=1
        else:
            depth+=1
            condition = repr(node.condition).split(';')[0].split('(')[1].rstrip(')')
            negated_condition =negate_condition(condition)
            result_dict[depth] = {}
            result_dict[depth]['left'], result_dict[depth]['right'] = {}, {}
            result_dict[depth]['left']['condition'] = f"({condition}) and"  # Process the node (record its value)
            result_dict[depth]['right']['condition'] = f"({negated_condition}) and"
            if node.pos_child and node.neg_child:
                left, right = node.pos_child, node.neg_child
                queue.append(left)
                queue.append(right)
            # elif node.pos_child:
            #     left = node.pos_child
            #     queue.append(left)
            #     depth+=1
            # elif node.neg_child:
            #     queue.append(right)
            #     depth+=1
        # print(result_dict)


    return(result_dict, label_dict)

