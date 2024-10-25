# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fraud Segmentation Tool 
# MAGIC ## Date: 10/16/2024
# MAGIC ## Author: Joby George
# MAGIC
# MAGIC # Background
# MAGIC
# MAGIC Fraud Decision Scientsits often have to clean up rules, this can invovle a couple of primary activities, namely:
# MAGIC
# MAGIC a. changing the splitting points of features that are currently being used to decrease action rate or increase accuracy
# MAGIC
# MAGIC b. adding additional risk splitters to existing rules to improve either accuracy or action rate
# MAGIC
# MAGIC c. creating new rules from scratch given a new feature or set of features
# MAGIC
# MAGIC This is the first script in a series of scripts that goes over how to automate the process and improve efficiency, creating a feature driver table that will then be used to conduct segmentation and evaluate performance.
# MAGIC
# MAGIC This demo starts with a common use case, improving a rule's performance. The rule chosen was **anz_fraud_online_network_address_phone_checks_v2_migrated**.
# MAGIC
# MAGIC The steps in this script include:
# MAGIC
# MAGIC     I. Set Up (imports, connection to snowflake)
# MAGIC     II. Grabbing the features used in the target rule 
# MAGIC     III. (Optional) Finding additional relevant features to use in segmentation 
# MAGIC     IV. Creating the feature driver table 
# MAGIC     V. Creating the analytical driver tables (control group, historical performance)
# MAGIC     VI. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #Step I: Set Up 
# MAGIC
# MAGIC Importing packages, connecting to snowflake

# COMMAND ----------

!pip install block-cloud-auth

# COMMAND ----------

!pip install pandasql==0.7.3

# COMMAND ----------

!pip install sq-pysnowflake

# COMMAND ----------

!pip install sqlalchemy-databricks==0.2.0

# COMMAND ----------

!pip install tensorflow  


# COMMAND ----------

!pip install tensorflow_decision_forests


# COMMAND ----------

!pip install flatten-dict

# COMMAND ----------

dbutils.library.restartPython() 
#this is a databricks specific command, it may not be necessary for google notebooks

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
from datetime import datetime, timedelta
import numpy as np

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)

pd.options.display.max_colwidth = 200

#pandas sql function
run_query = lambda query: sqldf(query, globals())

from flatten_dict import flatten

#tensor flow imports for decision trees
import tensorflow_decision_forests as tfdf
import tensorflow as tf


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

#use x_large warehouse to speed up querying 
conn.execute('use warehouse ADHOC__XLARGE')


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Step II: Grabbing features used in the target rule
# MAGIC
# MAGIC     Part 1: Use Rule Content table to grab features in the target rule
# MAGIC     Part 2: Use Decision Record Vars to grab additional vars of interest  (Semi-Manual exercise)

# COMMAND ----------

#replace below with your input rule
target_rule = 'anz_fraud_online_network_address_phone_checks_v2_migrated'

#grab features and rule logic used in that rule
from functions import grab_features_and_logic_using_rule_name
rule_features, rule_content = grab_features_and_logic_using_rule_name(target_rule, conn=conn)


# COMMAND ----------

rule_features 

print(f'rule_feats {rule_features}\n\n rule_content:\n {rule_content}')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Improving the rule
# MAGIC
# MAGIC The rule has two segments:
# MAGIC
# MAGIC Segment 1: for first orders, customers with a high WPP network score and identity score with a phone that does not link them to their name are declines
# MAGIC
# MAGIC Segment 2: for non first orders, if a customer meets the above criteria, **and** they have a No match for address matching their name, we will decline the transaction

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # (Optional) Step III: Finding additional relevant features to use in segmentation

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Finding complimentary features that could help improve the risk splititng power of the rule is relatively tricky, as the decision tree could overfit on very granular features (such as merchant id). I glanced at recent control group declines for this rule and saw there may be some value in adding the following features as risk splitters:
# MAGIC
# MAGIC
# MAGIC     IN_FLIGHT_ORDER_AMOUNT
# MAGIC     SP_ENTITY_LINKING_HOP0_TOT_ORDER_CNT_BY_MERCH_SIDE_EMAIL_H72_0
# MAGIC     SP_C_ONLINE_ORDR_ATTMPT_CREDIT_CARD_CNT_H12_0
# MAGIC     IN_FLIGHT_CARD_NAME_VS_PROFILE_NAME
# MAGIC     SP_C_ONLINE_DECL_TOPAZ_INSFFCNT_FUND_ORDR_CNT_H12_0
# MAGIC     SP_D_LINKING_HOP0_ORDER_ATTMPT_CNT_BY_DEVICE_ID_H1_0
# MAGIC     SP_C_PYMT_ATTMPT_CNT_H24_0
# MAGIC     SP_C_ORDER_ATTEMPT_CNT_D1
# MAGIC     BP_UDP_C_GRAPH_MODEL_SCORE
# MAGIC     INFLIGHT_DEVICE_ID_CONSUMER_DISTINCT_CNT
# MAGIC     SP_ADDRESS_LINKING_TOTAL_CONSUMER_CNT_BY_RAW_SHIPPING_HASH_D3_0
# MAGIC     BP_C_ALL_DEVICE_LINKING_CUST_CNT
# MAGIC     BP_C_SEED_BASED_LINKING_DEVICE_ID
# MAGIC     SP_C_ORDER_AMT_SAME_MERCHANT_AS_CURRENT_H24_0
# MAGIC     BP_C_OUTSTANDING_BALANCE_AVG_AMT_30D_v2
# MAGIC     BP_C_SEED_CNT_LINKED_BY_DEVICE_ID
# MAGIC     BP_C_SEED_CNT_LINKED_BY_RAW_SHIPPING_ADDRESS
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC However, for the purposees of this demo, we will go through the exercise twice, once without adding any incremental features, to see if threshold modification itself can lead to a better performing rule.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Step IV: Creating the driver table:
# MAGIC
# MAGIC     Part 1: Assess whether feature base table can be used to create the driver table
# MAGIC
# MAGIC     Part 2: Create the Driver table using feature base table 
# MAGIC

# COMMAND ----------

final_feat_list = rule_features

# COMMAND ----------

#replace start date and end date for analysis
start_date = pd.Timestamp('2024-08-20') #replace with your values
end_date = pd.Timestamp('2024-10-13') #replace with your values
par_region = 'AU' #replace this
checkpoint = 'CHECKOUT_CONFIRM' #replace this 
feature_base_driver_name='jobyg_fraud_segmentation_tool_notebook_demo' #replace this

# COMMAND ----------

# MAGIC %autoreload 2
# MAGIC from functions import create_feature_driver
# MAGIC
# MAGIC return_dict = create_feature_driver(final_feat_list, USER_NAME, conn, start_date, end_date, 
# MAGIC                                     par_region= par_region,
# MAGIC                                     checkpoint=checkpoint, 
# MAGIC                                     feature_base_driver_name=feature_base_driver_name
# MAGIC                                     )

# COMMAND ----------

# MAGIC %autoreload 2
# MAGIC from functions import create_control_table
# MAGIC control_table_name = 'jobyg_fraud_segmentation_tool_notebook_demo_cg'
# MAGIC create_control_table(control_table_name, USER_NAME, start_date, end_date, checkpoint, conn)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC from functions import pull_decision_tree_driver
# MAGIC
# MAGIC
# MAGIC dtree_driver = pull_decision_tree_driver(
# MAGIC     feat_driver_table='jobyg_fraud_segmentation_tool_notebook_demo',
# MAGIC     control_token_table= 'jobyg_fraud_segmentation_tool_notebook_demo_cg',
# MAGIC     target_column='p2_d0',
# MAGIC     conn=conn
# MAGIC )

# COMMAND ----------

from functions import grab_features_and_logic_using_rule_name
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

dtree_driver['target_rule_flag'] = dtree_driver[rule_features].apply(executable_rule, axis=1)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # del prep_for_training
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
# MAGIC

# COMMAND ----------

#split train data

train, val = split_dataset(X)

#specify column to be used as target
label = 'target'
#specify target values
classes = [0,1]
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label=label)
val_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(val, label=label)




# COMMAND ----------

#hyperparameter tuning 
bootstrap_size_ratio = [.25,.5,.75,.85,.9] 
#bootstrapping ratio determines what percentage of RECORDS is used for training the decision trees,
#decision trees that are built on different sets of records, are more likely to be different

num_candidate_attributes_ratio = [.4,.55,.7, .8,.85,.925]
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

num_trees = [100,250,500,600]
#num trees determines how many trees are built as part of the random forest, a larger number of trees results in a better performing model, but takes longer to train


# Create a Random Search tuner with 60 trials and specified hp configuration.
tuner = tfdf.tuner.RandomSearch(num_trials=25)
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

# MAGIC %md 
# MAGIC
# MAGIC Optional you can specify the first split if you would like by segmenting the dataframe using business intuition. For example

# COMMAND ----------

#segment the data further
first_split_logic = 'whitepages_identity_network_score >=.8'
X2 = run_query(f'select * from X where {first_split_logic}')
#split train data

train, val = split_dataset(X2)

#specify column to be used as target
label = 'target'
#specify target values
classes = [0,1]
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label=label)
val_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(val, label=label)




# COMMAND ----------

# Create a Random Search tuner with 60 trials and specified hp configuration.
#hyperparameter tuning 
bootstrap_size_ratio = [.25,.5,.75,.85,.9] 

num_candidate_attributes_ratio = [.4,.55,.7, .8,.85,.925]

train_rows = train.shape[0]
min_examples_param_list = [train_rows//200, train_rows//100, train_rows//50,train_rows//25]

max_depth = [3,5,7] 

num_trees = [100,250,500,600]

tuner = tfdf.tuner.RandomSearch(num_trials=25)
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
#specifiying the first split ended up improving the algorithms performance

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=8)


# COMMAND ----------

from functions import dfs_all_paths
inspector = tuned_model.make_inspector()
tree = inspector.extract_tree(tree_idx=0)
all_paths = []
dfs_all_paths(tree.root, [], all_paths)
        

# COMMAND ----------

threshold = .7
from functions import get_case_when_statement
outputs_dict = get_case_when_statement(tuned_model, threshold=threshold) 




# COMMAND ----------

final_dict = flatten(outputs_dict)
for k,v in final_dict.items():
    v[0]= v[0].replace('[', '(')
    v[0] = v[0].replace(']', ')')
                   

# COMMAND ----------

def evaluate_rules(dtree_driver, rule_dict,threshold):
    from tqdm import tqdm
    from pandasql import sqldf

    run_query = lambda query: sqldf(query, globals())

    target_rule_ctrl_trxn_ct = dtree_driver.loc[dtree_driver['target_rule_flag']==1].target.value_counts().sum()
    target_rule_p2_d0 = dtree_driver.loc[dtree_driver['target_rule_flag']==1].p2_overdue_d0_local.sum()/dtree_driver.loc[dtree_driver['target_rule_flag']==1].p2_due_local.sum()
    print(f'target rule ctrl transaction count is {target_rule_ctrl_trxn_ct}, target rule control group p2_d0 is {target_rule_p2_d0}')
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



performance_dict = evaluate_rules(dtree_driver, final_dict, .05)

# COMMAND ----------

dtree_driver.loc[dtree_driver['target_rule_flag']==1]

# COMMAND ----------

performance_df = pd.DataFrame(performance_dict.values(), columns =['new_rule_ctrl_trxn_ct','new_rule_p2_d0','rule'])
performance_df.sort_values(by='new_rule_p2_d0', ascending=False).head(50) #these rules don't look very promising, lets redo this endeavor using additional features

# COMMAND ----------

new_feat_list = [
 'SP_ENTITY_LINKING_HOP0_TOT_ORDER_CNT_BY_MERCH_SIDE_EMAIL_H72_0'
 ,'SP_C_ONLINE_ORDR_ATTMPT_CREDIT_CARD_CNT_H12_0'
 ,'IN_FLIGHT_CARD_NAME_VS_PROFILE_NAME'
 ,'SP_C_ONLINE_DECL_TOPAZ_INSFFCNT_FUND_ORDR_CNT_H12_0'
 ,'SP_D_LINKING_HOP0_ORDER_ATTMPT_CNT_BY_DEVICE_ID_H1_0'
 ,'SP_C_PYMT_ATTMPT_CNT_H24_0'
 ,'SP_C_ORDER_ATTEMPT_CNT_D1'
 ,'BP_UDP_C_GRAPH_MODEL_SCORE'
 ,'SP_ADDRESS_LINKING_TOTAL_CONSUMER_CNT_BY_RAW_SHIPPING_HASH_D3_0'
 ,'BP_C_ALL_DEVICE_LINKING_CUST_CNT'
 ,'IN_FLIGHT_ORDER_AMOUNT'
]
lowercase_new_feat_list = [feat.lower() for feat in new_feat_list]

# COMMAND ----------


final_feat_list = rule_features + lowercase_new_feat_list

#dedupe the final feature list in case there are duplicates
#duplicates will result in a SQL error where the same column is created multiple times
from functions import dedupe_list
final_feat_list = dedupe_list(final_feat_list)


# COMMAND ----------

from functions import create_feature_driver
feature_base_driver_name='jobyg_fraud_segmentation_tool_notebook_demo_v2' #replace this
return_dict = create_feature_driver(final_feat_list, USER_NAME, conn, start_date, end_date, 
                                    par_region= par_region,
                                    checkpoint=checkpoint, 
                                    feature_base_driver_name=feature_base_driver_name,
                                    skip_assessment=True
                                    )

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

dtree_driver = pull_decision_tree_driver(
    feat_driver_table='jobyg_fraud_segmentation_tool_notebook_demo_v2',
    control_token_table= 'jobyg_fraud_segmentation_tool_notebook_demo_cg',
    target_column='p2_d0',
    conn=conn
)

# COMMAND ----------

dtree_driver, dtype_dict = get_datatypes(decision_tree_driver=dtree_driver)
dtree_driver['target_rule_flag'] = dtree_driver[rule_features].apply(executable_rule, axis=1)
X, dtree_driver   = prep_for_training(decision_tree_driver = dtree_driver,
                                      exclude_features=['days_since_first_order_date', 'target_rule_flag'],
                                      test_ratio=.25) 

# COMMAND ----------

first_split_logic = 'whitepages_identity_network_score >=.8'
X2 = run_query(f'select * from X where {first_split_logic}')
#split train data

train, val = split_dataset(X2)

#specify column to be used as target
label = 'target'
#specify target values
classes = [0,1]
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label=label)
val_ds  = tfdf.keras.pd_dataframe_to_tf_dataset(val, label=label)



# COMMAND ----------

# Create a Random Search tuner with 60 trials and specified hp configuration.
#hyperparameter tuning 
bootstrap_size_ratio = [.25,.5,.75,.85,.9] 

num_candidate_attributes_ratio = [.4,.55,.7, .8,.85,.925]

train_rows = train.shape[0]
min_examples_param_list = [train_rows//200, train_rows//100, train_rows//50,train_rows//25]

max_depth = [3,4] 

num_trees = [100,250,500,600]

tuner = tfdf.tuner.RandomSearch(num_trials=25)
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

inspector = tuned_model.make_inspector()
tree = inspector.extract_tree(tree_idx=0)
all_paths = []
dfs_all_paths(tree.root, [], all_paths)
threshold = .7
outputs_dict = get_case_when_statement(tuned_model, threshold=threshold) 
final_dict = flatten(outputs_dict)
for k,v in final_dict.items():
    v[0]= v[0].replace('[', '(')
    v[0] = v[0].replace(']', ')')
performance_dict = evaluate_rules(dtree_driver, final_dict, .05)

# COMMAND ----------

performance_df = pd.DataFrame(performance_dict.values(), columns =['new_rule_ctrl_trxn_ct','new_rule_p2_d0','rule'])
# performance_df.sort_values(by='new_rule_p2_d0', ascending=False).head(50) 
# performance_df.sort_values(by='new_rule_p2_d0', ascending=False).iloc[60:150]



# COMMAND ----------

performance_df.loc[performance_df.rule.str.contains('whitepages')].sort_values(by='new_rule_p2_d0', ascending=False)

# COMMAND ----------


