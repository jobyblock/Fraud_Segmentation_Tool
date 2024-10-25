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
# MAGIC # Creating the feature driver table
# MAGIC
# MAGIC
# MAGIC Doordash has seen a higher toxicity in recent weeks than normal, and an additional rule is needed to mitigate these losses. To do so, we must first create a driver table which consists of:
# MAGIC
# MAGIC     1. Identifying columns
# MAGIC     2. Features we will use to risk split the new rule
# MAGIC     3. Control Group Identification
# MAGIC     4. Loss metrics
# MAGIC
# MAGIC The functions `create_feature_driver` and `create_control_table` will take care of the above:
# MAGIC
# MAGIC We have to specify some inputs for these functions, namely:
# MAGIC
# MAGIC          1. the time range of the analysis, 
# MAGIC          2. the par region
# MAGIC          3. checkpoint
# MAGIC          4. risk splitting features (for create_feature_driver) 
# MAGIC Note, this function is designed to analyze the **new user** population, if the rule in question should examine tenured users, you will have to modify the underlying function.
# MAGIC
# MAGIC If you are subsetting the new user population to first orders, or a specific merchant, further filtering must be done with SQL.

# COMMAND ----------

from functions import create_feature_driver, create_control_table
#replace start date and end date for analysis
start_date = pd.Timestamp('2024-06-01') #replace with your values
end_date = pd.Timestamp('2024-10-05') #replace with your values
par_region = 'AU' #replace this
checkpoint = 'CHECKOUT_CONFIRM' #replace this 
feature_base_driver_name='jobyg_fraud_segmentation_tool_notebook_usecase1_demo' #replace this

feature_list = ['in_flight_order_merchant_id'
             ,'in_flight_order_merchant_name'
             ,'in_flight_order_amount'
             ,'consumer_contact_address_postcode'
             ,'sp_entity_linking_hop0_tot_order_cnt_by_merch_side_email_h72_0'
             ,'sp_c_fraud_decline_attempt_d3_0'
             ,'sp_c_fraud_decline_attempt_h12_0'
             ,'sp_c_fraud_decline_attempt_h1_0'
             ,'sp_c_online_ordr_attmpt_credit_card_cnt_h12_0'
             ,'in_flight_card_name_vs_profile_name'
             ,'sp_c_online_decl_topaz_insffcnt_fund_ordr_cnt_h12_0'
             ,'sp_c_online_decl_topaz_insffcnt_fund_ordr_cnt_h168_0'
             ,'sp_d_linking_hop0_order_attmpt_cnt_by_device_id_h1_0'
             ,'sp_c_pymt_attmpt_cnt_h24_0'
             ,'sp_c_order_attempt_cnt_d1'
             ,'bp_udp_c_graph_model_score'
             ,'inflight_device_id_consumer_distinct_cnt'
             ,'sp_address_linking_total_consumer_cnt_by_raw_shipping_hash_d3_0'
             ,'bp_c_all_device_linking_cust_cnt'
             ,'bp_c_max_device_linking_new_cust_cnt'
             ,'consumer_account_linking_type'
             ,'bp_c_batch_consumer_batch_model_v1'
             ,'model_online_od_payback_non_us_april_2024_score'
             ,'consumer_active_order_number'
             ,'bp_card_issuing_bank_new_p2d0'
             ,'tmx_digital_id_confidence'
             ,'bp_profile_email_domain_new_matured_ntl_rate'
             ,'derived_minutes_since_account_created'
             ,'tmx_smart_learning_fraud_rating'
             ,'whitepages_primary_address_checks_is_commercial'
             ,'bp_c_acct_cnt_ab_od'
             ,'bp_c_seed_based_linking_device_id'
             ,'sp_c_order_amt_same_merchant_as_current_h24_0'
             ,'sp_c_order_amt_same_merchant_as_current_h12_0'
             ,'sp_c_order_amt_same_merchant_as_current_h1_0'
             ,'whitepages_identity_check_score'
             ,'model_gibberish_consumer_profile_email_august_2022_score'
             ,'in_flight_card_name_vs_profile_name'
             ,'whitepages_identity_network_score'
             ,'sp_c_order_attempt_cnt_d3'
             ,'whitepages_primary_email_address_checks_email_first_seen_days'
             ,'whitepages_primary_phone_checks_match_to_name'
             ,'whitepages_primary_address_checks_match_to_name'
             ,'bp_c_outstanding_balance_avg_amt_30d_v2'
             ,'bp_c_seed_cnt_linked_by_device_id'
             ,'bp_c_seed_cnt_linked_by_raw_shipping_address'
             ,'bp_c_trusted_merch_side_email_yn'
             ,'consumer_is_first_order'
]



# COMMAND ----------

return_dict = create_feature_driver(feat_list=feature_list, username=USER_NAME, 
                                    conn=conn, 
                                    start_date=start_date, 
                                    end_date=end_date, 
                                    par_region= par_region,
                                    checkpoint=checkpoint, 
                                    feature_base_driver_name=feature_base_driver_name,
                                    skip_assessment = True #speeds up query as I have manually confirmed all the features exist in my authorized view and thus the query won't error out
                                    )

# COMMAND ----------

from functions import create_control_table
control_table_name = 'jobyg_fraud_segmentation_tool_notebook_usecase1_demo_cg'
create_control_table(control_table_name, USER_NAME, start_date, end_date, checkpoint, conn)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create the pandas dataframe for the decision tree by pulling data from SF using SQL

# COMMAND ----------

from functions import pull_decision_tree_driver

dtree_driver = pull_decision_tree_driver(
    feat_driver_table=feature_base_driver_name,
    control_token_table= control_table_name,
    target_column='p2_d0',
    conn=conn
)

# COMMAND ----------

dtree_driver.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Specify Doordash new user online orders in the driver
# MAGIC
# MAGIC Since this will be a rule designed for Doordash, we must filter the above decision tree driver to just doordash orders, we can do so with the following:

# COMMAND ----------

doordash_driver = run_query('select * from dtree_driver where in_flight_order_merchant_id = "134317"')

# COMMAND ----------

#convert datatypes to appropriate values, otherwise we get TypeErrors
from functions import get_datatypes #note, if a feature is not behaving as anticipated, most commonly a string input will be treated as an integer, go to the functions script and modify the get_datatypes function to assign the feature it's appropriate type

doordash_driver, dtype_dict = get_datatypes(decision_tree_driver=doordash_driver)

# COMMAND ----------

doordash_driver.head()

# COMMAND ----------

from functions import prep_for_training


#note, i added columns to the exclude feature list if i found the decision tree's splitting
#to be too arbitrary and unexplainable and overfitting certain string variables
X, doordash_driver   = prep_for_training(decision_tree_driver = doordash_driver,
                                      exclude_features=['in_flight_order_merchant_name',
                                                        'in_flight_order_merchant_id',
                                                        'days_since_first_order_date',
                                                        'derived_minutes_since_account_created',
                                                        'consumer_contact_address_postcode'],
                                      test_ratio=.25) 

from functions import split_dataset



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
min_examples_param_list = [train_rows//100, train_rows//50,train_rows//25, train_rows//10]
#min_examples_param_list determines the minimum number of records that must be present when creating segments
#for example a min_examples hyper parameter of 100 means no final segment will contain less than 100 records
#a higher value of this parameter results in less over-fit segments
 
max_depth = [3,4,5] 
#max depth determines how many times the tree can split, a higher value of this parameter can result in over-fitting
#as the tree will be able to form precise segments that only perform well on the training data through use of excessive splitting

num_trees = [100,250,500,600]
#num trees determines how many trees are built as part of the random forest, a larger number of trees results in a better performing model, but takes longer to train


# Create a Random Search tuner with 50 trials and specified hp configuration.
tuner = tfdf.tuner.RandomSearch(num_trials=50)
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

#examine results of the hyper-parameter tuning, notice the difference between the highest and average scores
logs = tuned_model.make_inspector().tuning_logs() #assess tuning logs
num_trees =logs[logs.best==True]['num_trees'].iloc[0]
logs


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Optional you can specify the first split if you would like by segmenting the dataframe using business intuition. For example

# COMMAND ----------

tfdf.model_plotter.plot_model_in_colab(tuned_model, tree_idx=0, max_depth=8)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Find high potential segments
# MAGIC
# MAGIC The next function iterates over each tree created and identifies the logic used to get to each terminal node across all trees.

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

# MAGIC %md 
# MAGIC
# MAGIC # Assess which rules are valuable

# COMMAND ----------

def evaluate_rules(df, string_df, rule_dict, threshold=0, target_rule_present=False):
    from tqdm import tqdm
    from pandasql import sqldf

    run_query = lambda query: sqldf(query, globals())
    if target_rule_present:
        target_rule_ctrl_trxn_ct = df.loc[df['target_rule_flag']==1].target.value_counts().sum()
        target_rule_p2_d0 = df.loc[df['target_rule_flag']==1].p2_overdue_d0_local.sum()/df.loc[df['target_rule_flag']==1].p2_due_local.sum()
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
            
    query += f' from {string_df}'
    test_df = run_query(query)
    print('evaluating first third of the new rules as sql can only display a limited number of columns')
    print('there will be three progress bars as a part of this function, this is progress bar 1')
    for i in tqdm(range(0, first_third)):
        key = key_list[i]
        v = rule_dict[key]
        segment_name = v[0].split(' as ')[1]
        new_rule_p2_d0 = test_df.loc[test_df[segment_name]==1].p2_overdue_d0_local.sum()/test_df.loc[test_df[segment_name]==1].p2_due_local.sum()
        
        if target_rule_present:
            if new_rule_p2_d0 - target_rule_p2_d0 > threshold:
                new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
                performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]
        elif new_rule_p2_d0 >= .35:
            new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
            performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]

    query = 'select *'
    for i in range(first_third, second_third):
        key = key_list[i]
        val = rule_dict[key]
        query+=f',{val[0]}'
            
    query += f' from {string_df}'
    test_df = run_query(query)
    print('evaluating second third of the new rules as sql can only display a limited number of columns')
    print('there will be three progress bars as a part of this function, this is progress bar 2')
    for i in tqdm(range(first_third, second_third)):
        key = key_list[i]
        v = rule_dict[key]
        segment_name = v[0].split(' as ')[1]
        new_rule_p2_d0 = test_df.loc[test_df[segment_name]==1].p2_overdue_d0_local.sum()/test_df.loc[test_df[segment_name]==1].p2_due_local.sum()
        if target_rule_present:
            if new_rule_p2_d0 - target_rule_p2_d0 > threshold:
                new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
                performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]
        elif new_rule_p2_d0 >= .35:
            new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
            performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]
    query = 'select *'
    for i in range(second_third, last_third):
        key = key_list[i]
        val = rule_dict[key]
        query+=f',{val[0]}'
            
    query += f' from {string_df}'
    test_df = run_query(query)
    print('evaluating last third of the new rules as sql can only display a limited number of columns')
    print('there will be three progress bars as a part of this function, this is progress bar 3')
    for i in tqdm(range(second_third, last_third)):
        key = key_list[i]
        v = rule_dict[key]
        segment_name = v[0].split(' as ')[1]
        new_rule_p2_d0 = test_df.loc[test_df[segment_name]==1].p2_overdue_d0_local.sum()/test_df.loc[test_df[segment_name]==1].p2_due_local.sum()
        if target_rule_present:
            if new_rule_p2_d0 - target_rule_p2_d0 > threshold:
                new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
                performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]
        elif new_rule_p2_d0 >= .35:
            new_rule_ctrl_trxn_ct = test_df.loc[test_df[segment_name]==1].target.value_counts().sum()
            performance_dict[segment_name] = [new_rule_ctrl_trxn_ct,new_rule_p2_d0, v[0]]

    
    return(performance_dict)



performance_dict = evaluate_rules(doordash_driver, 'doordash_driver', final_dict, .05)

# COMMAND ----------

performance_df = pd.DataFrame(performance_dict.values(), columns =['new_rule_ctrl_trxn_ct','new_rule_p2_d0','rule'])
performance_df.sort_values(by='new_rule_p2_d0', ascending=False).head(50) 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Analyze performance splitting power

# COMMAND ----------

rule_to_author = performance_df.iloc[169].rule
rule_to_author

# COMMAND ----------

rule_to_author = performance_df.iloc[261].rule

# COMMAND ----------

rule_to_author

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### rename segment name to something more user friendly

# COMMAND ----------

from functions import modify_segment_name

# COMMAND ----------

rule_to_author = modify_segment_name(rule_to_author, 'test_segment')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # create driver tables for important metrics 
# MAGIC
# MAGIC Such as unique declines (# and $), total control group p2 d0, etc.

# COMMAND ----------

# MAGIC %autoreload 2
# MAGIC unique_rule_decline_table_name = 'jobyg_fraud_segmentation_tool_demo_unique_declines'
# MAGIC from functions import create_unique_decline_table
# MAGIC create_unique_decline_table(unique_rule_decline_table_name, 
# MAGIC USER_NAME,
# MAGIC start_date,
# MAGIC end_date,
# MAGIC par_region,
# MAGIC checkpoint,
# MAGIC conn)
# MAGIC
# MAGIC

# COMMAND ----------

from functions import get_decline_rate_denom

decline_rate_denoms = get_decline_rate_denom(start_date, end_date, par_region, checkpoint, USER_NAME, conn)

decline_rate_denom_ct = int(decline_rate_denoms.token_ct.values)
decline_rate_denom_amt = int(decline_rate_denoms.order_amount.values)

# COMMAND ----------

coverage_denom = dtree_driver.p2_overdue_d0_local.sum() #dtree driver is the inner join between attempt control group and all transaction attempts, summing the overdue column gives us the total new user control group p2_overdue for our respective par region and checkpoint


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #Create driver table for KPI analysis

# COMMAND ----------


new_rule_table_name = 'jobyg_new_fraud_segmentation_tool_notebook_usecase1_demo'
grab_new_rule_performance(new_rule_table_name=  new_rule_table_name,
                          unique_decline_table_name= unique_rule_decline_table_name,
                          rule = rule_to_author,
                          start_date=start_date,
                          end_date = end_date,
                          par_region = par_region,
                          checkpoint = checkpoint,
                          user_name = USER_NAME,
                          conn=conn)

#additional query to specify the merchant id for my tablename
conn.execute(f'''create or replace table ap_cur_frdrisk_g.public.{new_rule_table_name} as (
            select * from ap_cur_frdrisk_g.public.{new_rule_table_name} where in_flight_order_merchant_id = '134317')''')

# COMMAND ----------

#ensure order token has an order amount
from functions import order_amount_fixing
order_amount_fixing(new_rule_table_name, conn)

# COMMAND ----------

#ensure data is exclusively doordash 
validation = conn.download(f'''select a.*, coalesce(b.consumer_total_amount_amount, in_flight_order_amount) as order_amount_local from ap_cur_Frdrisk_g.public.{new_rule_table_name}  a
                           left join ap_cur_frdrisk_g.public.order_amt_fixed b
                           on a.order_token = b.token
                           and a.in_flight_order_merchant_id = '134317'
                           ''')
validation.in_flight_order_merchant_id.value_counts()


# COMMAND ----------

validation.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #Calculate KPI

# COMMAND ----------

analyze_performance('validation','test_segment',decline_rate_denom_ct,decline_rate_denom_amt, coverage_denom)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Expand upon the initial rule
# MAGIC
# MAGIC I wanted to see if I can improve the performance of this rule by adding another commonly seen risk splitter, `(sp_c_order_attempt_cnt_d1 >= 3.5)`. The functions created make reassessing the performance of a slightly modified rule -- quick.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 1: Modify rule logic

# COMMAND ----------

final_rule = 'CASE WHEN (model_online_od_payback_non_us_april_2024_score >= 150.0173797607422) and (in_flight_order_amount >= 13.805000305175781) and (whitepages_identity_check_score >= 302.0) and (whitepages_identity_network_score >= 0.781499981880188) and (sp_c_order_attempt_cnt_d1 >= 3.5) THEN 1 ELSE 0  END as test_segment'

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 2: Create new rule performance driver table

# COMMAND ----------

# MAGIC %autoreload 2
# MAGIC grab_new_rule_performance(new_rule_table_name=  new_rule_table_name,
# MAGIC                           unique_decline_table_name= unique_rule_decline_table_name,
# MAGIC                           rule = final_rule,
# MAGIC                           start_date=start_date,
# MAGIC                           end_date = end_date,
# MAGIC                           par_region = par_region,
# MAGIC                           checkpoint = checkpoint,
# MAGIC                           user_name = USER_NAME,
# MAGIC                           conn=conn)
# MAGIC
# MAGIC #additional query to specify the merchant id for my 
# MAGIC conn.execute(f'''create or replace table ap_cur_frdrisk_g.public.{new_rule_table_name} as (
# MAGIC             select * from ap_cur_frdrisk_g.public.{new_rule_table_name} where in_flight_order_merchant_id = '134317')''')
# MAGIC
# MAGIC #fix order amount to not have nulls
# MAGIC from functions import order_amount_fixing
# MAGIC analysis_df = order_amount_fixing(new_rule_table_name, conn)
# MAGIC
# MAGIC #additional query to specify mercahnt id for my table 
# MAGIC analysis_df = run_query('''select * from analysis_df where in_flight_order_merchant_id = '134317' ''')
# MAGIC analysis_df.in_flight_order_merchant_id.value_counts()
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Final Step: Analyze performance of new rule

# COMMAND ----------

analyze_performance('analysis_df','test_segment',decline_rate_denom_ct,decline_rate_denom_amt, coverage_denom) #worse unique KPI 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Still wasn't that pleased with performance
# MAGIC
# MAGIC Let's try one additional modification focusing on `sp_c_order_amt_same_merchant_as_current_h24_0`, increasing the threshold from 13.8 to 21

# COMMAND ----------

final_rule2 = 'CASE WHEN (model_online_od_payback_non_us_april_2024_score >= 150.0173797607422) and (in_flight_order_amount >= 13.80) and (whitepages_identity_check_score >= 280.0) and (whitepages_identity_network_score >= 0.761499981880188) and (sp_c_order_amt_same_merchant_as_current_h24_0 >= 13.575) THEN 1 ELSE 0  END as test_segment'

# COMMAND ----------

grab_new_rule_performance(new_rule_table_name =  new_rule_table_name,
                          unique_decline_table_name= unique_rule_decline_table_name,
                          rule = final_rule2,
                          start_date=start_date,
                          end_date = end_date,
                          par_region = par_region,
                          checkpoint = checkpoint,
                          user_name = USER_NAME,
                          conn=conn)

#additional query to specify the merchant id for my 
conn.execute(f'''create or replace table ap_cur_frdrisk_g.public.{new_rule_table_name} as (
            select * from ap_cur_frdrisk_g.public.{new_rule_table_name} where in_flight_order_merchant_id = '134317')''')

#fix order amount to not have nulls
from functions import order_amount_fixing
analysis_df = order_amount_fixing(new_rule_table_name, conn)

#additional query to specify mercahnt id for my table 
analysis_df = run_query('''select * from analysis_df where in_flight_order_merchant_id = '134317' ''')
analysis_df.in_flight_order_merchant_id.value_counts()



# COMMAND ----------

analyze_performance('analysis_df','test_segment',decline_rate_denom_ct,decline_rate_denom_amt, coverage_denom) #rule performance looks promising

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Export rule to python

# COMMAND ----------

# MAGIC %autoreload 2
# MAGIC from functions import create_py_rule
# MAGIC final_rule2 = create_py_rule(final_rule2,  rule_name='AU_Online_Doordash_rule.py', path_name ='/Workspace/Users/jobyg@squareup.com/Fraud Segmentation Tool /',debug=False)
# MAGIC
# MAGIC

# COMMAND ----------

|from functions import create_historical_decline_table, create_unique_decline_table
rule_decline_table_name = 'jobyg_'+target_rule+'_declines'
unique_rule_decline_table_name = 'jobyg_'+target_rule+'_unique_declines'


print(rule_decline_table_name, unique_rule_decline_table_name)

create_historical_decline_table(rule_decline_table_name=rule_decline_table_name,
rule_list=[target_rule],
user_name=USER_NAME,
start_date=start_date,
end_date=end_date,
par_region=par_region,
checkpoint=checkpoint,
conn=conn)


# COMMAND ----------



# COMMAND ----------




# COMMAND ----------

decline_rate_denoms

# COMMAND ----------

#now i need a new rule to work with to show performance improvement
#chose a rule and identify those transactions in the 
rule_to_author = performance_df.iloc[24].rule

# COMMAND ----------

def modify_segment_name(rule, new_segment_name):
    rule = rule.split('END as')
    rule.insert(1, f'END as {new_segment_name}')
    rule = rule[:2]
    rule = ' '.join(rule)
    print(rule)
    return(rule)

# COMMAND ----------

def add_split_condition_to_case_when(rule, split_condition, new_segment_name):
    rule = rule_to_author.split('CASE WHEN')
    rule.insert(1, split_condition + ' and')
    rule.insert(0, 'CASE WHEN')
    rule = ' '.join(rule)
    rule = rule.split('END as')
    rule.insert(1, f'END as {new_segment_name}')
    rule = rule[:2]
    rule = ' '.join(rule)
    print(rule)
    return(rule)

# COMMAND ----------

rule_to_author = add_split_condition_to_case_when(rule_to_author, first_split_logic, 'test_segment')

# COMMAND ----------

rule_to_author

# COMMAND ----------

def grab_new_rule_performance(new_rule_table_name, 
                              rule,
                              start_date,
                              end_date,
                              par_region,
                              checkpoint, 
                              user_name,
                             conn):
    query =     f'''create or replace table ap_cur_frdrisk_g.public.{new_rule_table_name} as (
    select 
        a.order_token,
        a.par_region,
        a.checkpoint,
        a. par_process_date,
        {rule},
        is_in_attempt_control_group,
        case when is_in_attempt_control_group = 1 then P2_OVERDUE_D0_LOCAL end as ctrl_p2_d0,
        case when is_in_attempt_control_group = 1 then P2_due_LOCAL end as ctrl_p2_due,
        from AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{user_name}_DSL3_SV a
        left join AP_CUR_RISKBI_G.CURATED_RISK_BI_GREEN.DWM_ORDER_LOSS_TAGGING  d
        on a.order_token = d.order_token
        where checkout_time between '{start_date}' and '{end_date}' and a.par_region = '{par_region}' and a.checkpoint = '{checkpoint}'
        and coalesce(a.days_since_first_order_date,0) < 15);'''
    conn.execute(query)



# COMMAND ----------

new_rule_table = 'jobyg_test_new_rule_performance'
query = grab_new_rule_performance(new_rule_table, rule_to_author, start_date, end_date, par_region, checkpoint, USER_NAME, conn)

# COMMAND ----------

data = conn.download('''select

                            count(distinct(case when test_segment = 1 then order_token end)) 
                            as total_decline_vol,
                            count(distinct(case when is_in_attempt_control_group =1 and test_segment = 1 then order_token end)) as ctrl_trxn_ct,
                            sum(case when test_segment = 1 then ctrl_p2_d0 end) ctrl_p2_d0,
                            sum(case when test_segment = 1 then ctrl_p2_due end) ctrl_p2_due,
                            sum(case when test_segment = 1 then ctrl_p2_d0 end)/sum(case when test_segment = 1 then ctrl_p2_due end) as ctrl_p2_d0_rate
                            from ap_cur_frdrisk_g.public.jobyg_test_new_rule_performance
                            ;''')


# COMMAND ----------

data

# COMMAND ----------


