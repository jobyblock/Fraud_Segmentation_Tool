def grab_features_and_logic_using_rule_name(target_rule,conn):
    """
    target_rule (str): the rule that you are interested in analyzing and improving, must be a string type
    conn (Session): sql connection that is necessary to query snowflake tables 
    returns:
        rule features (list): features for the rule, sourced from AP_RAW_GREEN.GREEN.RAW_C_E_RULESENGINEKARMA_RULE_CONTENT_CHANGE
        rule_content (str): string of the rule logic that will be used to create an executable function 
    """
    def grab_rule_and_logic(target_rule, conn):
    #download features
        df = conn.download(f''' SELECT
                            a.rule_name
                            ,a.last_modified_date
                            ,a.rule_features 
                            ,a.rule_content
                            FROM AP_RAW_GREEN.GREEN.RAW_C_E_RULESENGINEKARMA_RULE_CONTENT_CHANGE a
                            INNER JOIN (SELECT rule_name
                                                ,max(last_modified_date) as last_modified_date 
                                            FROM AP_RAW_GREEN.GREEN.RAW_C_E_RULESENGINEKARMA_RULE_CONTENT_CHANGE WHERE rule_name = '{target_rule}' 
                                            GROUP BY 1) b
                                            ON a.rule_name = b.rule_name
                                            AND a.last_modified_date = b.last_modified_date                   
                                            WHERE a.rule_name = '{target_rule}';''')
        #format output to get a list of strings that correspond to feature names
        rule_features = df.rule_features.iloc[0]
        rule_features = rule_features.replace("'", '')
        rule_features = rule_features.replace('"', '')
        rule_features = rule_features.replace('[', '')
        rule_features = rule_features.replace(']', '')
        rule_features = rule_features.replace(' ', '')
        rule_features = rule_features.split(',')

        #grab rule content
        rule_content = df.rule_content.iloc[0]

        #return feature_list and rule content
        return(rule_features,rule_content)
    
    rule_features,rule_content = grab_rule_and_logic(target_rule, conn)
    def format_code(code_str):
        """
        code_str (str): string that represents functions we want to run, in order to run,   the code must be properly linted with appropriate indention
        returns code_str (str): code string thats been linted
        """
        import black

        formatted_code = black.format_str(code_str, mode=black.FileMode())

        return formatted_code
    
    rule_content = format_code(rule_content)
    print(rule_content)
    return(rule_features,rule_content)

def format_code(code_str):

        import black

        formatted_code = black.format_str(code_str, mode=black.FileMode())

        return formatted_code

def format_rule(rule,features):
    """
    rule (string): rule logic in string format, needs to be meet python's expectation before it will work as an executable function
    features (list): list of features used in the rule
    returns executable_rule (string): a string version of a function that matches python's expectations and will run 
    """
    import re
    #function to make slight changes to the function used in the rule to return 1 if a token is impacted by
   
    #pass a row parameter into the arguement so it can run on the dataframe
    rule = rule.replace('def execute_rule():', 'def execute_rule(row):')
    
    for feat in features:
        pattern = rf"(?<!\w){feat}(?!\w)"
        if re.search(pattern, rule):
            rule=rule.replace(feat, 'row["'+feat+'"]')
    #change return condition when the if criteria is met to be 1
    rule = rule.replace('actions.append({"action_name": "is_rejected_assign"})','return(1)')
    #change return condition when the if criteria is not met to be 0
    rule = rule.replace('return actions','return(0)')
    executable_rule = rule
    return(executable_rule)



def grab_rule_content_for_rule(target_rule,conn):
    #download features
    rule_content = conn.download(f''' SELECT
                        a.rule_name
                        ,a.last_modified_date
                        ,a.rule_content 
                        FROM AP_RAW_GREEN.GREEN.RAW_C_E_RULESENGINEKARMA_RULE_CONTENT_CHANGE a
                        INNER JOIN (SELECT rule_name
                                            ,max(last_modified_date) as last_modified_date 
                                        FROM AP_RAW_GREEN.GREEN.RAW_C_E_RULESENGINEKARMA_RULE_CONTENT_CHANGE WHERE rule_name = '{target_rule}' 
                                        GROUP BY 1) b
                                        ON a.rule_name = b.rule_name
                                        AND a.last_modified_date = b.last_modified_date                   
                                        WHERE a.rule_name = '{target_rule}';''')
    return(rule_content)

def dedupe_list(inp_list):
    """
    input list (list): list of features
    returns input list (list) list of features with no duplicates, necessary to prevent creating the same column multiple
    times in future sql pulls
    """
    inp_list = set(inp_list)
    inp_list = list(inp_list)
    return(inp_list)

def create_feature_table_using_rule_name(feat_table_name, target_rule, conn, user_name, start_date=None, end_date =None, debug_mode=True):
    """
    feat_table_name (string): name that will be used to create a table in the  AP_CUR_FRDRISK_G.public database and schema
    feature_list (list): the deduped list of features used in the rule, plus any additional features of interest, this will be used
    to query AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.TBL_RAW_C_E_FC_DECISION_RECORD_RULE_VARS
    conn (Session): sql connection needed to query the database
    par_region (str): par region that will be filtered to
    checkpoint (str): checkpoint that will be filtered to
    user_name (string): user's ldap, necessary to query a particular authorized view
    start_date (string): start date of the analytical window
    end_date (string): end date of the analytical window
    debug_mode (True/False): A parameter that if set to true will not run the query, but will print it, useful for debugging as this function can take hours
    """
    # print(feature_set)
    #sp_c_online_order_count_h48_0
    query_intro = f'''create or replace table AP_CUR_FRDRISK_G.public.{feat_table_name} as (
        select 
            order_token
            ,key_event_info_id
            ,event_info_event_time
            ,checkpoint
            ,par_region
            ,consumer_id
            ,par_process_date
            '''
    feature_query = ''
    for i in range(len(feature_list)):
        feature = feature_list[i]
        feature_query +=f''',max(case when var_name ilike '{feature}' then var_value end) as {feature}
        '''
    query_outro = f''' from AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.TBL_RAW_C_E_FC_DECISION_RECORD_RULE_VARS__{user_name}_DSL3_SV 
       where 1=1 
    '''
    if start_date and end_date:
        query_outro += f''' and par_process_date between '{start_date}' and '{end_date}'
        '''
    
    query_outro+= ")"


    query_outro += ''' group by 1,2,3,4,5,6,7 );''' 
    query_f = query_intro + feature_query + query_outro
    if debug_mode:
        print(query_f)
    else:
        conn.execute(query_f)
    return()


def create_historical_decline_table(table_name, rule_list, start_date, end_date,  conn, ):
    """
    rule_decilne_table_name (string): name of the table to hold the historical decline data for the given rules
    rule_list (list): list of rules that are being analyzed
    start_date (string): start date of the analytical window
    end_date (string): end date of the analytical window
    par_region (string): par region that the query will filter to
    checkpoint (string): checkpoint that the query will filter to
    conn (Session): sql connection that is necessary to query snowflake tables 
    returns: nothing, but creates the dataframe ap_cur_frdrisk_g.public.{rule_decline_table_name}
    """    
    sql = f"""CREATE or replace table ap_cur_frdrisk_g.public.{table_name} AS (SELECT 
        a.key_Event_info_id, EVENT_INFO_EVENT_TIME, rule_id, par_region, checkpoint,
        FROM  AP_CUR_R_FEATSCI.curated_feature_science_red.TBL_RAW_C_E_FC_DECISION_RECORD_RULES__jobyg_DSL3_SV a
        WHERE 1=1 
        AND ((is_rejected = 'True' and is_in_treatment = 'True')
        AND a.par_process_date between '{start_date}' and '{end_date}'
        and rule_id in (select distinct """
    for i in range(len(rule_list)):
        rule = rule_list[i]
        if i != len(rule_list)-1:
            sql+=f"'{rule}', "
        else:
            sql+=f"'rule')"
    sql+=");"
    print(sql)
    conn.execute(sql)

def find_additional_relevant_features(feature_list, conn, username):
    """
    feat_list (list): deduped list of features used in the rule, plus any additional features of interest
    username (string): user ldap, necessary to query a user's authorized view
    conn (Session): sql connection that is necessary to query snowflake tables 
    returns feat_df (pandas dataframe): dataframe of additional features that matches the strings specified in the entries
    of features_list_of_interest
    """
    import pandas as pd

    today = pd.Timestamp.now().normalize()
    one_week_ago = (today - pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
    query = f'''select distinct var_name from AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.TBL_RAW_C_E_FC_DECISION_RECORD_RULE_VARS__{username}_DSL3_SV  where par_process_Date = '{one_week_ago}';'''
    print(query)
    additional_features = conn.download(query)
    temp_df_list = []
    for feat in feature_list:
        a= additional_features.var_name.str.contains(feat)
        temp_df_list.append(additional_features.loc[a])
    feat_df = pd.concat(temp_df_list)
    return(feat_df)

def create_feature_driver(feat_list, username, conn, start_date, end_date, par_region, checkpoint, feature_base_driver_name, skip_assessment = False):
    import pandas as pd
    from datetime import datetime, timedelta
    import calendar

    today = pd.Timestamp.now().normalize()
    one_week_ago = (today - pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
    test = None


    try:
        test = conn.download(f'''select * from AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{username}_DSL3_SV where par_process_date = '{one_week_ago}' limit 1''')
        if test is not None:
            print(f'User has access to AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{username}_DSL3_SV')
            print('assessing which features are in the feature base table')
    except Exception as e:
        print(f'User does not have access to AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{username}_DSL3_SV')
        print('for documentation on how to get access to the table, visit this link https://docs.google.com/document/d/1aEWQpkmSqLSpN_VjbNEMOvqr7g-0beiYF7x8XaUROUY/edit#heading=h.oba2foqtr1yx')
        return
    return_dict = {'in_feature_base':[],
                   'not_in_feature_base':[]}
    if skip_assessment == False:
        for feat in feat_list:
            test = None
            try:
                test = conn.download(f'''select {feat} from AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{username}_DSL3_SV where par_process_date = '{one_week_ago}' limit 1''')
                if test is not None:
                    print(f'{feat} is in the UNIFIED_FEATURE_DATAMART_BASE table')
                    return_dict['in_feature_base'].append(feat)
            except Exception as e:
                print(f"{feat} is not in the UNIFIED_FEATURE_DATAMART_BASE table")
                return_dict['not_in_feature_base'].append(feat)
                continue
            
        if len(return_dict['not_in_feature_base']) == 0:
            print(f'all variables are in AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{username}_DSL3_SV')
    else:
        print('skipping assessment of features in the feature base table')
    print('creating feature driver table')
    feat_set = set(feat_list)
    feat_set.add('order_token')
    feat_set.add('checkout_time')
    feat_set.add('checkpoint')
    feat_set.add('par_region')
    feat_set.add('consumer_id')
    feat_set.add('par_process_date')
    feat_set.add('days_since_first_order_date')


    create_statement = f'''create or replace table AP_CUR_FRDRISK_G.public.{feature_base_driver_name}'''
    query_intro=f''' as (
        select 
        '''
    feature_query = ''
    final_feat_list = list(feat_set)
    for i in range(len(final_feat_list)):
        feature = final_feat_list[i]
        if i == 0:
            feature_query += f'''{feature}'''
        else:
            feature_query +=f''',{feature}'''
    feature_query += f''' from AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{username}_DSL3_SV
       where 1=1 and coalesce(days_since_first_order_date,0) < 15'''
    query_outro = f''' and par_process_date >= '{start_date}' and par_process_date <= '{end_date}' and par_region ilike '{par_region}' and checkpoint ilike '{checkpoint}' '''
    query_outro += ''');''' 
    query_f = create_statement + query_intro + feature_query + query_outro
    print(query_f)
    conn.execute(query_f)
    print(f'sucessfully created table AP_CUR_FRDRISK_G.public.{feature_base_driver_name}')
    return()

def create_historical_decline_table(rule_decline_table_name, 
                                    
                                    rule_list,
                                    user_name, 
                                    start_date, 
                                    end_date,  
                                    par_region,
                                    checkpoint, 
                                    conn,
                                    need_decline_rate_denom=False):
        """
        rule_decilne_table_name (string): name of the table to hold the historical decline data for the given rules
        rule_list (list): list of rules that are being analyzed
        start_date (string): start date of the analytical window
        end_date (string): end date of the analytical window
        par_region (string): par region that the query will filter to
        checkpoint (string): checkpoint that the query will filter to
        conn (Session): sql connection that is necessary to query snowflake tables 
        returns: nothing, but creates the dataframe ap_cur_frdrisk_g.public.{rule_decline_table_name}
        """    
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

def create_unique_decline_table(unique_decline_table_name, user_name, start_date, end_date, par_region, checkpoint, conn): 
    """
    unique_decline_table_name (string): name of the table to hold the historical unique decline data
    start_date (string): start date of the analytical window
    end_date (string): end date of the analytical window
    par_region (string): par region that the query will filter to
    checkpoint (string): checkpoint that the query will filter to
    user_name (string): user ldap, necessary to query authorized views
    conn (Session): sql connection that is necessary to query snowflake tables 
    returns: nothing, but creates the dataframe ap_cur_frdrisk_g.public.{unique_decline_table_name}
    """    
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
    print(unique_decline_driver)
    conn.execute(unique_decline_driver)
    print(f'finished creating the unique decline driver')

def get_decline_rate_denom(start_date, end_date, par_region, checkpoint, user_name, conn):
    
    decline_rate_denom_q = f'''
        select checkpoint,
               par_region, 
               count(distinct(order_token)) as token_ct,
               sum(consumer_order_amount) as order_amount
               from  (
                   select 
                    checkpoint,
                    par_region,
                    order_token,
                    consumer_order_amount,
                    event_info_event_time,
                    ROW_NUMBER() OVER (PARTITION BY order_token ORDER BY event_info_event_time DESC) AS rn
                  from AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.TBL_RAW_C_E_FC_DECISION_RECORD__{user_name}_DSL3_SV 
                   where checkpoint ilike '%{checkpoint}%' and
                    par_region ilike '%{par_region}%'
                     and par_process_date between '{start_date}' and '{end_date}' 
                    and days_since_first_order_date < 15
               )
               where rn = 1
               group by 1,2
               order by 1,2
               '''
    print(decline_rate_denom_q)
    decline_rate = conn.download(decline_rate_denom_q)
    print(f'decline rate denom is {decline_rate}')
    return(decline_rate)


def create_control_table(control_table_name, user_name, start_date, end_date, checkpoint, conn):
    control_query = f"""CREATE or replace table AP_CUR_FRDRISK_G.public.{control_table_name} as (
    select a.order_token, a.is_in_attempt_control_Group,
        d.p2_overdue_d0_local,
        d.p2_due_local,
        d.ntl_forecast_local,
        from AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{user_name}_DSL3_SV a
    left join  AP_CUR_RISKBI_G.CURATED_RISK_BI_GREEN.DWM_ORDER_LOSS_TAGGING  d
    on a.order_token = d.order_token
    and to_varchar(a.checkout_Time, 'YYYY-MM-DD')  = d.order_date   
         where a.checkpoint = '{checkpoint}'
        and a.checkout_time between '{start_date}' and '{end_date}'
    );
    """
    conn.execute(control_query)
    
def pull_decision_tree_driver(feat_driver_table,
                                control_token_table,
                                target_column,
                                conn):
    """
    feat_drier_table (string): the table name that was used to store relevant new users features
    control_token_table (string): the table name tha identifies control group transactions
    target_column (string): either 'p2_d0' or  'ntl', this column determines the binary classification objective
    either when a transaction has an ntl >0 or p2_d0 > 0
    conn (Session): sql connection that is necessary to query snowflake tables 
    returns (pandas dataframe): a dataframe that has all the features, for the control group, and a column 'target' that takes value 1 of the 
    transaction has a {target_column} > 0
    """
    decision_tree_driver = 'Select A.*'
    if target_column.lower() == 'p2_d0':
        decision_tree_driver += f''',b.p2_overdue_d0_local    
                                    ,b.p2_due_local
                                    ,b.ntl_forecast_local
                                    ,case when b.p2_overdue_d0_local > 0 then 1 else 0 end as target
                                    '''

    else:
        decision_tree_driver += f''',b.p2_overdue_d0_local    
                                    ,b.p2_p2_due_local
                                    ,b.ntl_forecast_local
                                    ,case when b.ntl_forecast_local > 0 then 1 else 0 end as target
                                    '''
    decision_tree_driver += f''' from AP_CUR_FRDRISK_G.public.{feat_driver_table} a
                                inner join (select distinct order_token, P2_OVERDUE_D0_LOCAL, p2_due_local, ntl_forecast_local from AP_CUR_FRDRISK_G.public.{control_token_table} where is_in_attempt_control_group = 1) b
                                on a.order_token = b.order_token
                                where coalesce(a.days_since_first_order_date,0) < 15
                                '''
    print(decision_tree_driver)
    dtree_driver = conn.download(decision_tree_driver)
    return(dtree_driver)

#function used to make a string representation of a function into an executable function
def string_to_function(func_str):
    """
    func_string (string): a string that represents a function, (i.e. rule logic)
    returns function: the same string, but as an executable string
    """
    # Create a local namespace dictionary to execute the function string
    local_namespace = {}
    # Execute the function string to define the function in the local namespace
    exec(func_str, globals(), local_namespace)
    # Return the function object from the local namespace
    return local_namespace['execute_rule']


def get_datatypes(decision_tree_driver):
    """
    decision_tree_driver (pandas dataframe): the output of pull_decision_tree_driver
    returns decision_tree_driver (pandas dataframe) a properly datatyped version of the pandas dataframe that will be able to
    passed into the tensorflow decision tree implementation with minimal issues 
    """
    import dateutil.parser
    import pandas as pd
    from datetime import datetime
    import numpy as np
    return_dict = {'numeric':[],
                   'string':[],
                   'datetime':[]
                   }
    loss_tagging_cols = set(['order_token','key_event_info_id', 'event_info_event_time', 'checkpoint', 'par_region', 'consumer_id', 'par_process_date'])

    cols = set(decision_tree_driver.columns.tolist()[:-1])
    feats = cols - loss_tagging_cols
    
    for col in feats:
        if 'timestamp' in col or 'millis' in col:
            print(f'assessing whether {col} is a timestamp')
            try:
                pd.to_datetime(decision_tree_driver[col].iloc[0],unit='ms')
                #don't actually convert the column because TensorFlow needs integer datatypes
                # decision_tree_driver[col] = pd.to_datetime(dtree_driver[col],unit='ms')
                return_dict['datetime'].append(col)
            except:
                print('{col} is not a timestamp, this issue will need to be fixed before continuing the script')
                return()
        elif 'datetime' in col:
           print(f'assessing whether {col} is a datetime')
           #convert to datetime
           decision_tree_driver[col] = pd.to_datetime(decision_tree_driver[col])
           decision_tree_driver[col] = decision_tree_driver[col].astype('float32')


           return_dict['datetime'].append(col)
        elif 'first_seen' in col and 'days' not in col:
           print(f'assessing whether {col} is a datetime')
           decision_tree_driver[col] = pd.to_datetime(decision_tree_driver[col])
           #have to replace NaT with a timestamp
           decision_tree_driver[col] = decision_tree_driver[col].fillna('1970-01-01 00:00:00')
           decision_tree_driver[col] = decision_tree_driver[col].astype('str')

           #https://discuss.python.org/t/conveting-timestamp-into-integer-or-float/22934/7
           decision_tree_driver[col] = decision_tree_driver[col].apply(dateutil.parser.parse).apply(datetime.timestamp)
        #    decision_tree_driver[col] = decision_tree_driver[col].astype('datetime64[D]')
        #    decision_tree_driver[col] = decision_tree_driver[col].astype('float')

           return_dict['datetime'].append(col)

        elif 'category_code' in col or 'merchant_id' in col or 'postcode' in col:
             print(f'{col} is a string type variable')
             decision_tree_driver[col] = decision_tree_driver[col].astype('str')
             return_dict['string'].append(col)
        else:
            try:
                    decision_tree_driver[col] = pd.to_numeric(decision_tree_driver[col])
                    return_dict['numeric'].append(col)
            except:
                    print(f'{col} is a string type variable')
                    return_dict['string'].append(col)
            
    print('replacing -999 with np.nan to ensure decision trees do not form bad splits based off na data')
    decision_tree_driver = decision_tree_driver.replace(-999, np.nan)
    return(decision_tree_driver, return_dict)

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

def create_feature_table_using_rule_name(feat_table_name, feature_list, conn, user_name, par_region, checkpoint, start_date=None, end_date =None, debug_mode=True):
    import pandas as pd
    if start_date and end_date:
       time_diff = pd.Timestamp(end_date) - pd.Timestamp(start_date)
       if int(time_diff.days) > 124: #124 is the an approximation for 4 months in days
            print('analytical window is longer than four months, splitting the window in half to assist in making sure the query runs')
            midpoint = (pd.Timestamp(start_date) + (time_diff / 2)).strftime('%Y-%m-%d')
            print(f'mid point is {midpoint}')
            feat_table_name1=f'{feat_table_name}_driver1'
            feat_table_name2=f'{feat_table_name}_driver2'
    create_statement = f'''create or replace table AP_CUR_FRDRISK_G.public.{feat_table_name1}'''
    query_intro=f''' as (
        select 
            order_token
            ,key_event_info_id
            ,event_info_event_time
            ,checkpoint
            ,par_region
            ,consumer_id
            ,par_process_date
            '''
    feature_query = ''
    for i in range(len(feature_list)):
        feature = feature_list[i]
        feature_query +=f''',max(case when var_name ilike '{feature}' then var_value end) as {feature}
        '''
    feature_query += f''' from AP_CUR_R_FEATSCI.CURATED_FEATURE_SCIENCE_RED.TBL_RAW_C_E_FC_DECISION_RECORD_RULE_VARS__{user_name}_DSL3_SV 
       where 1=1 
    '''
    
    query_outro = f''' and par_process_date >= '{start_date}' and par_process_date < '{midpoint}' and par_region ilike '{par_region}'
    and checkpoint ilike '{checkpoint}'
        '''


    query_outro += ''' group by 1,2,3,4,5,6,7 );''' 
    query_f = create_statement + query_intro + feature_query + query_outro
    if debug_mode:
        print(query_f)
    else:
        print('running the query, this will take 30 minutes 1hr , while you wait, please go to the next script to examine rule performance')
        conn.execute(query_f)
        print(f'sucessfully created table AP_CUR_FRDRISK_G.public.{feat_table_name1}')

    #now repeat for the second driver table
    create_statement = f'''create or replace table AP_CUR_FRDRISK_G.public.{feat_table_name2}'''
    query_outro = f''' and par_process_date >= '{midpoint}' and par_process_date <= '{end_date}' and par_region ilike '{par_region}'
    and checkpoint ilike '{checkpoint}'
    '''
    query_outro += ''' group by 1,2,3,4,5,6,7 );''' 
    query_f = create_statement + query_intro + feature_query + query_outro
    if debug_mode:
        print(query_f)
    else:
        print('running the query, this will take 30 minutes 1hr , while you wait, please go to the next script to examine rule performance')
        conn.execute(query_f)
        print(f'sucessfully created table AP_CUR_FRDRISK_G.public.{feat_table_name2}')
    
def prep_for_training(decision_tree_driver,exclude_features,test_ratio):

    #specify which columns to exclude from training,
        loss_tagging_cols = set(['order_token','key_event_info_id', 'event_info_event_time', 'checkpoint', 'par_region', 'consumer_id', 'par_process_date', 
                            'p2_overdue_d0_local',
                            'p2_due_local',
                            'ntl_forecast_local',
                            'checkout_time',
                            ])
        for feat in exclude_features:
            loss_tagging_cols.add(feat)



        #specify features used in training
        cols = set(decision_tree_driver.columns.tolist())
        feats = cols - loss_tagging_cols
         

        #create train data
        X = decision_tree_driver[feats]
        return(X, decision_tree_driver)

def split_dataset(dataset, test_ratio=0.25): #create a larget test set to computationally build and evaluate trees on smaller dataset
    """Splits a panda dataframe in two."""
    import numpy as np
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

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

def create_py_rule(rule,rule_name,path_name,split=False,split_condition=None, debug=True):
    from functions import format_code
    import re

    intro = '''def execute_rule():
        actions = []
        ### BEGIN RULE CONTENT ###'
        '''
    rule = rule.split('END as')[0]
    if split == True:
        rule = rule.replace('CASE WHEN',f'if ({split_condition}) and')
    else: 
        rule = rule.replace('CASE WHEN',f'if ')
    rule = rule.replace("THEN 1 ELSE 0 ",":")
    # intro+='\n'
    intro+= f'{rule}'
    # intro+='\n\t\t'
    intro+="\n\t\tactions.append({'action_name': 'is_rejected_assign'})"
    intro+="""\n\t   ### END RULE CONTENT ###\n"""
    intro+='\t\treturn actions'
    # intro+='\n\treturn actions'
    print(intro)
    pattern = r'\b(in)\s*\(\s*((?:\'[^\']*\'|\d+)(?:\s*,\s*(?:\'[^\']*\'|\d+))*)\s*\)'
    replacement = r'\1 {\2}' 
    intro = re.sub(pattern, replacement, intro)
    intro = format_code(intro)
    #replace in parenthesis logic with braces 

    print(intro)
    if debug==False:
        with open(f'{path_name}{rule_name}', 'w') as f:
            f.write(intro)
    return(intro)

#what do i need to do, take the decline rate (#), decline rate ($)
#unique decline rate (#), unique decline rate ($)
#control p2d0 rate and unique control p2 d0 rate
#coverage and unique coverage

def analyze_performance(str_df, segment_name, decline_rate_denom_ct, decline_rate_denom_amt, coverage_denom):
    pd.options.display.float_format = '{:.3%}'.format
    decline_rate_ct_num = int(run_query(f'''select count(distinct(case when {segment_name} = 1 then order_token end)) as decline_rate_ct_num
                          from {str_df}
                          ''').iloc[0])
    decline_rate_ct = decline_rate_ct_num/decline_rate_denom_ct
    decline_rate_amt_num = int(run_query(f'''select sum(case when {segment_name} = 1 then order_amount_local end) as decline_rate_amt_num
                          from {str_df}
                          ''').iloc[0])
    decline_rate_amt = decline_rate_amt_num/decline_rate_denom_amt

    rule_p2_d0 = float(run_query(f'''select sum(case when {segment_name} = 1 then ctrl_p2_d0 end)/sum(case when {segment_name} = 1 then ctrl_p2_due end) as p2_d0_rate
                          from {str_df}
                          ''').iloc[0])

    rule_coverage = float(run_query(f'''select sum(case when {segment_name} = 1 then ctrl_p2_d0 end)/{coverage_denom} as p2_d0_num
                          from {str_df}
                          ''').iloc[0])

    unique_decline_rate_ct_num = float(run_query(f'''select count(distinct(case when {segment_name} = 1 then order_token end)) as unique_decline_rate_ct_num
                          from {str_df}
                          where unique_decline_token is not null

                          ''').iloc[0])
    unique_decline_rate_ct = unique_decline_rate_ct_num/decline_rate_denom_ct                      
    unique_decline_rate_amt_num = int(run_query(f'''select sum(case when {segment_name} = 1 then order_amount_local end) as unique_decline_rate_amt_num
                          from {str_df}
                          where unique_decline_token is not null
                          ''').iloc[0])
    unique_decline_rate_amt = unique_decline_rate_amt_num/decline_rate_denom_amt

    unique_rule_p2_d0 = float(run_query(f'''select sum(case when {segment_name} = 1 then ctrl_p2_d0 end)/sum(case when {segment_name} = 1 then ctrl_p2_due end) as unique_p2_d0_rate
                          from {str_df}
                        where unique_decline_token is not null

                          ''').iloc[0])


    unique_rule_coverage = float(run_query(f'''select sum(case when {segment_name} = 1 then ctrl_p2_d0 end)/{coverage_denom} as unique_p2_d0_coverage
                          from {str_df}
                          where unique_decline_token is not null
                          ''').iloc[0])
    output_df = pd.DataFrame([[decline_rate_ct, decline_rate_amt, rule_p2_d0, rule_coverage, unique_decline_rate_ct, unique_decline_rate_amt, unique_rule_p2_d0,unique_rule_coverage]], columns= ['decline_rate_ct', 'decline_rate_amt', 'rule_p2_d0', 'rule_coverage', 'unique_decline_rate_ct', 'unique_decline_rate_amt', 'unique_p2_d0_rate', 'unique_rule_coverage'])
    return(output_df)

def modify_segment_name(rule, new_segment_name):
    rule = rule.split('END as')
    rule.insert(1, f'END as {new_segment_name}')
    rule = rule[:2]
    rule = ' '.join(rule)
    print(rule)
    return(rule)

def grab_new_rule_performance(rule_performance_table_name, 
                              unique_decline_table_name,
                              rule,
                              start_date,
                              end_date,
                              par_region,
                              checkpoint, 
                              user_name,
                             conn):
    query =     f'''create or replace table ap_cur_frdrisk_g.public.{rule_performance_name} as (
    select 
        a.order_token,
        a.par_region,
        a.checkpoint,
        a. par_process_date,
        {rule},
        a.in_flight_order_merchant_id,
        a.in_flight_order_amount,
        case when e.order_token is null then a.order_token end as unique_decline_token,
        is_in_attempt_control_group,
        case when is_in_attempt_control_group = 1 then P2_OVERDUE_D0_LOCAL end as ctrl_p2_d0,
        case when is_in_attempt_control_group = 1 then P2_due_LOCAL end as ctrl_p2_due,
        from AP_CUR_R_FRDRISK.CURATED_FRAUD_RISK_RED.UNIFIED_FEATURE_DATAMART_BASE__{user_name}_DSL3_SV a
        left join AP_CUR_RISKBI_G.CURATED_RISK_BI_GREEN.DWM_ORDER_LOSS_TAGGING  d
        on a.order_token = d.order_token
        left join ap_cur_frdrisk_g.public.{unique_decline_table_name} e
        on a.order_token = e.order_token
        where checkout_time between '{start_date}' and '{end_date}' and a.par_region = '{par_region}' and a.checkpoint = '{checkpoint}'
        and coalesce(a.days_since_first_order_date,0) < 15);'''
    conn.execute(query)

def order_amount_fixing(rule_performance_table_name, conn):
    dedupe_query = f'''create or replace temp table ap_cur_frdrisk_g.public.order_amt_temp as (
        select token, max(event_info_event_time) as latest_event_Time from ap_raw_green.green.raw_c_e_order where token in (select distinct order_token from ap_cur_frdrisk_g.public.{rule_performance_table_name}) 
        group by 1
    )'''
    conn.execute(dedupe_query)

    order_fixing_query = f'''create or replace temp table  ap_cur_frdrisk_g.public.order_amt_fixed as (
        select b.token, a.CONSUMER_TOTAL_AMOUNT_AMOUNT from ap_raw_green.green.raw_c_e_order a
        inner join  ap_cur_frdrisk_g.public.order_amt_temp b
        on a.token = b.token
        and a.event_info_event_time = b.latest_event_time
         where a.token in (select distinct order_token from ap_cur_frdrisk_g.public.{rule_performance_table_name}) 
        )'''
    conn.execute(order_fixing_query)
    
    analysis_df = conn.download(f'''select a.*, coalesce(b.consumer_total_amount_amount, in_flight_order_amount) as order_amount_local from ap_cur_Frdrisk_g.public.{rule_performance_table_name}  a
                           left join ap_cur_frdrisk_g.public.order_amt_fixed b
                           on a.order_token = b.token''')
    return(analysis_df)

