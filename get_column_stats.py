import pymysql
import pandas as pd
doris_host='101.6.43.203'
doris_port =9030
doris_user='root'
doris_password=''
doris_db='tpch_sf100'

conn=pymysql.connect(host=doris_host,port=doris_port, user=doris_user, passwd=doris_password, db=doris_db)

table_list = ["region", "nation", "supplier", "customer", "lineitem", "orders", "part", "partsupp"]
column_stats_label = ["count", "ndv", "num_null", "data_size", "avg_size_byte", "min","max"]

column_stats={}
for table_name in table_list:
    column_stats[table_name] = {}
    query=f"show column stats {table_name};"

    df=pd.read_sql_query(query, conn)

    column_list = df['column_name'].tolist()
    for column_name in column_list:
        column_stats[table_name][column_name] = {}

        column_df = df[df['column_name'] == column_name]
        for label in column_stats_label:
            column_stats[table_name][column_name][label] = column_df[label].item()
        column_stats[table_name][column_name]['count']=float(column_stats[table_name][column_name]['count'])
        column_stats[table_name][column_name]["ndv"]=float(column_stats[table_name][column_name]["ndv"])
        column_stats[table_name][column_name]["num_null"]=float(column_stats[table_name][column_name]["num_null"])
        column_stats[table_name][column_name]['data_size'] = float(column_stats[table_name][column_name]['data_size'])
        column_stats[table_name][column_name]['avg_size_byte'] = float(column_stats[table_name][column_name]['avg_size_byte'])

print(column_stats)

conn.close()