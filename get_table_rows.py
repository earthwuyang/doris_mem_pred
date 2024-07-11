import pymysql
import pandas as pd
doris_host='101.6.43.203'
doris_port =9030
doris_user='root'
doris_password=''
doris_db='tpch_sf100'

conn=pymysql.connect(host=doris_host,port=doris_port, user=doris_user, passwd=doris_password, db=doris_db)

table_list = ["region", "nation", "supplier", "customer", "lineitem", "orders", "part", "partsupp"]
table_rows={}
for table_name in table_list:
    query = f"show table stats {table_name}"
    df=pd.read_sql_query(query, conn)
    table_rows[table_name]=df['row_count'][0]

print(table_rows)