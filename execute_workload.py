import pymysql
import pandas as pd
import os
from tqdm import tqdm

doris_host='101.6.5.211'
doris_port =9030
doris_user='root'
doris_password=''
doris_db='tpch_sf100'


conn=pymysql.connect(host=doris_host,port=doris_port, user=doris_user, passwd=doris_password, db=doris_db)

workload_dir = 'workloads/tpc_h'
file='workload_100k_s1.sql'
with open(os.path.join(workload_dir, file)) as f:
    workload = f.read()

count=0
for query in tqdm(workload.split('\n')):
    try:
        query=query.replace('"','')
        query=query.strip()
        cursor=conn.cursor()
        # print(f"Executing query: \n{query}\n")
        cursor.execute(query)
        rows = cursor.fetchall()
        # print(rows)
        # plan=""
        # for row in rows:
        #     plan+='\n' + row[0]
        # print(plan)
        cursor.close()
    except Exception as e:
        print(f"Error executing query: {e}")
        continue
conn.close()
