# %%
# version 2 saves explain plan to external file for each queryid


import re
import json
import os
# from tqdm import tqdm
from collections import defaultdict


def collect_benchmark(benchmark):
    fe_log_dir='/home/wuy/doris-master/output/fe/log'

    ### how mem.txt is obtained: cat be.INFO | grep runtime_query_statistics_mgr\.cpp\:67 > ~/wuy/DB/doris/mem.txt
    be_log_dir = '/home/wuy/doris-master/output/be/log'

    # output_file='/home/ahzgroup/wuy/DB/doris/query_mem_data.json'
    # extra_output_file = '/home/ahzgroup/wuy/DB/doris/query_mem_data_full.json'
    output_file_csv=f'/home/wuy/DB/doris_mem_pred/tpch_data/query_mem_data_{benchmark}.csv'
    output_plan_dir=f"/home/wuy/DB/doris_mem_pred/tpch_data/plans"
    if not os.path.exists(output_plan_dir):
        os.makedirs(output_plan_dir)

    N=10

    # %%
    state_list=["OK", "ERR", "EOF"]
    # fe_log_info_format={
    #     0:"Client", 1:"User", 2:"Ctl", 3:""
    queries_info={}

    # %%

    files=os.listdir(fe_log_dir)
    for filename in files:
        if not filename.startswith("fe.audit.log"):
            continue
        print(f"reading {filename}")
        with open(os.path.join(fe_log_dir, filename)) as f:
            lines=f.readlines()  # num(lines)=1898
            for line in lines:
                try:
                    query={}
                    query['time']=line[:23]
                    reexp=r'\[(.*?)\]'
                    query['query_or_slow_query']=re.findall(reexp,line)[0]
                    info=line.split('|')[1:]
                    is_query=info[13].split('=')[1]
                    if is_query == "false":
                        continue
                    state=info[4].split('=')[1]
                    if state == 'ERR':
                        continue

                    for i in [4,7,8,9,10,11,12,13,16,17,18,19,21]:
                        item=info[i]
                        split_index = item.find('=')
                        name = item[:split_index]
                        value = item[split_index+1:]
                        query[name]=value
                    query["Time(ms)"]=int(query["Time(ms)"])
                    query["ScanBytes"]=int(query["ScanBytes"])
                    query["ReturnRows"]=int(query["ReturnRows"])
                    query["StmtId"]=int(query["StmtId"])
                    query["CpuTimeMS"]=int(query["CpuTimeMS"])
                    # query["ShuffleSendBytes"]=int(query["ShuffleSendBytes"])
                    # query["ShuffleSendRows"]=int(query["ShuffleSendRows"])
                    query["peakMemoryBytes"]=int(query["peakMemoryBytes"])
                    queryid=query["QueryId"]
                    queries_info[queryid]=query
                except Exception as e:
                    print(f"exception {e}")
                finally:
                    continue

    print(f"fe audit logs read")
    # %%
    queries_info

    # %%
    def reduce(lt,N):
        mem_list=[]
        length=len(lt)
        for i in range(N):
            mem_list.append(max( [list(x.values())[0] for x in lt[round(length/N*i): round(length/N*(i+1))]] ))
        return mem_list

    import pymysql
    import pandas as pd
    doris_host='101.6.5.211'
    doris_port =9030
    doris_user='root'
    doris_password=''
    doris_db=benchmark
    conn=pymysql.connect(host=doris_host,port=doris_port, user=doris_user, passwd=doris_password, db=doris_db)
    cursor=conn.cursor()



    csv_header=['queryid','time','mem_list', 'stmt']
    with open(output_file_csv, 'w') as fout:
        fout.write(';'.join(csv_header)+'\n')

    valid_query_count=0


    files=os.listdir(be_log_dir)
    for filename in files:
        # if not filename == 'be.INFO.log.20240605-135502':
        #     continue
        if not filename.startswith("be.INFO."):
            continue
        print(f"reading {filename}")
        with open(os.path.join(be_log_dir, filename)) as fin:
            line=fin.readline()
            while line:
                try:
                    timepoint=line[1:25]
                    queryid=line.split('queryid:')[1].split(',')[0].strip()
                    is_query_finished = line.split('is_query_finished:')[1].split(',')[0].strip()
                    mem_bytes=int(line.split('bytes:')[1].strip())
                    # print(f"queryid {queryid}, {is_query_finished}, {mem_bytes}")
                    if queryid in queries_info:
                        if "mem_list" not in queries_info[queryid]:
                            print(f"add mem_list to {queryid}")
                            queries_info[queryid]["mem_list"]=[]
                        else:
                            queries_info[queryid]["mem_list"].append({timepoint:mem_bytes})
                        if is_query_finished == '1':
                            if len(queries_info[queryid]['mem_list']) > N:
                                try:
                                    stmt=queries_info[queryid]["Stmt"]
                                    stmt = stmt.replace(';',' ').strip()
                                    query_stmt = " explain optimized plan " + stmt + ";"
                                    cursor.execute(query_stmt)
                                    rows = cursor.fetchall()
                                    plan=""
                                    for row in rows:
                                        plan += row[0]+'\n'
                                except Exception as mysqle:
                                    print(f"error {mysqle}")
                                lt = queries_info[queryid]["mem_list"]
                                mem_list = reduce(lt, N)
                                total_time = queries_info[queryid]['Time(ms)']
                                entry = str(queryid) + "; " + str(total_time) + "; " + str(mem_list) + "; " + stmt
                                with open(output_file_csv, 'a') as fout:
                                    fout.write(entry + '\n')
                                print(f"write queryid {queryid} to {output_file_csv}")
                                output_plan_file = os.path.join(output_plan_dir, queryid + ".txt")
                                with open(output_plan_file, 'w') as fout:
                                    fout.write(plan)
                                print(f"write plan of {queryid} to {output_plan_file}")
                                valid_query_count += 1
                                del queries_info[queryid] # this is needed to release memory otherwise memory will boom.
                except Exception as e:
                    pass
                line=fin.readline()



    print(f"valid_query number: {valid_query_count}")


    cursor.close()
    conn.close()


# benchmark_list = ["tpcds_sf100", "tpch_sf100", "tpcds_sf1", "ssb_sf100"]
benchmark_list = ["tpch_sf100"]
for benchmark in benchmark_list:
    collect_benchmark(benchmark)