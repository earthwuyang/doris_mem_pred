import re
import json

from node import Node, extract_stats

def parse_physical_node(node_str, node_level):
    # Match the main node with its attributes
   #  node_pattern = r'([A-Za-z0-9]+)(?:\[([0-9]+)\])?(?:@([0-9]+)?)? \(([^)]+)\)'  # CAUTION that for tpcds, this node_pattern is different from below pattern, in that ?:@( ?:[0-9]+)
    node_pattern = r'([A-Za-z0-9]+)(?:\[([0-9]+)\])?(?:@([0-9]+)?)? \((.+)'  
    match = re.match(node_pattern, node_str)
    if not match:
        raise ValueError(f"Unable to parse node: {node_str}")
    name, id_, order, attributes = match.groups()
    cardinality = extract_stats(attributes)

    # attributes_dict = {}

    # # Handle attributes with potential spaces in values
    # attr_pairs = re.findall(r'(\w+)=\[((?:[^\]]|(?<=\\)\])*)\]|\w+=\S+|\w+=(?:"[^"]+"|\'[^\']+\')', attributes)
    # for pair in attr_pairs:
    #     if '=' in pair:
    #         key, value = pair.split("=", 1)
    #         attributes_dict[key.strip()] = value.strip()
    #     else:
    #         attributes_dict[pair[0].strip()] = pair[1].strip() if pair[1] else ''

    node = Node(name, int(id_) if id_ else -1, int(order) if order else -1, attributes, int(float(cardinality)) if cardinality else 0, node_level)  

    return node

def parse_tree(text):
    texts=text.strip().split('\n')
    stack=[]
    for line in texts:
        # print(f"line: [{line}]")
        # node_pattern = r'([A-Za-z0-9]+\[[0-9]+\](?:@[0-9]+)? \([^)]+\))'
        # node_pattern = r'([A-Za-z0-9]+(?:\[[0-9]+\])?(?:@(?:[0-9]+)?)? \([^)]+\))'
        # node_pattern = r'([A-Za-z0-9]+(?:\[[0-9]+\])?(?:@(?:[0-9]+)?)? \([.]+)'

        # node_str = re.findall(node_pattern, line)[0]
        prefix = line.split('Physical')[0]
        node_level = len(prefix)
        if prefix:
            node_str = line.split(prefix)[1]
        else: # for first line, prefix='',so need this branch
            node_str = line
        node = parse_physical_node(node_str, node_level)

        while stack and stack[-1].node_level >= node.node_level:
            stack.pop()
        if stack:
            # print(node.children)
            stack[-1].children.append(node) # It's weird why node#226 also has a child after this line, if in __init__, we set self.children=children while children is [] by default
            # print(node.children)

        stack.append(node)


    return stack[0] # return root node
  



input_str="""
PhysicalResultSink[890] ( outputExprs=[l_orderkey#18, revenue#33, o_orderdate#9, o_shippriority#15] )
+--PhysicalTopN[887]@15 ( limit=10, offset=0, orderKeys=[revenue#33 desc, o_orderdate#9 asc null first], phase=MERGE_SORT, enableRuntimeFilter=false )
   +--PhysicalDistribute[884]@17 ( distributionSpec=DistributionSpecGather, stats=10 1 )
      +--PhysicalTopN[881]@17 ( limit=10, offset=0, orderKeys=[revenue#33 desc, o_orderdate#9 asc null first], phase=LOCAL_SORT, enableRuntimeFilter=false )
         +--PhysicalProject[878]@14 ( projects=[l_orderkey#18, revenue#33, o_orderdate#9, o_shippriority#15], stats=48,024,890.63 1 )
            +--PhysicalHashAggregate[875]@13 ( aggPhase=LOCAL, aggMode=INPUT_TO_RESULT, maybeUseStreaming=false, groupByExpr=[l_orderkey#18, o_orderdate#9, o_shippriority#15], outputExpr=[l_orderkey#18, o_orderdate#9, o_shippriority#15, sum((l_extendedprice#23 * (1.00 - cast(l_discount#24 as DECIMALV3(16, 2))))) AS `revenue`#33], partitionExpr=Optional[[l_orderkey#18, o_orderdate#9, o_shippriority#15]], requireProperties=[DistributionSpecHash ( orderedShuffledColumns=[18, 9, 15], shuffleType=REQUIRE, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[18], [9], [15]], exprIdToEquivalenceSet={18=0, 9=1, 15=2} ) Order: ([])], stats=48,024,890.63 1 )
               +--PhysicalProject[872]@12 ( projects=[l_orderkey#18, o_orderdate#9, o_shippriority#15, l_extendedprice#23, l_discount#24], stats=48,024,890.63 3 )
                  +--PhysicalHashJoin[869]@11 ( type=INNER_JOIN, stats=48,024,890.63 3, hashCondition=[(l_orderkey#18 = o_orderkey#8)], otherCondition=[], runtimeFilters=[RF1[o_orderkey#8->[l_orderkey#18](ndv/size = 22583259/33554432) ] )
                     |--PhysicalProject[837]@10 ( projects=[l_orderkey#18, l_extendedprice#23, l_discount#24], stats=322,475,815.06 1 )
                     |  +--PhysicalFilter[834]@9 ( predicates=(l_shipdate#17 > 1995-03-15), stats=322,475,815.06 1 )
                     |     +--PhysicalOlapScan[2]@8 ( qualified=tpch_sf100.lineitem, stats=600,037,902 1, fr=Optional[1] )
                     +--PhysicalDistribute[866]@7 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[8], shuffleType=STORAGE_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[8]], exprIdToEquivalenceSet={8=0} ), stats=22,583,259.53 2 )
                        +--PhysicalProject[863]@7 ( projects=[o_orderkey#8, o_orderdate#9, o_shippriority#15], stats=22,583,259.53 2 )
                           +--PhysicalHashJoin[860]@6 ( type=INNER_JOIN, stats=22,583,259.53 2, hashCondition=[(c_custkey#0 = o_custkey#10)], otherCondition=[], runtimeFilters=[RF0[c_custkey#0->[o_custkey#10](ndv/size = 3000000/4194304) ] )
                              |--PhysicalDistribute[847]@5 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[10], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[10]], exprIdToEquivalenceSet={10=0} ), stats=72,910,602.91 1 )
                              |  +--PhysicalProject[844]@5 ( projects=[o_orderkey#8, o_orderdate#9, o_custkey#10, o_shippriority#15], stats=72,910,602.91 1 )
                              |     +--PhysicalFilter[841]@4 ( predicates=(o_orderdate#9 < 1995-03-15), stats=72,910,602.91 1 )
                              |        +--PhysicalOlapScan[1]@3 ( qualified=tpch_sf100.orders, stats=150,000,000 1, fr=Optional[3] )
                              +--PhysicalDistribute[857]@2 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[0], shuffleType=EXECUTION_BUCKETED, tableId=85811, selectedIndexId=85812, partitionIds=[85810], equivalenceExprIds=[[0]], exprIdToEquivalenceSet={0=0} ), stats=3,000,000 1 )
                                 +--PhysicalProject[854]@2 ( projects=[c_custkey#0], stats=3,000,000 1 )
                                    +--PhysicalFilter[851]@1 ( predicates=(c_mktsegment#6 = 'BUILDING'), stats=3,000,000 1 )
                                       +--PhysicalOlapScan[0]@0 ( qualified=tpch_sf100.customer, stats=15,000,000 1, fr=Optional[4] )
"""

# tpcds
input_str = """
PhysicalCTEAnchor ( cteId=CTEId#0 )
|--PhysicalCTEProducer[2232] ( cteId=CTEId#0 )
|  +--PhysicalProject[2229]@8 ( projects=[sr_customer_sk#4 AS `ctr_customer_sk`#48, sr_store_sk#8 AS `ctr_store_sk`#49, ctr_total_return#50], stats=5,180,660.56 1 )
|     +--PhysicalHashAggregate[2226]@7 ( aggPhase=GLOBAL, aggMode=BUFFER_TO_RESULT, maybeUseStreaming=false, groupByExpr=[sr_customer_sk#4, sr_store_sk#8], outputExpr=[sr_customer_sk#4, sr_store_sk#8, sum(partial_sum(sr_fee)#106) AS `ctr_total_return`#50], partitionExpr=Optional[[sr_customer_sk#4, sr_store_sk#8]], requireProperties=[DistributionSpecHash ( orderedShuffledColumns=[4, 8], shuffleType=REQUIRE, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[4], [8]], exprIdToEquivalenceSet={4=0, 8=1} ) Order: ([])], stats=5,180,660.56 1 )
|        +--PhysicalDistribute[2223]@28 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[4, 8], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[4], [8]], exprIdToEquivalenceSet={4=0, 8=1} ), stats=5,180,660.56 1 )
|           +--PhysicalHashAggregate[2220]@28 ( aggPhase=LOCAL, aggMode=INPUT_TO_BUFFER, maybeUseStreaming=true, groupByExpr=[sr_customer_sk#4, sr_store_sk#8], outputExpr=[sr_customer_sk#4, sr_store_sk#8, partial_sum(sr_fee#14) AS `partial_sum(sr_fee)`#106], partitionExpr=Optional[[sr_customer_sk#4, sr_store_sk#8]], requireProperties=[ANY], stats=5,180,660.56 1 )
|              +--PhysicalProject[2217]@6 ( projects=[sr_customer_sk#4, sr_store_sk#8, sr_fee#14], stats=5,180,660.56 2 )
|                 +--PhysicalHashJoin[2214]@5 ( type=INNER_JOIN, stats=5,180,660.56 2, hashCondition=[(sr_returned_date_sk#2 = d_date_sk#20)], otherCondition=[], runtimeFilters=[RF0[d_date_sk#20->[sr_returned_date_sk#2](ndv/size = 361/512) ] )
|                    |--PhysicalProject[2201]@1 ( projects=[sr_returned_date_sk#2, sr_customer_sk#4, sr_store_sk#8, sr_fee#14], stats=28,795,080 1 )
|                    |  +--PhysicalOlapScan[4]@0 ( qualified=tpcds_sf100.store_returns, stats=28,795,080 1, fr=Optional[1] )
|                    +--PhysicalDistribute[2211]@4 ( distributionSpec=DistributionSpecReplicated, stats=361.63 1 )
|                       +--PhysicalProject[2208]@4 ( projects=[d_date_sk#20], stats=361.63 1 )
|                          +--PhysicalFilter[2205]@3 ( predicates=(d_year#26 = 2000), stats=361.63 1 )
|                             +--PhysicalOlapScan[5]@2 ( qualified=tpcds_sf100.date_dim, stats=73,049 1, fr=Optional[2] )
+--PhysicalResultSink[2302] ( outputExprs=[c_customer_id#84] )
   +--PhysicalTopN[2299]@25 ( limit=100, offset=0, orderKeys=[c_customer_id#84 asc null first], phase=MERGE_SORT, enableRuntimeFilter=false )
      +--PhysicalDistribute[2296]@29 ( distributionSpec=DistributionSpecGather, stats=100 4 )
         +--PhysicalTopN[2293]@29 ( limit=100, offset=0, orderKeys=[c_customer_id#84 asc null first], phase=LOCAL_SORT, enableRuntimeFilter=false )
            +--PhysicalProject[2290]@24 ( projects=[c_customer_id#84], stats=522,120.15 4 )
               +--PhysicalHashJoin[2287]@34 ( type=INNER_JOIN, stats=522,120.15 4, hashCondition=[(ctr_customer_sk#51 = c_customer_sk#83)], otherCondition=[], runtimeFilters=[RF2[ctr_customer_sk#51->[c_customer_sk#83](ndv/size = 520656/524288) ] )
                  |--PhysicalDistribute[2239]@17 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[83], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[83]], exprIdToEquivalenceSet={83=0} ), stats=2,000,000 1 )
                  |  +--PhysicalProject[2236]@17 ( projects=[c_customer_sk#83, c_customer_id#84], stats=2,000,000 1 )
                  |     +--PhysicalOlapScan[2]@16 ( qualified=tpcds_sf100.customer, stats=2,000,000 1, fr=Optional[4] )
                  +--PhysicalDistribute[2284]@33 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[51], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[51]], exprIdToEquivalenceSet={51=0} ), stats=520,656.39 3 )
                     +--PhysicalHashJoin[2281]@33 ( type=INNER_JOIN, stats=520,656.39 3, hashCondition=[(ctr_store_sk#52 = ctr_store_sk#102)], otherCondition=[(cast(ctr_total_return#53 as DOUBLE) > cast((avg(cast(ctr_total_return as DECIMALV3(38, 4)))#105 * 1.2) as DOUBLE))] )
                        |--PhysicalProject[2259]@15 ( projects=[ctr_customer_sk#51, ctr_store_sk#52, ctr_total_return#53], stats=1,041,312.77 2 )
                        |  +--PhysicalHashJoin[2256]@14 ( type=INNER_JOIN, stats=1,041,312.77 2, hashCondition=[(s_store_sk#54 = ctr_store_sk#52)], otherCondition=[], runtimeFilters=[RF1[s_store_sk#54->[ctr_store_sk#52](ndv/size = 40/64) ] )
                        |     |--PhysicalDistribute[2243]@10 ( distributionSpec=DistributionSpecExecutionAny, stats=5,180,660.56 1 )
                        |     |  +--PhysicalCTEConsumer[2240] ( cteId=CTEId#0 )
                        |     +--PhysicalDistribute[2253]@13 ( distributionSpec=DistributionSpecReplicated, stats=40.2 1 )
                        |        +--PhysicalProject[2250]@13 ( projects=[s_store_sk#54], stats=40.2 1 )
                        |           +--PhysicalFilter[2247]@12 ( predicates=(s_state#78 = 'SD'), stats=40.2 1 )
                        |              +--PhysicalOlapScan[1]@11 ( qualified=tpcds_sf100.store, stats=402 1, fr=Optional[7] )
                        +--PhysicalDistribute[2278]@22 ( distributionSpec=DistributionSpecReplicated, stats=200 1 )
                           +--PhysicalHashAggregate[2275]@22 ( aggPhase=GLOBAL, aggMode=BUFFER_TO_RESULT, maybeUseStreaming=false, groupByExpr=[ctr_store_sk#102], outputExpr=[ctr_store_sk#102, avg(partial_avg(cast(ctr_total_return as DECIMALV3(38, 4)))#107) AS `avg(cast(ctr_total_return as DECIMALV3(38, 4)))`#105], partitionExpr=Optional[[ctr_store_sk#102]], requireProperties=[DistributionSpecHash ( orderedShuffledColumns=[102], shuffleType=REQUIRE, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[102]], exprIdToEquivalenceSet={102=0} ) Order: ([])], stats=200 1 )
                              +--PhysicalDistribute[2272]@32 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[102], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[102]], exprIdToEquivalenceSet={102=0} ), stats=200 1 )
                                 +--PhysicalHashAggregate[2269]@32 ( aggPhase=LOCAL, aggMode=INPUT_TO_BUFFER, maybeUseStreaming=true, groupByExpr=[ctr_store_sk#102], outputExpr=[ctr_store_sk#102, partial_avg(cast(ctr_total_return#103 as DECIMALV3(38, 4))) AS `partial_avg(cast(ctr_total_return as DECIMALV3(38, 4)))`#107], partitionExpr=Optional[[ctr_store_sk#102]], requireProperties=[ANY], stats=200 1 )
                                    +--PhysicalDistribute[2266]@21 ( distributionSpec=DistributionSpecExecutionAny, stats=5,180,660.56 1 )
                                       +--PhysicalProject[2263]@21 ( projects=[ctr_store_sk#102, ctr_total_return#103], stats=5,180,660.56 1 )
                                          +--PhysicalCTEConsumer[2260] ( cteId=CTEId#0 )
"""

input_str = """
PhysicalResultSink[3647] ( outputExprs=[s_acctbal#14, s_name#10, n_name#22, p_partkey#0, p_mfgr#2, s_address#11, s_phone#13, s_comment#15] )
+--PhysicalTopN[3644]@22 ( limit=100, offset=0, orderKeys=[s_acctbal#14 desc, n_name#22 asc null first, s_name#10 asc null first, p_partkey#0 asc null first], phase=MERGE_SORT, enableRuntimeFilter=false )
   +--PhysicalDistribute[3641]@24 ( distributionSpec=DistributionSpecGather, stats=1 1 )
      +--PhysicalTopN[3638]@24 ( limit=100, offset=0, orderKeys=[s_acctbal#14 desc, n_name#22 asc null first, s_name#10 asc null first, p_partkey#0 asc null first], phase=LOCAL_SORT, enableRuntimeFilter=false )
         +--PhysicalProject[3635]@21 ( projects=[s_acctbal#14, s_name#10, n_name#22, p_partkey#0, p_mfgr#2, s_address#11, s_phone#13, s_comment#15], stats=1 1 )
            +--PhysicalFilter[3632]@20 ( predicates=(ps_supplycost#19 = min(ps_supplycost) OVER(PARTITION BY p_partkey)#48), stats=1 1 )
               +--PhysicalWindow[3629]@19 ( windowFrameGroup=(Funcs=[min(ps_supplycost#19) WindowSpec(PARTITION BY p_partkey#0 RANGE BETWEEN UNBOUNDED_PRECEDING AND CURRENT_ROW) AS `min(ps_supplycost) OVER(PARTITION BY p_partkey RANGE BETWEEN UNBOUNDED_PRECEDING AND CURRENT_ROW)`#48], PartitionKeys=[p_partkey#0], OrderKeys=[], WindowFrame=WindowFrame(RANGE, UNBOUNDED_PRECEDING, CURRENT_ROW)), requiredProperties=[DistributionSpecHash ( orderedShuffledColumns=[0], shuffleType=REQUIRE, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[0]], exprIdToEquivalenceSet={0=0} ) Order: ([p_partkey#0 asc])], stats=79,882.21 1 )
                  +--PhysicalQuickSort[3626]@18 ( orderKeys=[p_partkey#0 asc], phase=LOCAL_SORT, stats=79,882.21 5 )
                     +--PhysicalDistribute[3623]@18 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[0], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[0]], exprIdToEquivalenceSet={0=0} ), stats=79,882.21 5 )
                        +--PhysicalProject[3620]@18 ( projects=[p_partkey#0, p_mfgr#2, ps_supplycost#19, n_name#22, s_name#10, s_address#11, s_phone#13, s_acctbal#14, s_comment#15], stats=79,882.21 5 )
                           +--PhysicalHashJoin[3617]@36 ( type=INNER_JOIN, stats=79,882.21 5, hashCondition=[(s_suppkey#9 = ps_suppkey#17)], otherCondition=[], runtimeFilters=[RF3[s_suppkey#9->[ps_suppkey#17](ndv/size = 250000/262144) ] )
                              |--PhysicalDistribute[3584]@6 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[17], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[17]], exprIdToEquivalenceSet={17=0} ), stats=319,193 2 )
                              |  +--PhysicalProject[3581]@6 ( projects=[p_partkey#0, ps_suppkey#17, p_mfgr#2, ps_supplycost#19], stats=319,193 2 )
                              |     +--PhysicalHashJoin[3578]@5 ( type=INNER_JOIN, stats=319,193 2, hashCondition=[(p_partkey#0 = ps_partkey#16)], otherCondition=[], runtimeFilters=[RF2[p_partkey#0->[ps_partkey#16](ndv/size = 80000/131072) ] )
                              |        |--PhysicalProject[3568]@4 ( projects=[ps_partkey#16, ps_suppkey#17, ps_supplycost#19], stats=80,000,000 1 )
                              |        |  +--PhysicalOlapScan[2]@3 ( qualified=tpch_sf100.partsupp, stats=80,000,000 1, fr=Optional[3] )
                              |        +--PhysicalProject[3575]@2 ( projects=[p_partkey#0, p_mfgr#2], stats=80,000 1 )
                              |           +--PhysicalFilter[3572]@1 ( predicates=((p_size#5 = 15) AND (p_type#4 like '%BRASS')), stats=80,000 1 )
                              |              +--PhysicalOlapScan[0]@0 ( qualified=tpch_sf100.part, stats=20,000,000 1, fr=Optional[3] )
                              +--PhysicalDistribute[3614]@35 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[9], shuffleType=EXECUTION_BUCKETED, tableId=85862, selectedIndexId=85863, partitionIds=[85861], equivalenceExprIds=[[9]], exprIdToEquivalenceSet={9=0} ), stats=250,000 3 )
                                 +--PhysicalProject[3611]@35 ( projects=[s_suppkey#9, s_name#10, s_address#11, s_phone#13, s_acctbal#14, s_comment#15, n_name#22, n_regionkey#23, r_regionkey#25], stats=250,000 3 )
                                    +--PhysicalHashJoin[3608]@34 ( type=INNER_JOIN, stats=250,000 3, hashCondition=[(s_nationkey#12 = n_nationkey#21)], otherCondition=[], runtimeFilters=[RF1[n_nationkey#21->[s_nationkey#12](ndv/size = 6/8) ] )
                                       |--PhysicalOlapScan[1]@7 ( qualified=tpch_sf100.supplier, stats=1,000,000 1, fr=Optional[4] )
                                       +--PhysicalDistribute[3605]@32 ( distributionSpec=DistributionSpecReplicated, stats=6.25 2 )
                                          +--PhysicalHashJoin[3602]@32 ( type=INNER_JOIN, stats=6.25 2, hashCondition=[(n_regionkey#23 = r_regionkey#25)], otherCondition=[], runtimeFilters=[RF0[r_regionkey#25->[n_regionkey#23](ndv/size = 1/1) ] )
                                             |--PhysicalProject[3589]@11 ( projects=[n_nationkey#21, n_name#22, n_regionkey#23], stats=25 1 )
                                             |  +--PhysicalOlapScan[3]@10 ( qualified=tpch_sf100.nation, stats=25 1, fr=Optional[5] )
                                             +--PhysicalDistribute[3599]@16 ( distributionSpec=DistributionSpecReplicated, stats=1 1 )
                                                +--PhysicalProject[3596]@16 ( projects=[r_regionkey#25], stats=1 1 )
                                                   +--PhysicalFilter[3593]@15 ( predicates=(r_name#26 = 'EUROPE'), stats=1 1 )
                                                      +--PhysicalOlapScan[4]@14 ( qualified=tpch_sf100.region, stats=5 1, fr=Optional[6] )
"""

# input_str = """
# PhysicalResultSink[229] ( outputExprs=[c_custkey#0, c_name#1, c_address#2, c_nationkey#3, c_phone#4, c_acctbal#5, c_mktsegment#6, c_comment#7, o_orderkey#8, o_orderdate#9, o_custkey#10, o_orderstatus#11, o_totalprice#12, o_orderpriority#13, o_clerk#14, o_shippriority#15, o_comment#16] )
#  +--PhysicalDistribute[226]@5 ( distributionSpec=DistributionSpecGather, stats=90 2 )
#     +--PhysicalProject[223]@5 ( projects=[c_custkey#0, c_name#1, c_address#2, c_nationkey#3, c_phone#4, c_acctbal#5, c_mktsegment#6, c_comment#7, o_orderkey#8, o_orderdate#9, o_custkey#10, o_orderstatus#11, o_totalprice#12, o_orderpriority#13, o_clerk#14, o_shippriority#15, o_comment#16], stats=90 2 )
#        +--PhysicalHashJoin[220]@4 ( type=INNER_JOIN, stats=90 2, hashCondition=[(c_custkey#0 = o_custkey#10)], otherCondition=[], runtimeFilters=[RF0[c_custkey#0->[o_custkey#10](ndv/size = 9/8) ] )
#           |--PhysicalFilter[210]@3 ( predicates=(o_custkey#10 < 10), stats=90 1 )
#           |  +--PhysicalOlapScan[1]@2 ( qualified=tpch_sf100.orders, stats=150,000,000 1, fr=Optional[1] )
#           +--PhysicalDistribute[217]@1 ( distributionSpec=DistributionSpecReplicated, stats=9 1 )
#              +--PhysicalFilter[214]@1 ( predicates=(c_custkey#0 < 10), stats=9 1 )
#                 +--PhysicalOlapScan[0]@0 ( qualified=tpch_sf100.customer, stats=15,000,000 1, fr=Optional[2] )
# """

# Parse the tree
root_node = parse_tree(input_str)

print("Debug tree:")
root_node.print_tree()
# root_node.children[0].print_tree()

# # Convert to JSON
# json_output = json.dumps(parsed_tree, indent=2)
# print(json_output)
