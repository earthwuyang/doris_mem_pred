import re
import json

def parse_physical_node(node_str):
    # Match the main node with its attributes
    node_pattern = r'([A-Za-z0-9]+)\[([0-9]+)\](?:@([0-9]+))? \(([^)]+)\)'
    match = re.match(node_pattern, node_str)
    if not match:
        raise ValueError(f"Unable to parse node: {node_str}")

    name, id_, level, attributes = match.groups()
    attributes_dict = {}

    # Handle attributes with potential spaces in values
    attr_pairs = re.findall(r'(\w+)=\[((?:[^\]]|(?<=\\)\])*)\]|\w+=\S+|\w+=(?:"[^"]+"|\'[^\']+\')', attributes)
    for pair in attr_pairs:
        if '=' in pair:
            key, value = pair.split("=", 1)
            attributes_dict[key.strip()] = value.strip()
        else:
            attributes_dict[pair[0].strip()] = pair[1].strip() if pair[1] else ''

    node = {
        "name": name,
        "id": int(id_),
        "level": int(level) if level else 0,  # Set level to 0 if it's None
        "attributes": attributes_dict,
        "children": []
    }

    return node

def parse_tree(text):
    # Use regex to split the text into nodes at the correct depth
    node_pattern = r'(\+--|\|--)?([A-Za-z0-9]+\[[0-9]+\](?:@[0-9]+)? \([^)]+\))'
    matches = re.findall(node_pattern, text)
    nodes = []

    current_level = -1
    stack = []

    for prefix, node_str in matches:
        node = parse_physical_node(node_str)
        node_level = prefix.count("--") if prefix else 0

        while stack and node_level <= current_level:
            stack.pop()
            if stack:
                current_level = stack[-1]["level"]
            else:
                current_level = -1

        if stack:
            stack[-1]["children"].append(node)
        else:
            nodes.append(node)

        stack.append(node)
        current_level = node_level

    return nodes

input_str = """
PhysicalResultSink[229] ( outputExprs=[c_custkey#0, c_name#1, c_address#2, c_nationkey#3, c_phone#4, c_acctbal#5, c_mktsegment#6, c_comment#7, o_orderkey#8, o_orderdate#9, o_custkey#10, o_orderstatus#11, o_totalprice#12, o_orderpriority#13, o_clerk#14, o_shippriority#15, o_comment#16] )
 +--PhysicalDistribute[226]@5 ( distributionSpec=DistributionSpecGather, stats=90 2 )
    +--PhysicalProject[223]@5 ( projects=[c_custkey#0, c_name#1, c_address#2, c_nationkey#3, c_phone#4, c_acctbal#5, c_mktsegment#6, c_comment#7, o_orderkey#8, o_orderdate#9, o_custkey#10, o_orderstatus#11, o_totalprice#12, o_orderpriority#13, o_clerk#14, o_shippriority#15, o_comment#16], stats=90 2 )
       +--PhysicalHashJoin[220]@4 ( type=INNER_JOIN, stats=90 2, hashCondition=[(c_custkey#0 = o_custkey#10)], otherCondition=[], runtimeFilters=[RF0[c_custkey#0->[o_custkey#10](ndv/size = 9/8) ] )
          |--PhysicalFilter[210]@3 ( predicates=(o_custkey#10 < 10), stats=90 1 )
          |  +--PhysicalOlapScan[1]@2 ( qualified=tpch_sf100.orders, stats=150,000,000 1, fr=Optional[1] )
          +--PhysicalDistribute[217]@1 ( distributionSpec=DistributionSpecReplicated, stats=9 1 )
             +--PhysicalFilter[214]@1 ( predicates=(c_custkey#0 < 10), stats=9 1 )
                +--PhysicalOlapScan[0]@0 ( qualified=tpch_sf100.customer, stats=15,000,000 1, fr=Optional[2] )
"""
# input_str = """
# PhysicalResultSink[138] ( outputExprs=[c_nationkey#3, __count_1#8] )
# +--PhysicalDistribute[135]@2 ( distributionSpec=DistributionSpecGather, stats=25 1 )
#    +--PhysicalHashAggregate[132]@2 ( aggPhase=GLOBAL, aggMode=BUFFER_TO_RESULT, maybeUseStreaming=false, groupByExpr=[c_nationkey#3], outputExpr=[c_nationkey#3, count(partial_count(*)#9) AS `count(*)`#8], partitionExpr=Optional[[c_nationkey#3]], requireProperties=[DistributionSpecHash ( orderedShuffledColumns=[3], shuffleType=REQUIRE, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[3]], exprIdToEquivalenceSet={3=0} ) Order: ([])], stats=25 1 )
#       +--PhysicalDistribute[129]@4 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[3], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[3]], exprIdToEquivalenceSet={3=0} ), stats=25 1 )
#          +--PhysicalHashAggregate[126]@4 ( aggPhase=LOCAL, aggMode=INPUT_TO_BUFFER, maybeUseStreaming=true, groupByExpr=[c_nationkey#3], outputExpr=[c_nationkey#3, partial_count(*) AS `partial_count(*)`#9], partitionExpr=Optional[[c_nationkey#3]], requireProperties=[ANY], stats=25 1 )
#             +--PhysicalProject[123]@1 ( projects=[c_nationkey#3], stats=15,000,000 1 )
#                +--PhysicalOlapScan[0]@0 ( qualified=tpch_sf100.customer, stats=15,000,000 1, fr=Optional[2] )
# """

q2 = """
PhysicalResultSink[3628] ( outputExprs=[s_acctbal#14, s_name#10, n_name#22, p_partkey#0, p_mfgr#2, s_address#11, s_phone#13, s_comment#15] )
+--PhysicalQuickSort[3625]@22 ( orderKeys=[s_acctbal#14 desc, n_name#22 asc null first, s_name#10 asc null first, p_partkey#0 asc null first], phase=MERGE_SORT, stats=1 1 )
   +--PhysicalDistribute[3622]@24 ( distributionSpec=DistributionSpecGather, stats=1 1 )
      +--PhysicalQuickSort[3619]@24 ( orderKeys=[s_acctbal#14 desc, n_name#22 asc null first, s_name#10 asc null first, p_partkey#0 asc null first], phase=LOCAL_SORT, stats=1 1 )
         +--PhysicalProject[3616]@21 ( projects=[s_acctbal#14, s_name#10, n_name#22, p_partkey#0, p_mfgr#2, s_address#11, s_phone#13, s_comment#15], stats=1 1 )
            +--PhysicalFilter[3613]@20 ( predicates=(ps_supplycost#19 = min(ps_supplycost) OVER(PARTITION BY p_partkey)#48), stats=1 1 )
               +--PhysicalWindow[3610]@19 ( windowFrameGroup=(Funcs=[min(ps_supplycost#19) WindowSpec(PARTITION BY p_partkey#0 RANGE BETWEEN UNBOUNDED_PRECEDING AND CURRENT_ROW) AS `min(ps_supplycost) OVER(PARTITION BY p_partkey RANGE BETWEEN UNBOUNDED_PRECEDING AND CURRENT_ROW)`#48], PartitionKeys=[p_partkey#0], OrderKeys=[], WindowFrame=WindowFrame(RANGE, UNBOUNDED_PRECEDING, CURRENT_ROW)), requiredProperties=[DistributionSpecHash ( orderedShuffledColumns=[0], shuffleType=REQUIRE, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[0]], exprIdToEquivalenceSet={0=0} ) Order: ([p_partkey#0 asc])], stats=79,882.21 1 )
                  +--PhysicalQuickSort[3607]@18 ( orderKeys=[p_partkey#0 asc], phase=LOCAL_SORT, stats=79,882.21 5 )
                     +--PhysicalDistribute[3604]@18 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[0], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[0]], exprIdToEquivalenceSet={0=0} ), stats=79,882.21 5 )
                        +--PhysicalProject[3601]@18 ( projects=[p_partkey#0, p_mfgr#2, ps_supplycost#19, n_name#22, s_name#10, s_address#11, s_phone#13, s_acctbal#14, s_comment#15], stats=79,882.21 5 )
                           +--PhysicalHashJoin[3598]@36 ( type=INNER_JOIN, stats=79,882.21 5, hashCondition=[(s_suppkey#9 = ps_suppkey#17)], otherCondition=[], runtimeFilters=[RF3[s_suppkey#9->[ps_suppkey#17](ndv/size = 250000/262144) ] )
                              |--PhysicalDistribute[3565]@6 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[17], shuffleType=EXECUTION_BUCKETED, tableId=-1, selectedIndexId=-1, partitionIds=[], equivalenceExprIds=[[17]], exprIdToEquivalenceSet={17=0} ), stats=319,193 2 )
                              |  +--PhysicalProject[3562]@6 ( projects=[p_partkey#0, ps_suppkey#17, p_mfgr#2, ps_supplycost#19], stats=319,193 2 )
                              |     +--PhysicalHashJoin[3559]@5 ( type=INNER_JOIN, stats=319,193 2, hashCondition=[(p_partkey#0 = ps_partkey#16)], otherCondition=[], runtimeFilters=[RF2[p_partkey#0->[ps_partkey#16](ndv/size = 80000/131072) ] )
                              |        |--PhysicalProject[3549]@4 ( projects=[ps_partkey#16, ps_suppkey#17, ps_supplycost#19], stats=80,000,000 1 )
                              |        |  +--PhysicalOlapScan[2]@3 ( qualified=tpch_sf100.partsupp, stats=80,000,000 1, fr=Optional[3] )
                              |        +--PhysicalProject[3556]@2 ( projects=[p_partkey#0, p_mfgr#2], stats=80,000 1 )
                              |           +--PhysicalFilter[3553]@1 ( predicates=((p_size#5 = 15) AND (p_type#4 like '%BRASS')), stats=80,000 1 )
                              |              +--PhysicalOlapScan[0]@0 ( qualified=tpch_sf100.part, stats=20,000,000 1, fr=Optional[3] )
                              +--PhysicalDistribute[3595]@35 ( distributionSpec=DistributionSpecHash ( orderedShuffledColumns=[9], shuffleType=EXECUTION_BUCKETED, tableId=85862, selectedIndexId=85863, partitionIds=[85861], equivalenceExprIds=[[9]], exprIdToEquivalenceSet={9=0} ), stats=250,000 3 )
                                 +--PhysicalProject[3592]@35 ( projects=[s_suppkey#9, s_name#10, s_address#11, s_phone#13, s_acctbal#14, s_comment#15, n_name#22, n_regionkey#23, r_regionkey#25], stats=250,000 3 )
                                    +--PhysicalHashJoin[3589]@34 ( type=INNER_JOIN, stats=250,000 3, hashCondition=[(s_nationkey#12 = n_nationkey#21)], otherCondition=[], runtimeFilters=[RF1[n_nationkey#21->[s_nationkey#12](ndv/size = 6/8) ] )
                                       |--PhysicalOlapScan[1]@7 ( qualified=tpch_sf100.supplier, stats=1,000,000 1, fr=Optional[4] )
                                       +--PhysicalDistribute[3586]@32 ( distributionSpec=DistributionSpecReplicated, stats=6.25 2 )
                                          +--PhysicalHashJoin[3583]@32 ( type=INNER_JOIN, stats=6.25 2, hashCondition=[(n_regionkey#23 = r_regionkey#25)], otherCondition=[], runtimeFilters=[RF0[r_regionkey#25->[n_regionkey#23](ndv/size = 1/1) ] )
                                             |--PhysicalProject[3570]@11 ( projects=[n_nationkey#21, n_name#22, n_regionkey#23], stats=25 1 )
                                             |  +--PhysicalOlapScan[3]@10 ( qualified=tpch_sf100.nation, stats=25 1, fr=Optional[5] )
                                             +--PhysicalDistribute[3580]@16 ( distributionSpec=DistributionSpecReplicated, stats=1 1 )
                                                +--PhysicalProject[3577]@16 ( projects=[r_regionkey#25], stats=1 1 )
                                                   +--PhysicalFilter[3574]@15 ( predicates=(r_name#26 = 'EUROPE'), stats=1 1 )
                                                      +--PhysicalOlapScan[4]@14 ( qualified=tpch_sf100.region, stats=5 1, fr=Optional[6] )

"""
input_str = q2

# Parse the tree
parsed_tree = parse_tree(input_str)

# Convert to JSON
json_output = json.dumps(parsed_tree, indent=2)
print(json_output)
