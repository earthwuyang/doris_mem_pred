import re
import math

NodeType = {'PhysicalIntersect', 
            'PhysicalLimit', 
            'PhysicalCTEProducer', 
            'PhysicalHashJoin', 
            'PhysicalUnion', 
            'PhysicalExcept', 
            'PhysicalPartitionTopN', 
            'PhysicalResultSink', 
            'PhysicalQuickSort', 
            'PhysicalDistribute', 
            'PhysicalFilter', 
            'PhysicalHashAggregate', 
            'PhysicalAssertNumRows', 
            'PhysicalCTEConsumer', 
            'PhysicalOlapScan', 
            'PhysicalCTEAnchor', 
            'PhysicalTopN', 
            'PhysicalProject', 
            'PhysicalRepeat', 
            'PhysicalWindow', 
            'PhysicalNestedLoopJoin',
            'PhysicalEmptyRelation',
            'PhysicalOneRowRelation',
            'PhysicalStorageLayerAggregate'
            }
nodetype2idx = {t:i for i, t in enumerate(NodeType)}

distributionSpec_list = {'DistributionSpecReplicated', 'DistributionSpecGather', 'DistributionSpecExecutionAny', 'DistributionSpecHash'}
distributionSpec2idx = {t:i for i, t in enumerate(distributionSpec_list)}

class Node:
    def __init__(self, node_id, name, id, order, attributes, cardinality, table, node_level):
        self.nodeid = node_id
        self.name = name
        self.id = id
        self.order = order # Note that the root node has order -1, because its order number is not explicitly given in the query plan
        self.attributes = attributes
        self.cardinality = cardinality
        self.table = table
        self.node_level = node_level
        self.children = []  # it's weird why we replace [] with children here, the code will run into problem
        # print(f"in Node,  name {name}, id {id}, order {order}, cardinality {cardinality}, node_level {node_level}")

        self.table = None
        self.columns = None
        self.limit = 0
        # if self.name == 'PhysicalOlapScan':
        #     self.extract_olap_scan(attributes)
        if self.name == 'PhysicalProject':
            self.extract_project(attributes)
        if self.name == 'PhysicalDistribute':
            self.extract_distribute(attributes)
        if self.name == 'PhysicalFilter':
            self.extract_filter(attributes)
        if self.name == 'PhysicalTopN':
            self.extract_topn(attributes)

        nodetype=nodetype2idx[self.name]
        card= int(float(self.cardinality))
        card = math.log1p(card)  # beware what the inverse function of log1p is
        # table_rows = get_table_rows(self.table) if self.table is not None else 0
        num_of_columns = len(self.columns) if self.columns is not None else 0
        limit = int(self.limit)

        # self.features = torch.tensor([nodetype, card, table_rows, num_of_columns, limit])
        self.features = [nodetype, card, num_of_columns, limit]
   
    def print_tree(self):
        print(' '*self.node_level + self.__str__())
        for child in self.children:
            child.print_tree()

    def print_children(self):
        print(f"{self.__str__()} has {len(self.children)} children: ",end='')
        for child in self.children:
            print(child, end=' ')
        print('[\\n]')

    def __str__(self):
        return_str = f"NodeId#{self.id}"
        # return_str += '__' + 'stats_' + str(self.cardinality)
        # if self.name == 'PhysicalOlapScan':
        #     return_str += '__' + 'database_' + self.database + '__' + 'table_' + self.table
        # if self.name == 'PhysicalProject':
        #     return_str += '__' + 'columns_(' + ','.join(self.columns) + ')'
        # if self.name == 'PhysicalDistribute':
        #     return_str += '__' + 'distributionSpec_' + self.distributionSpec
        # if self.name == 'PhysicalFilter':
        #     return_str += '__' + 'predicates_(' + self.predicates + ')'
        # if self.name == 'PhysicalTopN':
        #     return_str += '__' + 'limit_' + self.limit
        return_str += '_' + str(self.features)
        if self.nodeid is not None:
            return_str += "_" + str(self.nodeid)
        return return_str
    
    # def extract_olap_scan(self, attributes):
    #     olap_pattern = r"qualified=([A-Za-z0-9_]+\.[A-Za-z0-9_]+)"
    #     match=re.search(olap_pattern, attributes)

    #     if match:
    #         qualified = match.groups(0)
    #         self.database, self.table = qualified[0].split('.')
    #     else:
    #         raise ValueError(f"Unable to extract qualified from attributes: {attributes}")
        
    def extract_project(self, attributes):
        project_pattern = r"projects=\[([^\]]+)\]"
        match = re.search(project_pattern, attributes)

        if match:
            self.columns= match.groups(0)[0].split(',')
        else:
            raise ValueError(f"Unable to extract project columns from attributes: {attributes}")
        

    def extract_distribute(self, attributes):
        distribute_pattern = r"distributionSpec=([A-Za-z]+)"
        match = re.search(distribute_pattern, attributes)

        if match:
            self.distributionSpec = match.groups(0)[0]
        else:
            raise ValueError(f"Unable to extract distributionSpec from attributes: {attributes}")
        

    def extract_filter(self, attributes):
        text = attributes
        start_keyword = "predicates=("
        start_index = text.find(start_keyword)
        if start_index == -1:
            self.predicates = None

        start_index += len(start_keyword)
        open_parens = 1
        end_index = start_index

        while open_parens > 0 and end_index < len(text):
            if text[end_index] == '(':
                open_parens += 1
            elif text[end_index] == ')':
                open_parens -= 1
            end_index += 1

        if open_parens != 0:
            self.predicates = None

        self.predicates = text[start_index:end_index-1]

    def extract_topn(self, attributes):
        topn_pattern = r"limit=([0-9]+)"
        match = re.search(topn_pattern, attributes)

        if match:
            self.limit = match.groups(0)[0]
        else:
            raise ValueError(f"Unable to extract limit from attributes: {attributes}")
    
    # def get_label_for_each_node(self, node_id):
    #     self.nodeid = node_id
    #     node_id += 1
    #     for child in self.children:
    #         node_id = child.get_label_for_each_node(node_id)
    #     return node_id

def extract_stats(attributes):
    stats_pattern = r'stats=([\d,\.]+) ([\d]+)'
    match=re.search(stats_pattern, attributes)

    if match:
        return match.groups()[0].replace(',','') # e.g. return 150000 from 'stats=150,000 1' 
    # else:   # some node indeed does not have stats
    #     raise ValueError(f"Unable to extract stats from attributes: {attributes}")