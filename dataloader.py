import torch
from tqdm import tqdm
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import re
import os
from torch_geometric.data import Data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# Define the PlanNode class
class PlanNode:
    def __init__(self, name: str, properties: Dict[str, str], depth: int):
        self.name = name  # e.g., PhysicalHashJoin
        self.properties = properties  # Dictionary of properties
        self.children: List['PlanNode'] = []
        self.depth = depth  # Depth in the tree
        self.parent: Optional['PlanNode'] = None  # Reference to parent node

    def add_child(self, child: 'PlanNode'):
        self.children.append(child)
        child.parent = self

# Function to parse the execution plan into a tree of PlanNodes
def parse_execution_plan(plan_str: str) -> PlanNode:
    lines = plan_str.strip().split('\n')
    root = None
    stack = []  # Stack to keep track of parent nodes

    for line in lines:
        # Determine the depth based on the number of leading spaces and symbols
        # Each level is indicated by "+--" or "|--"
        match = re.match(r'^(\s*)(\+--|\|--)?\s*(.+)', line)
        if match:
            indent, connector, content = match.groups()
            # Each level of indentation (e.g., 4 spaces) increases depth by 1
            depth = len(indent.replace('|', ' ').replace('+', ' ').replace('-', ' ')) // 4
            # Parse the node name and properties
            node_match = re.match(r'^(\w+)\[(\d+)\](?:@(\d+))?\s*\((.*)\)', content)
            if node_match:
                node_type, node_id, at_number, props_str = node_match.groups()
                node_name = node_type
                properties = {}
                # Split properties by comma, but handle nested brackets
                props = split_properties(props_str)
                for prop in props:
                    key, value = parse_property(prop)
                    properties[key] = value
                node = PlanNode(name=node_name, properties=properties, depth=depth)
                # Assign parent based on depth
                if depth == 0:
                    root = node
                    stack = [node]
                else:
                    # The parent is the last node in the stack with depth = current depth -1
                    while stack and stack[-1].depth >= depth:
                        stack.pop()
                    if stack:
                        stack[-1].add_child(node)
                    stack.append(node)
            else:
                # Handle lines without properties, if any
                node_name = content.strip()
                node = PlanNode(name=node_name, properties={}, depth=depth)
                if depth == 0:
                    root = node
                    stack = [node]
                else:
                    while stack and stack[-1].depth >= depth:
                        stack.pop()
                    if stack:
                        stack[-1].add_child(node)
                    stack.append(node)
    return root

def split_properties(props_str: str) -> List[str]:
    """
    Split the properties string into individual properties, handling nested brackets.
    """
    props = []
    current = ''
    bracket_level = 0
    for char in props_str:
        if char == '(' or char == '[':
            bracket_level += 1
        elif char == ')' or char == ']':
            bracket_level -= 1
        if char == ',' and bracket_level == 0:
            props.append(current.strip())
            current = ''
        else:
            current += char
    if current:
        props.append(current.strip())
    return props

def parse_property(prop: str) -> (str, str):
    """
    Parse a single property into key and value.
    """
    if '=' in prop:
        key, value = prop.split('=', 1)
        return key.strip(), value.strip()
    elif ':' in prop:
        key, value = prop.split(':', 1)
        return key.strip(), value.strip()
    else:
        return prop.strip(), ''

# Function to traverse the tree and collect all nodes
def traverse_tree(root: PlanNode) -> List[PlanNode]:
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        # Add children to stack in reverse order to maintain original order
        stack.extend(reversed(node.children))
    return nodes

# Function to extract features from nodes
def extract_features(nodes: List[PlanNode]) -> pd.DataFrame:
    feature_list = []
    for node in nodes:
        features = {}
        # Categorical Features
        features['node_type'] = node.name
        # Extract additional categorical features based on node properties
        # Example: for join nodes, extract join type
        if node.name == 'PhysicalHashJoin':
            features['join_type'] = node.properties.get('type', 'UNKNOWN')
        else:
            features['join_type'] = 'NONE'
        
        # Distribution Spec Type
        distribution_spec = node.properties.get('distributionSpec', 'NONE')
        if 'DistributionSpecHash' in distribution_spec:
            features['distribution_spec'] = 'DistributionSpecHash'
        elif 'DistributionSpecReplicated' in distribution_spec:
            features['distribution_spec'] = 'DistributionSpecReplicated'
        elif 'DistributionSpecGather' in distribution_spec:
            features['distribution_spec'] = 'DistributionSpecGather'
        else:
            features['distribution_spec'] = 'OTHER'
        
        # Aggregate Phase
        agg_phase = node.properties.get('aggPhase', 'NONE')
        features['agg_phase'] = agg_phase
        
        # Aggregate Mode
        agg_mode = node.properties.get('aggMode', 'NONE')
        features['agg_mode'] = agg_mode
        
        # Other Categorical Attributes
        maybe_use_streaming = node.properties.get('maybeUseStreaming', 'false')
        features['maybe_use_streaming'] = maybe_use_streaming
        
        topn_opt = node.properties.get('topnOpt', 'false')
        features['topn_opt'] = topn_opt
        
        # Numerical Features
        # Extract numerical values, handling commas and scientific notation
        cost = node.properties.get('cost', '0')
        features['cost'] = parse_numeric(cost)
        
        limit = node.properties.get('limit', '0')
        features['limit'] = parse_numeric(limit)
        
        offset = node.properties.get('offset', '0')
        features['offset'] = parse_numeric(offset)
        
        stats = node.properties.get('stats', '0')
        # Stats might be two numbers separated by comma
        stats_val = parse_stats(stats)
        features['stats_1'] = stats_val[0]
        features['stats_2'] = stats_val[1]
        
        # Runtime Filters
        runtime_filters = node.properties.get('runtimeFilters', '0')
        features['runtime_filters'] = parse_runtime_filters(runtime_filters)
        
        feature_list.append(features)
    df = pd.DataFrame(feature_list)
    return df

def parse_numeric(value: str) -> float:
    """
    Parse a numerical value from string, handling scientific notation and commas.
    """
    value = value.replace(',', '').replace('E', 'e')
    try:
        return float(value)
    except:
        return 0.0

def parse_stats(stats: str) -> (float, float):
    """
    Parse the stats property which might contain two numerical values separated by a comma.
    """
    parts = stats.split(',')
    if len(parts) == 2:
        return float(parse_numeric(parts[0])), float(parse_numeric(parts[1]))
    elif len(parts) == 1:
        return float(parse_numeric(parts[0])), 0.0
    else:
        return 0.0, 0.0

def parse_runtime_filters(rf_str: str) -> int:
    """
    Parse the runtimeFilters property to count the number of runtime filters.
    """
    if 'RF' in rf_str:
        return rf_str.count('RF')
    else:
        return 0

# Function to encode categorical features and normalize numerical features
def encode_features(df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler) -> torch.Tensor:
    # Categorical Columns
    categorical_cols = ['node_type', 'join_type', 'distribution_spec', 'agg_phase', 'agg_mode', 'maybe_use_streaming', 'topn_opt']
    categorical_encoded = encoder.transform(df[categorical_cols])
    
    # Numerical Columns
    numerical_cols = ['cost', 'limit', 'offset', 'stats_1', 'stats_2', 'runtime_filters']
    numerical_scaled = scaler.transform(df[numerical_cols])
    
    # Combine Features
    combined_features = np.hstack([categorical_encoded, numerical_scaled])
    
    # Convert to PyTorch tensor
    x = torch.tensor(combined_features, dtype=torch.float)
    return x

# Function to create edge_index for PyTorch Geometric
def create_edge_index(root: PlanNode) -> torch.Tensor:
    nodes = traverse_tree(root)
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    edge_index = []
    for parent in nodes:
        parent_idx = node_indices[parent]
        for child in parent.children:
            child_idx = node_indices[child]
            edge_index.append([parent_idx, child_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# Paths to data
output_file_csv = '/home/wuy/DB/doris_mem_pred/tpch_data/query_mem_data_tpch_sf100.csv'
output_plan_dir = "/home/wuy/DB/doris_mem_pred/tpch_data/plans"

# Read CSV, skipping specific rows
data = pd.read_csv(output_file_csv, sep=';', skiprows=lambda x: x in [978, 1968, 2202, 3129, 4259])
print(f"processing {len(data)} queries")

# Initialize lists to store Data objects
data_list = []

# Step 1: Collect all features from the first 10 rows to fit encoders
all_feature_dfs = []
mem_list_all = []

print("Collecting features for encoder fitting...")
# for line in tqdm(data.head(10).itertuples(index=False), total=10):
for line in tqdm(data.itertuples(index=False), total=len(data)):
    queryid = line.queryid
    time = float(line.time)
    mem_list = eval(line.mem_list)
    stmt = line.stmt
    plan_file = os.path.join(output_plan_dir, f'{queryid}.txt')
    
    if not os.path.exists(plan_file):
        print(f"Plan file {plan_file} does not exist. Skipping.")
        continue
    
    with open(plan_file, 'r') as f:
        plan = f.read()
    plan = plan.strip()

    # Extract cost
    try:
        cost = float(plan.split('\n', 1)[0].split(' = ')[1].strip())
    except (IndexError, ValueError):
        print(f"Failed to extract cost for query {queryid}. Skipping.")
        continue

    if cost == 0.0:
        continue
    try:
        plan = plan.split('\n', 1)[1].strip()
    except IndexError:
        print(f"Plan for query {queryid} does not contain node information. Skipping.")
        continue

    # Parse the execution plan
    root = parse_execution_plan(plan)
    if root is None:
        print(f"Failed to parse the execution plan for query {queryid}. Skipping.")
        continue

    # Traverse the tree and extract features
    nodes = traverse_tree(root)
    df_features = extract_features(nodes)
    all_feature_dfs.append(df_features)
    mem_list_all.append(max(mem_list))

# Combine all features into a single DataFrame
combined_features = pd.concat(all_feature_dfs, ignore_index=True)

# Initialize and fit the encoders
categorical_cols = ['node_type', 'join_type', 'distribution_spec', 'agg_phase', 'agg_mode', 'maybe_use_streaming', 'topn_opt']
numerical_cols = ['cost', 'limit', 'offset', 'stats_1', 'stats_2', 'runtime_filters']

print("Fitting OneHotEncoder and StandardScaler...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(combined_features[categorical_cols])

scaler = StandardScaler()
scaler.fit(combined_features[numerical_cols])

# Step 2: Iterate over the first 10 rows again to create Data objects
print("Creating Data objects with consistent feature dimensions...")
for line in tqdm(data.head(10).iterrows(), total=10):
    queryid = line[1]['queryid']
    time = float(line[1]['time'])
    mem_list = eval(line[1]['mem_list'])
    stmt = line[1]['stmt']
    plan_file = os.path.join(output_plan_dir, f'{queryid}.txt')
    
    if not os.path.exists(plan_file):
        print(f"Plan file {plan_file} does not exist. Skipping.")
        continue

    with open(plan_file, 'r') as f:
        plan = f.read()
    plan = plan.strip()

    # Extract cost
    try:
        cost = float(plan.split('\n', 1)[0].split(' = ')[1].strip())
    except (IndexError, ValueError):
        print(f"Failed to extract cost for query {queryid}. Skipping.")
        continue

    if cost == 0.0:
        continue
    try:
        plan = plan.split('\n', 1)[1].strip()
    except IndexError:
        print(f"Plan for query {queryid} does not contain node information. Skipping.")
        continue

    # Parse the execution plan
    root = parse_execution_plan(plan)
    if root is None:
        print(f"Failed to parse the execution plan for query {queryid}. Skipping.")
        continue

    # Traverse the tree and extract features
    nodes = traverse_tree(root)
    df_features = extract_features(nodes)

    # Transform features using the pre-fitted encoders
    categorical_encoded = encoder.transform(df_features[categorical_cols])
    numerical_scaled = scaler.transform(df_features[numerical_cols])

    # Combine encoded categorical and scaled numerical features
    combined_features = np.hstack([categorical_encoded, numerical_scaled])

    # Convert to PyTorch tensor
    x = torch.tensor(combined_features, dtype=torch.float)

    # Create edge_index tensor
    edge_index = create_edge_index(root)

    # Get the target value (max memory usage)
    max_mem = max(mem_list)

    # Create Data object
    data_obj = Data(x=x, edge_index=edge_index, y=torch.tensor([max_mem], dtype=torch.float))
    data_list.append(data_obj)

# Create a simple Dataset and DataLoader
from torch_geometric.loader import DataLoader

class ExecutionPlanDataset(torch.utils.data.Dataset):
    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

dataset = ExecutionPlanDataset(data_list)
loader = DataLoader(dataset, batch_size=2, shuffle=True)