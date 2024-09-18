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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class Node:
    def __init__(self, name, attributes, children=None):
        self.name = name  # e.g., PhysicalHashJoin
        self.attributes = attributes  # Dict of attributes like cost, stats, etc.
        self.children = children if children else []

    def add_child(self, child):
        self.children.append(child)

def parse_execution_plan(plan_str):
    lines = plan_str.strip().split('\n')
    root = None
    stack = []
    previous_indent = -1

    for line in lines:
        # Determine the indentation level
        indent_match = re.match(r'(\s*)(\+--|\|--)?', line)
        indent = len(indent_match.group(1)) if indent_match else 0

        # Extract node name and attributes
        node_match = re.match(r'\s*(\+--|\|--)?(\w+)\[(\d+)\]@(\d+)\s*\((.*?)\)', line)
        if node_match:
            node_type = node_match.group(2)
            node_id = node_match.group(3)
            node_at = node_match.group(4)
            attrs_str = node_match.group(5)

            # Parse attributes into a dictionary
            attrs = {}
            for attr in attrs_str.split(','):
                key_value = attr.strip().split('=', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    attrs[key.strip()] = value.strip()
                else:
                    attrs[key_value[0].strip()] = None

            node = Node(name=node_type, attributes=attrs)

            if indent > previous_indent:
                if stack:
                    stack[-1].add_child(node)
                stack.append(node)
            else:
                while indent <= previous_indent and stack:
                    stack.pop()
                    previous_indent -= 3  # Assuming indent increases by 3 spaces
                if stack:
                    stack[-1].add_child(node)
                stack.append(node)

            if root is None:
                root = node

            previous_indent = indent

    return root

def extract_features(node, features):
    if node is None:
        return

    # Initialize collectors if not already initialized
    if 'stats_values' not in features:
        features['stats_values'] = []
    if 'limit_values' not in features:
        features['limit_values'] = []
    if 'offset_values' not in features:
        features['offset_values'] = []

    # Feature: Count of each node type
    node_type = node.name
    features['count_' + node_type] = features.get('count_' + node_type, 0) + 1

    # Feature: Total cost
    cost = float(node.attributes.get('cost', '0').replace('E', 'e') if 'cost' in node.attributes else 0)
    features['total_cost'] = features.get('total_cost', 0) + cost

    # Feature: Total stats
    stats = node.attributes.get('stats', '0').replace(',', '').replace('.', '')
    try:
        stats_val = float(stats)
    except ValueError:
        stats_val = 0
    features['total_stats'] = features.get('total_stats', 0) + stats_val
    features['stats_values'].append(stats_val)

    # Feature: Limit
    limit = node.attributes.get('limit')
    if limit:
        try:
            limit_val = int(limit)
            features['total_limit'] = features.get('total_limit', 0) + limit_val
            features['limit_values'].append(limit_val)
        except ValueError:
            pass

    # Feature: Offset
    offset = node.attributes.get('offset')
    if offset:
        try:
            offset_val = int(offset)
            features['total_offset'] = features.get('total_offset', 0) + offset_val
            features['offset_values'].append(offset_val)
        except ValueError:
            pass

    # Recursively extract features from children
    for child in node.children:
        extract_features(child, features)

# Paths to data
test_dataset = 'tpch'
output_file_csv = f'/home/wuy/DB/doris_mem_pred/{test_dataset}_data/query_mem_data_{test_dataset}_sf100.csv'
output_plan_dir = f"/home/wuy/DB/doris_mem_pred/{test_dataset}_data/plans"

# Read CSV, skipping specific rows
data = pd.read_csv(output_file_csv, sep=';', skiprows=lambda x: x in [978, 1968, 2202, 3129, 4259])
print(f"processing {len(data)} queries")

# Initialize lists to store Data objects
data_list = []

# Step 1: Collect all features from the data to fit encoders
feature_list = []
mem_list_all = []
print("Collecting features for encoder fitting...")

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

    tree = parse_execution_plan(plan)
    features = {}

    # Extract features from the tree
    extract_features(tree, features)

    # Compute min, max, mean for stats_values
    if features['stats_values']:
        features['stats_min'] = min(features['stats_values'])
        features['stats_max'] = max(features['stats_values'])
        features['stats_mean'] = sum(features['stats_values']) / len(features['stats_values'])
    else:
        features['stats_min'] = 0
        features['stats_max'] = 0
        features['stats_mean'] = 0
    del features['stats_values']  # Remove the list to keep the features clean

    # Compute min, max, mean for limit_values
    if features['limit_values']:
        features['limit_min'] = min(features['limit_values'])
        features['limit_max'] = max(features['limit_values'])
        features['limit_mean'] = sum(features['limit_values']) / len(features['limit_values'])
    else:
        features['limit_min'] = 0
        features['limit_max'] = 0
        features['limit_mean'] = 0
    del features['limit_values']

    # Compute min, max, mean for offset_values
    if features['offset_values']:
        features['offset_min'] = min(features['offset_values'])
        features['offset_max'] = max(features['offset_values'])
        features['offset_mean'] = sum(features['offset_values']) / len(features['offset_values'])
    else:
        features['offset_min'] = 0
        features['offset_max'] = 0
        features['offset_mean'] = 0
    del features['offset_values']

    feature_list.append(features)
    max_mem = max(mem_list)
    mem_list_all.append(max_mem)

# Create a DataFrame from the features
df_features = pd.DataFrame(feature_list).fillna(0)

# Add the target variable
df_features['peak_memory'] = mem_list_all

# Prepare feature matrix X and target vector y
X = df_features.drop('peak_memory', axis=1)
y = df_features['peak_memory']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor
# model = XGBRegressor(objective='reg:squarederror', n_estimators=80, random_state=42)

# model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# # save the model as json
# with open('xgboost_model.json', 'w') as f:
#     model.get_booster().save_model(f'{os.getcwd()}/xgboost_model.json')

# Load the model from json
model = XGBRegressor()
dataset = 'tpcds'
model.load_model(f'{os.getcwd()}/{dataset}_xgboost_model.json')

model_feature_names = model.get_booster().feature_names
test_feature_names = X_test.columns.tolist()

# Find the columns that are in the model but not in the test data
missing_cols = [col for col in model_feature_names if col not in test_feature_names]

# Add missing columns to the test dataset with default values (e.g., 0 for numeric columns)
for col in missing_cols:
    X_test[col] = 0

# Find the columns that are in the test data but not in the model
extra_cols = [col for col in test_feature_names if col not in model_feature_names]

# Drop the extra columns from the test dataset
X_test = X_test.drop(columns=extra_cols)
X_test = X_test[model_feature_names]

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")



# Function to calculate q-error
def qerror(y_true, y_pred):
    """
    Calculate the median q-error.
    q-error = max(y_pred / y_true, y_true / y_pred)
    """
    epsilon = 1e-10
    y_pred_safe = np.where(y_pred <= 0, epsilon, y_pred)
    y_true_safe = np.where(y_true <= 0, epsilon, y_true)
    qerror_values = np.maximum(y_pred_safe / y_true_safe, y_true_safe / y_pred_safe)
    return np.percentile(qerror_values, 50)

print(f"Median q-error on test set: {qerror(y_test, y_pred)}")
