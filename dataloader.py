from my_parse_v2 import parse_tree, input_str
import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd



def plan2graph(batch):
    graphs = []
    costs = []
    labels = []
    for edge_src_nodes, edge_tgt_nodes, features_list, cost, label in batch:
        g=dgl.graph((edge_src_nodes, edge_tgt_nodes))
        g.ndata['feat'] = torch.tensor(features_list)
        graphs.append(g)
        costs.append(cost)
        labels.append(label)

    root_node_indexes = [0]
    for g in graphs[:-1]:
        root_node_indexes.append(g.number_of_nodes())
    graphs = dgl.batch(graphs)

    costs = torch.FloatTensor(costs)
    labels = torch.FloatTensor(labels)
    return graphs, costs, labels, root_node_indexes

class PlanDataset(DGLDataset):
    def __init__(self, output_file_csv, output_plan_dir):
        self.output_file_csv = output_file_csv
        self.output_plan_dir = output_plan_dir
        super().__init__(name='plan')
        


    def process(self):
        self.graphs = []
        self.edge_src_nodes = []
        self.edge_tgt_nodes = []

        self.features_list = []

        self.labels = []
        self.costs = []
        

        # data_file='data/query_mem_data_tpch_sf100.csv'

        data_file = self.output_file_csv
        # data=pd.read_csv(data_file, sep=';', on_bad_lines='warn')
        data=pd.read_csv(data_file, sep=';', skiprows= lambda x: x in [978, 1968, 2202, 3129, 4259])
        print(f"processing {len(data)} queries")
        for line in data.iterrows():
            queryid = line[1]['queryid']
            time = float(line[1]['time'])
            mem_list = eval(line[1]['mem_list'])
            stmt = line[1]['stmt']
            with open(os.path.join(self.output_plan_dir, f'{queryid}.txt'), 'r') as f:
                plan = f.read()
            plan=plan.strip()

            
            cost = float(plan.split('\n',1)[0].split(' = ')[1].strip())
            if cost == 0.0:
                continue
            plan = plan.split('\n',1)[1].strip()
            # print(queryid, time, mem_list, stmt)
            input_str = plan
            # print(input_str)
            root_node, edge_src_nodes, edge_tgt_nodes, features_list = parse_tree(input_str)

            self.edge_src_nodes.append(edge_src_nodes)
            self.edge_tgt_nodes.append(edge_tgt_nodes)
            self.features_list.append(features_list)
            max_mem = max(mem_list)
            log_max_mem = torch.log(torch.tensor(max_mem))
            self.labels.append(log_max_mem)
            log_cost = torch.log(torch.tensor(cost))
            self.costs.append(log_cost)
        print(f"data processing finished")

    def __getitem__(self, i):
        return self.edge_src_nodes[i], self.edge_tgt_nodes[i], self.features_list[i], self.costs[i], self.labels[i]
    
    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    dataset = PlanDataset()
    dataset.process()   

    print("length", len(dataset))
    # print(dataset[0])