from dataloader import PlanDataset, plan2graph
from TreeLSTM import PlanNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

def Qerror(output, memory): # meandian Q-Error
    count = 0 
    for i in range(len(output)):
        count += max(output[i], memory[i]) / (min(output[i], memory[i]) + 1e-10 )
    return count/len(output)

dataset = PlanDataset(output_file_csv=f'/home/wuy/DB/doris_mem_pred/tpch_data/query_mem_data_tpch_sf100.csv',
                    output_plan_dir=f"/home/wuy/DB/doris_mem_pred/tpch_data/plans"
                    )

test_dataset = dataset
print(f"test_dataset len {len(test_dataset)}")

batch_size = 5120
num_epochs = 1000
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
testset = test_dataset

x_size = 4
h_size = 4
dropout = 0.5
model = PlanNet(x_size, h_size, dropout)

model.load_state_dict(torch.load('plan_net.pth'))

# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

loss_fn = nn.MSELoss()

# print('Epoch ', epoch, 'Mean Loss: ', epoch_loss/len(trainset))
with torch.no_grad():
    model.eval()

    val_epoch_loss = 0.0
    for i, batch in enumerate(testset):  
        g, cost, memory, root_node_indexes = batch
        n=g.num_nodes()
        h = torch.zeros((n,h_size))  
        c = torch.zeros(n,h_size)
        # cost = torch.FloatTensor([cost])
        # memory = torch.FloatTensor([memory])
        g = g.to(device)
        cost = cost.to(device)
        memory = memory.to(device)
        h = h.to(device)
        c = c.to(device)

        output = model(g, g.ndata['feat'], h, c, cost, root_node_indexes)
        loss = loss_fn(output, memory)
        qerror = Qerror(output, memory)
        val_epoch_loss += loss.item()
    print(' Validation Loss: ', val_epoch_loss/len(testset), 'QError: ', qerror)



