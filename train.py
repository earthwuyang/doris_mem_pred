from dataloader import PlanDataset, plan2graph
from TreeLSTM import PlanNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

dataset = PlanDataset()
dataset.process()  

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(1)
)
print(f"train_dataset len {len(train_dataset)}")
print(f"test_dataset len {len(test_dataset)}")

batch_size = 16
num_epochs = 1000
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
trainset = train_dataset
testset = test_dataset

x_size = 4
h_size = 4
dropout = 0.5
model = PlanNet(x_size, h_size, dropout)

optimizer = optim.SGD(model.parameters(), lr=0.001)


loss_fn = nn.MSELoss()

print(f"training begins...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for i, batch in enumerate(trainset):
        # print(f"epoch {epoch}, iter {i}")
        g, cost, memory = batch
        
        n=g.num_nodes()
        h = torch.zeros((n,h_size))
        c = torch.zeros(n,h_size)
        # cost = torch.FloatTensor([cost])
        # memory = torch.FloatTensor([memory])
        output = model(g, g.ndata['feat'], h, c, cost)
        
        optimizer.zero_grad()
        loss = loss_fn(output, memory)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i % 1 == 0:
            print('Epoch: ', epoch, 'Iter: ', i, 'Loss: ', loss.item())

    # print('Epoch ', epoch, 'Mean Loss: ', epoch_loss/len(trainset))
    with torch.no_grad():
        model.eval()

        val_epoch_loss = 0.0
        for i, batch in enumerate(testset):  
            g, cost, memory = batch
            n=g.num_nodes()
            h = torch.zeros((n,h_size))  
            c = torch.zeros(n,h_size)
            cost = torch.FloatTensor([cost])
            memory = torch.FloatTensor([memory])
            output = model(g, g.ndata['feat'], h, c, cost)
            loss = nn.MSELoss()(output, memory)
            val_epoch_loss += loss.item()
        print('Epoch ', epoch, 'Train Loss: ', epoch_loss/len(trainset), ' Validation Loss: ', val_epoch_loss/len(testset))
    torch.save(model.state_dict(), 'plan_net.pth')


torch.save(model.state_dict(), 'plan_net.pth')


