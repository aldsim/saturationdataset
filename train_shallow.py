import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from readdataset import ALDDataset_N
from aldnets import ALDNet0


def train(model, train_loader, optimizer, epoch):
    loss_fn = nn.MSELoss()

    model.train()
    for batch_idx, (profile, tdose, tsat) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(profile, tdose)
        loss = loss_fn(output, tsat)
        loss.backward()
        optimizer.step()
#        if batch_idx % 5 == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(profile), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_dataset, epoch):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        profile, tdose, tsat = ald_test[:]
        output = model(profile, tdose)
        test_loss += loss_fn(output, tsat).item()
        test_loss /= len(test_dataset)

    tsat = tsat.detach().numpy()[:,0]
    tpred = output.detach().numpy()[:,0]
    tdose = tdose.detach().numpy()[:,0]
    tdose = np.exp(tdose)
    tpred = np.exp(tpred)
    tsat = np.exp(tsat)
    ep = (tpred-tsat)/tsat
    corr = np.corrcoef(tsat, tpred)[0,1]
    ep = (tpred-tsat)/tsat
    m = np.mean(ep)
    s = np.std(ep)
    print('Epoch {}: Average loss: {:.6f}, Mean: {:.4f}, Std: {:.4f}'.format(epoch, test_loss,
        m, s))
    return test_loss, corr, m, s, tpred, tsat, tdose


if __name__ == "__main__":


    Nlist = [4, 5, 8, 10, 16, 20]

    data = []

    for N in Nlist:

        ald_test = ALDDataset_N(N, train=False)
        ald_train = ALDDataset_N(N, train=True)

        train_dataloader = DataLoader(ald_train, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(ald_test, batch_size=64, shuffle=True)

        ac = ALDNet0(N)

        optimizer = optim.Adam(ac.parameters(), lr=1e-3)
        for epoch in range(1, 101):
            train(ac, train_dataloader, optimizer, epoch)
            loss, corr, m, s, tpred, tsat, tdose = test(ac, ald_test, epoch)
            data.append([N, epoch, loss, corr, m, s])

        predfile = "sat_predicted_%d.npy" % N
        np.save(predfile, tpred)
    
    np.save("point_series.npy", np.array(data))

