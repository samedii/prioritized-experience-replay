from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torchvision import datasets, transforms
import tensorboardX
import numpy as np

from network import Net

def train(model, device, training_dataset, optimizer, epoch, log_interval, n_batches, batch_size, sample_weight_logit):
    model.train()

    probs = F.softmax(sample_weight_logit, dim=-1)
    batch_indices = torch.multinomial(probs, n_batches*batch_size, replacement=True).view(n_batches, batch_size)
    importance_sampling_weights = 1/(probs[batch_indices]*len(training_dataset))
    importance_sampling_weights /= importance_sampling_weights.sum(dim=1, keepdim=True)
    importance_sampling_weights = importance_sampling_weights.to(device)

    for batch_idx in range(n_batches):
        data, target = training_dataset[batch_indices[batch_idx]]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = -dist.Categorical(logits=output).log_prob(target)*importance_sampling_weights[batch_idx]
        loss = loss.mean()
        # loss = F.nll_loss(output, target, importance_sampling_weights[batch_idx].view(-1, 1).repeat(1, 10))
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return {
        'loss': test_loss,
        'accuracy': correct / len(test_loader.dataset)
    }


def main():
    # Training settings

    batch_size = 64
    training_dataset_size = 500
    n_batches = int(np.ceil(training_dataset_size/batch_size))
    print('n_batches', n_batches)
    test_batch_size = 1000
    epochs = 100
    lr = 0.001
    use_cuda = True
    seed = 1
    log_interval = 100

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # decrease size of training dataset
    training_dataset = datasets.MNIST('data', train=True, download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ]))
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=test_batch_size, shuffle=True)
    training_dataset = torch.utils.data.TensorDataset(*next(iter(train_loader)))

    # print(training_dataset[0:2])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sample_weight_logit = torch.ones(training_dataset_size)

    tb = tensorboardX.SummaryWriter()
    for epoch in range(1, epochs + 1):
        train(model, device, training_dataset, optimizer, epoch, log_interval, n_batches, batch_size, sample_weight_logit)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            data, target = training_dataset[:]
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='none')

            alpha = 1
            sample_weight_logit = loss*alpha
            # clip logits for stability
            percentiles = np.percentile(sample_weight_logit, [10, 90])
            sample_weight_logit = torch.clamp(sample_weight_logit, percentiles[0], percentiles[1])

        training_results = test(model, device, train_loader, epoch)
        test_results = test(model, device, test_loader, epoch)

        tb.add_scalars('loss', {
            'training': training_results['loss'],
            'test': test_results['loss'],
        }, epoch)

        tb.add_scalars('accuracy', {
            'training': training_results['accuracy'],
            'test': test_results['accuracy'],
        }, epoch)

        print(f'Epoch: {epoch}/{epochs}')


#if __name__ == '__main__':
main()
