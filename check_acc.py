import torch

def check_acc(loader, model, device):
    '''
    Accuracy check for given loader (train or val) and model.
    input: loader - train or val loader; model - model trained; device - cuda or cpu).
    Fuction returns accuracy value.
    '''
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = float(num_correct)/float(num_samples)*100
        print(f'For {num_correct} / {num_samples} with acc {acc:.2f}')
    return round(acc,2)


