# This project was created for educational, non commercial use.
# https://github.com/hvkat/music-genres-classificator

import torch
import torch.nn as nn
import torch.optim as optim
from model import model_cmg_fc1, model_cmg_fc2, check_acc
from data import prepare_data
import yaml
import os

# Open config file
config_path = 'D:/git-testy/classifier1/config/'

with open(os.path.join(config_path,'config.yaml')) as c:
    configs = yaml.safe_load(c)

# Set device
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Determine model used
model_type = configs["model"]
if model_type == "fc1":
    model = model_cmg_fc1.ClassifierMusicGenres(configs["input_size"], configs["num_classes"]).to(device)
    output_path = os.path.join(configs["output_path_1"])
elif model_type == "fc2":
    model = model_cmg_fc2.ClassifierMusicGenres(configs["input_size"], configs["num_classes"]).to(device)
    output_path = os.path.join(configs["output_path_2"])

# Create files for writing down training progress
avg_train_loss_per_epoch = open(os.path.join(output_path,'avg_train_loss_per_epoch.txt'), 'w')
avg_val_loss_per_epoch = open(os.path.join(output_path,'avg_val_loss_per_epoch.txt'), 'w')

# Get dataloaders
train_loader = prepare_data.train_loader
val_loader = prepare_data.val_loader

# Determine loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=configs["learning_rate"])

# Training loop
for epoch in range(configs["num_epochs"]):
    model.train()
    train_loss=[]
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0],-1)
        scores = model(data)
        loss = criterion(scores, targets)
        train_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = torch.stack(train_loss).mean().item()
    avg_train_loss_per_epoch.write(str(avg_train_loss)+',')
    print(f'Mean train loss: {avg_train_loss:.5f} for epoch {epoch}')
    model.eval()
    with torch.no_grad():
        val_loss=[]
        for batch_idx, (data,targets) in enumerate(val_loader):
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0],-1)
            scores = model(data)
            loss = criterion(scores,targets)
            val_loss.append(loss)
        avg_val_loss = torch.stack(val_loss).mean().item()
        avg_val_loss_per_epoch.write(str(avg_val_loss)+',')
        print(f'Mean val loss: {avg_val_loss:.5f} for epoch {epoch}')

# Save the model
torch.save(model,os.path.join(output_path,'model_output.pth'))

# Check and write down accuracy
print(f'Training data accuracy:')
avg_train_loss_per_epoch.write(str(check_acc.check_acc(train_loader,model,device))+',')
print(f'Val data accuracy:')
avg_val_loss_per_epoch.write(str(check_acc.check_acc(val_loader,model,device))+',')

avg_train_loss_per_epoch.close()
avg_val_loss_per_epoch.close()


