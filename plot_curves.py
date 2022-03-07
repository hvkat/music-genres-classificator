from matplotlib import pyplot as plt
import yaml
import os

# Determine the path to file with written down loss and accuracy data
config_path = 'D:/git-testy/classifier1/config/'

with open(os.path.join(config_path,'config.yaml')) as c:
    configs = yaml.safe_load(c)

model_type = configs["model"]
if model_type == "fc1":
    output_path = os.path.join(configs["output_path_1"])
    title = 'Learning curves - model: Fully Connected 1'
elif model_type == "fc2":
    output_path = os.path.join(configs["output_path_2"])
    title = 'Learning curves - model: Fully Connected 2'

# Open files and get saved data
with open(os.path.join(output_path,'avg_train_loss_per_epoch.txt')) as t:
    trains = [float(f) for line in t for f in line.split(',') if f.strip()]
    train_acc = trains[-1]
    trains = trains[:-1]

with open(os.path.join(output_path,'avg_val_loss_per_epoch.txt')) as v:
    vals = [float(f) for line in v for f in line.split(',') if f.strip()]
    val_acc = vals[-1]
    vals = vals[:-1]

t.close()
v.close()

# Present loss and acc data
print(f'Training accuracy: {train_acc}\nValidation accuracy: {val_acc}.')

plt.plot(range(configs["num_epochs"]),trains,label='trains')
plt.plot(range(configs["num_epochs"]),vals,label='vals')
plt.title(title)
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.legend()
plt.show()
