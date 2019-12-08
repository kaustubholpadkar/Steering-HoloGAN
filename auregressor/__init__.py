from __future__ import print_function, division
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
from auregressor.audataset import ActionUnitDataset


class ActionUnitRegressor:

    def __init__(self, csv_dir, img_dir, lr=0.001, momentum=0.9, batch_size=128):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 17)
        )
        self.last_epoch = 0
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.dataset = ActionUnitDataset(csv_dir=csv_dir, img_dir=img_dir)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def train_model(self, num_epochs=25):
        since = time.time()

        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            self.model.train()  # Set model to training mode

            running_loss = 0.0

            # Iterate over data.
            for idx, batch in enumerate(self.dataloader):
                inputs = batch["image"]
                labels = batch["action_units"]

                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.float)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            self.scheduler.step(epoch)

            epoch_loss = running_loss / len(self.dataset)

            print('Loss: {:.4f}'.format(epoch_loss))
            print()

        self.last_epoch = epoch
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def save(self, filename, epoch, legacy=False):
        dd = dict()
        dd['model'] = self.model.state_dict()
        dd['optimizer'] = self.optimizer.state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename, legacy=False, ignore_d=False):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        dd = torch.load(filename, map_location=map_location)
        self.model.load_state_dict(dd['model'])
        self.optimizer.load_state_dict(dd['optimizer'])
        self.last_epoch = dd['epoch']
