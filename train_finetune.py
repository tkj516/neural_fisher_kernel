import os

import torch
import torch.nn as nn
import torch.optim as optim

def train_finetune(configs, model, train_dataloader, val_dataloader, saving_path, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=configs.lr, momentum=0.9)
    for epoch in range(configs.total_iterations):
        epoch_loss = 0.0
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                epoch_loss += running_loss
                running_loss = 0.0

        if epoch % configs.saving_frequency == 0:
            checkpoint_name = "-".join(["checkpoint", str(epoch) + ".pt"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": epoch_loss,
                },
                os.path.join(saving_path, checkpoint_name),
            )

        # Calcualate validation accuracy
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                images, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on validation set at epoch %d: %d %%" % (epoch, 100 * correct / total))





