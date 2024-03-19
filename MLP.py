import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import classification_report



class EmbeddingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        embedding = self.data.iloc[idx, 1:].to_numpy(dtype='float32')
        return embedding, label


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def main():
    # Define the paths to the train and test files
    train_csv_file = 'dataset/imdb_pooling/train.csv'
    test_csv_file = 'dataset/imdb_pooling/test.csv'

    # Instantiate the datasets and dataloaders
    train_dataset = EmbeddingDataset(train_csv_file)
    test_dataset = EmbeddingDataset(test_csv_file)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define your input size and number of classes
    input_size = 1024
    num_classes = 2

    # Instantiate the MLP model
    model = MLP(input_size, num_classes)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Create TensorBoard writer
    current_time = datetime.now().strftime("%b%d-%H-%M-%S")
    log_dir = f"./output/MLP_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    num_epochs = 5
    steps = 0
    # Training loop
    for epoch in tqdm(range(num_epochs)):  # Adjust the number of epochs as needed
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.long()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps += 1
            writer.add_scalar('Running Loss', loss.item(), steps)

        # Calculate validation metrics
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(test_dataloader):
                outputs = model(inputs)
                targets = targets.long()
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.tolist())
                val_targets.extend(targets.tolist())

        # Calculate metrics
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds)
        val_f1score = f1_score(val_targets, val_preds)

        # Log metrics to TensorBoard
        writer.add_scalar('Train Loss', train_loss /
                          len(train_dataloader), epoch)
        writer.add_scalar('Val Loss', val_loss / len(test_dataloader), epoch)
        writer.add_scalar('Val Accuracy', val_accuracy, epoch)
        writer.add_scalar('Val Precision', val_precision, epoch)
        writer.add_scalar('Val F1-Score', val_f1score, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss / len(train_dataloader)}, Val Loss: {val_loss / len(test_dataloader)}, Val Accuracy: {val_accuracy}, Val Precision: {val_precision}, Val F1-Score: {val_f1score}')


    # Close the TensorBoard writer
    writer.close()
    
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader):
            outputs = model(inputs)
            targets = targets.long()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_targets.extend(targets.tolist())
            
    report = classification_report(val_targets, val_preds)
    print(report)


if __name__ == "__main__":
    main()
