import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
import argparse
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import classification_report


class EmbeddingDataset(Dataset):
    def __init__(self, csv_file, unsupervised=False):
        self.data = pd.read_csv(csv_file)
        self.unsupervised = unsupervised

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.unsupervised:
            embedding = self.data.iloc[idx].to_numpy(dtype='float32')
            return embedding
        else:
            label = self.data.iloc[idx, 0]
            embedding = self.data.iloc[idx, 1:].to_numpy(dtype='float32')
            return embedding, label


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size*0.25))
        self.fc2 = nn.Linear(int(input_size*0.25), num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def main(arg=None):
    parser = argparse.ArgumentParser(
        description="Pipeline for MLP classifier"
    )
    parser.add_argument(
        '--save', help='Whether to save the model', action="store_true"
    )
    parser.add_argument(
        '--model_dir', help="Directory for saving the model for training mode",
        default="./model/MLP"
    )
    parser.add_argument(
        '--model_path', help="Path to model.joblib file for testing mode or predicting mode."
    )
    parser.add_argument(
        '--dataset_dir', required=True, help='Directory of dataset. Should contain train.csv and test.csv'
    )
    parser.add_argument(
        '--output_dir', help="Directory to output the prediction file and report.", default=None
    )
    parser.add_argument(
        '--exp_name', help="Experiment name for logging.", default=""
    )
    parser.add_argument(
        '--pred', action='store_true', help='Enable prediction for the unsupervised.csv under dataset_dir.'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=5
    )

    parser = parser.parse_args()

    dataset_dir = parser.dataset_dir
    pred_mode = parser.pred

    if pred_mode:
        print("Predicting...")
        
        unsupervised_csv_file = os.path.join(dataset_dir, "unsupervised.csv")
        unsupervised_dataset = EmbeddingDataset(unsupervised_csv_file, unsupervised=True)
        unsupervised_dataloader = DataLoader(unsupervised_dataset)
        
        input_size = len(unsupervised_dataset[0])
        num_classes = 2
        
        model_path = parser.model_path
        model = MLP(input_size, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        time_stamp = os.path.split(model_path)[-1].split(".")[
            0].split("_")[-1]
        
        prediction = []
        with torch.no_grad():
            for input in tqdm(unsupervised_dataloader, total=len(unsupervised_dataloader)):
                outputs = model(input)
                preds = torch.argmax(outputs, dim=1)
                prediction.extend(preds.tolist())
        
        output_dir = parser.output_dir
        if output_dir == None:
            output_dir = os.path.dirname(model_path)
            
        pred_path = "prediction_" + time_stamp + ".csv"
        pred_path = os.path.join(output_dir, pred_path)
        
        y_pred_df = pd.DataFrame(prediction, columns=['pred'])
        y_pred_df.to_csv(pred_path, index=False)
        print("Prediction for unsupervised dataset saved as:", pred_path)
        
        
        return

    model_dir = parser.model_dir
    exp_name = parser.exp_name
    model_dir = os.path.join(model_dir, exp_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    time_stamp = datetime.now().strftime("%b%d-%H-%M-%S")

    # Define the paths to the train and test files
    train_csv_file = os.path.join(dataset_dir, "train.csv")
    test_csv_file = os.path.join(dataset_dir, "test.csv")

    # Instantiate the datasets and dataloaders
    train_dataset = EmbeddingDataset(train_csv_file)
    test_dataset = EmbeddingDataset(test_csv_file)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define your input size and number of classes
    input_size = len(train_dataset[0][0])
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

    num_epochs = parser.num_epochs
    steps = 0
    best_model_state_dict = None
    best_metric = 2147483647
    report = ""
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):  # Adjust the number of epochs as needed
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
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
            for inputs, targets in tqdm(test_dataloader, total=len(test_dataloader)):
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
        val_recall = recall_score(val_targets, val_preds)

        # Use loss as the criterion
        if val_loss < best_metric:
            best_metric = val_loss
            best_model_state_dict = deepcopy(model.state_dict())
            report = classification_report(val_targets, val_preds)

        # Log metrics to TensorBoard
        writer.add_scalar('Train Loss', train_loss /
                          len(train_dataloader), epoch)
        writer.add_scalar('Val Loss', val_loss / len(test_dataloader), epoch)
        writer.add_scalar('Val Accuracy', val_accuracy, epoch)
        writer.add_scalar('Val Precision', val_precision, epoch)
        writer.add_scalar('Val Recall', val_recall, epoch)
        writer.add_scalar('Val F1-Score', val_f1score, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss / len(train_dataloader)}, Val Loss: {val_loss / len(test_dataloader)}, Val Accuracy: {val_accuracy}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val F1-Score: {val_f1score}')

    # Close the TensorBoard writer
    writer.close()
    
    print("Best model report:")
    print(report)

    output_dir = parser.output_dir
    if output_dir == None:
        output_dir = model_dir

    report_path = "report_" + time_stamp + ".txt"
    report_path = os.path.join(output_dir, report_path)
    with open(report_path, "w") as out:
        out.write(report)
    print("Report saved as:", report_path)

    model_path = "MLP_"+time_stamp+".pt"
    model_path = os.path.join(model_dir, model_path)
    torch.save(best_model_state_dict, model_path)
    print("Best model saved as: ", model_path)


if __name__ == "__main__":
    main()
