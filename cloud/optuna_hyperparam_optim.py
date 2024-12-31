import optuna
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import random
import os
import time
from calflops import calculate_flops


def load_and_fill_in_gaps():
    df_og = pd.read_csv("data/btcusd.csv", index_col="time")
    df_og.index = pd.to_datetime(df_og.index, unit="ms")

    # df.resample('1min').fillna()
    all_times = pd.date_range(
        start=df_og.index.min(), end=df_og.index.max(), freq="1min"
    )
    df = df_og.reindex(all_times)

    # Forward-fill the 'Close' column
    df["close"] = df["close"].ffill()

    # Copy the forward-filled 'Close' value to 'Open', 'High', and 'Low'
    df["open"] = df["open"].combine_first(df["close"])
    df["high"] = df["high"].combine_first(df["close"])
    df["low"] = df["low"].combine_first(df["close"])

    # Set 'Volume' to 0 for the newly created rows
    df["volume"] = df["volume"].fillna(0)

    return df


def normalize_volume(df):
    df_norm = df.copy()
    df_norm["volume"] = np.log(np.add(df_norm["volume"], 1))

    df_norm["volume"].plot.hist()

    df_norm["volume"] = (df_norm["volume"] - df_norm["volume"].mean()) / df_norm[
        "volume"
    ].std()

    return df_norm


def normalize_prices(df_norm):
    columns_combined = pd.concat([
        df_norm["open"],
        df_norm["high"],
        df_norm["low"],
        df_norm["close"],
    ])
    columns_combined_log = np.log(np.add(columns_combined, 1))

    columns = ["open", "high", "low", "close"]
    for column in columns:
        df_norm[column] = np.log(np.add(df_norm[column], 1))

    for column in columns:
        df_norm[column] = (
            df_norm[column] - columns_combined_log.mean()
        ) / columns_combined_log.std()

    return df_norm


def main():
    df = load_and_fill_in_gaps()
    df_norm = normalize_volume(df)
    df_norm = normalize_prices(df_norm)
    return df_norm


df_norm = main()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=420)

# Convert DataFrame to numpy array
data = df_norm.values


# Create sequences and targets
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i: i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def get_dataloaders(seq_length=64, batch_size=64):
    # Create sequences and targets
    sequences, targets = create_sequences(data, seq_length)

    # Sequential split
    train_size = int(len(sequences) * 0.3)
    X_train, X_test = (
        sequences[:train_size],
        sequences[train_size: train_size * 2],
    )
    y_train, y_test = (
        targets[:train_size],
        targets[train_size: train_size * 2],
    )

    # Create Datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create RNN Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, conv1=16, conv2=8):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.conv1_channels = conv1
        self.conv2_channels = conv2

        if self.conv2_channels > 0 and self.conv1_channels == 0:
            self.conv1_channels = self.conv2_channels
            self.conv2_channels = 0

        self.sigma = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, self.conv1_channels, (3, 3), padding=1)

        self.selu1 = nn.SELU()

        self.conv2 = nn.Conv2d(
            self.conv1_channels, self.conv2_channels, (3, 3), padding=1
        )

        self.selu2 = nn.SELU()

        self.conv3 = None
        if self.conv2_channels == 0:
            self.conv3 = nn.Conv2d(self.conv1_channels, 1, (3, 3), padding=1)
        else:
            self.conv3 = nn.Conv2d(self.conv2_channels, 1, (3, 3), padding=1)

        self.selu3 = nn.SELU()

        # RNN
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        if x.size(0) == 1:
            h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)

        inp = self.sigma(x)
        out = None
        if self.conv1_channels == 0 and self.conv2_channels == 0:
            out, _ = self.lstm(self.selu1(inp), (h0, c0))
        else:
            inp = inp.unsqueeze(1)
            conv1 = self.selu1(self.conv1(inp))
            conv2 = None
            conv3 = None
            if self.conv2_channels > 0:
                conv2 = self.selu2(self.conv2(conv1))
                conv3 = self.selu3(self.conv3(conv2))
            else:
                conv3 = self.selu3(self.conv3(conv1))

            conv3 = conv3.squeeze()
            out, _ = self.lstm(conv3, (h0, c0))
        if x.size(0) == 1:
            out = self.fc(out[-1, :])
        else:
            out = self.fc(out[:, -1, :])
        return out


def train_model(model, optimizer, criterion, train_loader):
    start = time.time()
    model.train()
    for i, (x, y) in enumerate(train_loader):
        if time.time() - start > 600:
            break

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
    return model


def eval_model(model, criterion, seq_length, test_loader):
    # Evaluate on test data
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_test_pred = model(x_test)
            test_loss += criterion(y_test_pred, y_test).item()

    test_loss /= len(test_loader)  # Average test loss
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, seq_length, 5),
        output_as_string=False,
        output_precision=4,
        # print_result=False,
        # print_detailed=False,
    )
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
    return flops, test_loss


criterions = [
    nn.L1Loss(),
    nn.MSELoss(),
    nn.HuberLoss(),
    nn.SmoothL1Loss(),
    nn.SoftMarginLoss(),
]


def optimalize(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 1, 4096, log=True)
    layer_dim = trial.suggest_int("layer_dim", 1, 1024, log=True)
    conv1 = trial.suggest_int("conv1", 1, 1025, log=True) - 1
    conv2 = trial.suggest_int("conv2", 1, 513, log=True) - 1
    model = LSTMModel(
        input_dim=5,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=5,
        conv1=conv1,
        conv2=conv2,
    ).to(device)
    seq_length = trial.suggest_int("seq_length", 2, 512)
    batch_size = trial.suggest_int("batch_size", 1, 128)
    train_loader, test_loader = get_dataloaders(seq_length, batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )
    criterion = criterions[trial.suggest_int(
        "criterion", 0, len(criterions) - 1)]
    model = train_model(model, optimizer, criterion, train_loader)
    criterion = nn.MSELoss()
    flops, test_loss = eval_model(model, criterion, seq_length, test_loader)
    return flops, test_loss


def objective(trial):
    flops, test_loss = float("inf"), float("inf")
    try:
        flops, test_loss = optimalize(trial)
        print(flops, test_loss)
    except Exception as e:
        print(e)

    return test_loss, flops


study = optuna.load_study(
    study_name="distributed-optuna", storage="mysql+pymysql://root@localhost/optuna"
)
study.optimize(objective, n_trials=500)

print(study.best_params)
