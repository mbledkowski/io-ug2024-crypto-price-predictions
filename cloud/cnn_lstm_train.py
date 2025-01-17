import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import random
import os
import yfinance as yf
import wandb
import math

# From BTCUSD
vol_mean = 1.1782553285681006
vol_std = 1.3461329963656354
price_mean = 8.135346188400915
price_std = 1.9180121525920217


def load_and_fill_in_gaps(data):
    data.index = pd.to_datetime(data.index, unit="ms")

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    # df.resample('1min').fillna()
    all_times = pd.date_range(start=data.index.min(),
                              end=data.index.max(), freq="1min")
    df = data.reindex(all_times)

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
    df_norm["volume"] = (df_norm["volume"] - vol_mean) / vol_std
    return df_norm


def normalize_prices(df_norm):
    columns = ["open", "high", "low", "close"]
    for column in columns:
        df_norm[column] = np.log(np.add(df_norm[column], 1))
        df_norm[column] = (df_norm[column] - price_mean) / price_std

    return df_norm


def get_normalized_data(pair=None, test=False):
    pairs = [
        "1inchusd.csv",
        "aaveusd.csv",
        "adausd.csv",
        "aixusd.csv",
        "algusd.csv",
        "ampusd.csv",
        "ancusd.csv",
        "antusd.csv",
        "apenftusd.csv",
        "apeusd.csv",
        "aptusd.csv",
        "arbusd.csv",
        "astusd.csv",
        "atlasusd.csv",
        "atousd.csv",
        "avaxusd.csv",
        "avtusd.csv",
        "axsusd.csv",
        "b2musd.csv",
        "balusd.csv",
        "bandusd.csv",
        "batusd.csv",
        "bchabcusd.csv",
        "bchnusd.csv",
        "bestusd.csv",
        "bftusd.csv",
        "bgbusd.csv",
        "blurusd.csv",
        "bmiusd.csv",
        "bmnusd.csv",
        "bntusd.csv",
        "bobausd.csv",
        "boousd.csv",
        "bosonusd.csv",
        "boxusd.csv",
        "briseusd.csv",
        "bsvusd.csv",
        "btcusd.csv",
        "btgusd.csv",
        "btseusd.csv",
        "bttusd.csv",
        "ccdusd.csv",
        "celusd.csv",
        "chexusd.csv",
        "chsbusd.csv",
        "chzusd.csv",
        "clousd.csv",
        "cndusd.csv",
        "compusd.csv",
        "convusd.csv",
        "crvusd.csv",
        "ctkusd.csv",
        "ctxusd.csv",
        "daiusd.csv",
        "datusd.csv",
        "dcrusd.csv",
        "dgbusd.csv",
        "dgxusd.csv",
        "dogeusd.csv",
        "dogusd.csv",
        "dorausd.csv",
        "dotusd.csv",
        "drnusd.csv",
        "dshusd.csv",
        "dtausd.csv",
        "dtxusd.csv",
        "duskusd.csv",
        "dvfusd.csv",
        "edousd.csv",
        "egldusd.csv",
        "enjusd.csv",
        "eosusd.csv",
        "essusd.csv",
        "etcusd.csv",
        "eth2xusd.csv",
        "ethusd.csv",
        "ethwusd.csv",
        "etpusd.csv",
        "eususd.csv",
        "eutusd.csv",
        "exousd.csv",
        "fbtusd.csv",
        "fclusd.csv",
        "fetusd.csv",
        "filusd.csv",
        "flokiusd.csv",
        "flrusd.csv",
        "forthusd.csv",
        "ftmusd.csv",
        "fttusd.csv",
        "funusd.csv",
        "galausd.csv",
        "genusd.csv",
        "gmtusd.csv",
        "gnousd.csv",
        "gntusd.csv",
        "gocusd.csv",
        "gotusd.csv",
        "gptusd.csv",
        "grtusd.csv",
        "gstusd.csv",
        "gtxusd.csv",
        "gxtusd.csv",
        "hecusd.csv",
        "hezusd.csv",
        "hixusd.csv",
        "hmtusd.csv",
        "hotusd.csv",
        "htxusd.csv",
        "iceusd.csv",
        "icpusd.csv",
        "idxusd.csv",
        "iosusd.csv",
        "iotusd.csv",
        "iqxusd.csv",
        "jasmyusd.csv",
        "jstusd.csv",
        "kaiusd.csv",
        "kanusd.csv",
        "kncusd.csv",
        "ksmusd.csv",
        "laiusd.csv",
        "ldousd.csv",
        "leousd.csv",
        "linkusd.csv",
        "lrcusd.csv",
        "ltcusd.csv",
        "luna2usd.csv",
        "lunausd.csv",
        "luxousd.csv",
        "lymusd.csv",
        "manusd.csv",
        "maticusd.csv",
        "mgousd.csv",
        "mimusd.csv",
        "mirusd.csv",
        "mkrusd.csv",
        "mlnusd.csv",
        "mnausd.csv",
        "mobusd.csv",
        "mtnusd.csv",
        "mxntusd.csv",
        "ncausd.csv",
        "nearusd.csv",
        "necusd.csv",
        "neousd.csv",
        "nexousd.csv",
        "nomusd.csv",
        "nutusd.csv",
        "nxrausd.csv",
        "oceanusd.csv",
        "odeusd.csv",
        "ognusd.csv",
        "okbusd.csv",
        "omgusd.csv",
        "omnusd.csv",
        "oneusd.csv",
        "onlusd.csv",
        "opxusd.csv",
        "orsusd.csv",
        "oxyusd.csv",
        "pasusd.csv",
        "paxusd.csv",
        "planetsusd.csv",
        "pluusd.csv",
        "pngusd.csv",
        "pnkusd.csv",
        "poausd.csv",
        "polcusd.csv",
        "polisusd.csv",
        "prmxusd.csv",
        "qrdousd.csv",
        "qshusd.csv",
        "qtfusd.csv",
        "qtmusd.csv",
        "rbtusd.csv",
        "rcnusd.csv",
        "reefusd.csv",
        "repusd.csv",
        "requsd.csv",
        "rifusd.csv",
        "rlyusd.csv",
        "rrbusd.csv",
        "rrtusd.csv",
        "sandusd.csv",
        "sanusd.csv",
        "seiusd.csv",
        "senateusd.csv",
        "sgbusd.csv",
        "shftusd.csv",
        "shibusd.csv",
        "sidususd.csv",
        "smrusd.csv",
        "sngusd.csv",
        "sntusd.csv",
        "snxusd.csv",
        "solusd.csv",
        "spellusd.csv",
        "srmusd.csv",
        "stgusd.csv",
        "stjusd.csv",
        "suiusd.csv",
        "sukuusd.csv",
        "sunusd.csv",
        "sushiusd.csv",
        "sweatusd.csv",
        "swmusd.csv",
        "sxxusd.csv",
        "terraustusd.csv",
        "thetausd.csv",
        "tknusd.csv",
        "tlosusd.csv",
        "tonusd.csv",
        "tradeusd.csv",
        "treebusd.csv",
        "triusd.csv",
        "trxusd.csv",
        "tsdusd.csv",
        "udcusd.csv",
        "uniusd.csv",
        "uopusd.csv",
        "uosusd.csv",
        "uskusd.csv",
        "ustusd.csv",
        "utkusd.csv",
        "veeusd.csv",
        "velousd.csv",
        "vetusd.csv",
        "vrausd.csv",
        "vsyusd.csv",
        "wavesusd.csv",
        "waxusd.csv",
        "wbtusd.csv",
        "wildusd.csv",
        "wminimausd.csv",
        "wncgusd.csv",
        "woousd.csv",
        "wprusd.csv",
        "wtcusd.csv",
        "xautusd.csv",
        "xcadusd.csv",
        "xchusd.csv",
        "xcnusd.csv",
        "xdcusd.csv",
        "xlmusd.csv",
        "xmrusd.csv",
        "xrausd.csv",
        "xrdusd.csv",
        "xrpusd.csv",
        "xsnusd.csv",
        "xtzusd.csv",
        "xvgusd.csv",
        "yfiusd.csv",
        "yggusd.csv",
        "yywusd.csv",
        "zbtusd.csv",
        "zcnusd.csv",
        "zecusd.csv",
        "zilusd.csv",
        "zmtusd.csv",
        "zrxusd.csv",
    ]
    data = None
    if test:
        test_pair = "MATIC-USD"
        data = yf.download(test_pair, interval="1m", group_by="Ticker")
        data = data[test_pair]
        data = data.iloc[:, [0, 3, 1, 2, 4]]
        data.columns = map(str.lower, data.columns)
    else:
        if pair is None:
            pair = pairs[random.randint(0, len(pairs) - 1)]
        data = pd.read_csv("data/" + pair, index_col="time")
        print(data.columns)
    df = load_and_fill_in_gaps(data)
    df_norm = normalize_volume(df)
    df_norm = normalize_prices(df_norm)
    return df_norm


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=420)


# Create sequences and targets
def create_sequences(data, seq_length, start=0):
    sequences = []
    targets = []
    for j in range(start, len(data) - seq_length, seq_length):
        seq = data[j: j + seq_length]
        target = data[j + seq_length]
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


def get_dataloader(data, seq_length=64, batch_size=64, epoch=0):
    # Create sequences and targets
    sequence, target = create_sequences(data, seq_length, epoch % seq_length)

    # Create Datasets
    dataset = TimeSeriesDataset(sequence, target)

    # Create DataLoaders
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create RNN Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, conv1=16, conv2=8):
        super(LSTMModel, self).__init__()

        self.dropout_rate = 0.0

        self.input_dim = input_dim

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.conv1_channels = conv1
        self.conv2_channels = conv2

        self.sigma = nn.Sigmoid()

        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.conv1 = nn.Conv2d(1, self.conv1_channels, (3, 3), padding=1)

        self.selu1 = nn.SELU()

        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.conv2 = nn.Conv2d(
            self.conv1_channels, self.conv2_channels, (3, 3), padding=1
        )

        self.selu2 = nn.SELU()

        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.conv3 = nn.Conv2d(self.conv2_channels, 1, (3, 3), padding=1)

        self.selu3 = nn.SELU()

        self.dropout4 = nn.Dropout(self.dropout_rate)

        # RNN
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout_rate,
        )

        self.dropout5 = nn.Dropout(self.dropout_rate)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        if x.size(0) == 1:
            h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)

        self.dropout1.p = self.dropout_rate
        self.dropout2.p = self.dropout_rate
        self.dropout3.p = self.dropout_rate
        self.dropout4.p = self.dropout_rate
        self.dropout5.p = self.dropout_rate
        self.lstm.dropout = self.dropout_rate

        inp = self.sigma(x)
        inp = inp.unsqueeze(1)
        conv1 = self.selu1(self.conv1(self.dropout1(inp)))
        conv2 = self.selu2(self.conv2(self.dropout2(conv1)))
        conv3 = self.selu3(self.conv3(self.dropout3(conv2)))

        conv3 = conv3.squeeze()
        out, _ = self.lstm(self.dropout4(conv3), (h0, c0))
        if x.size(0) == 1:
            out = self.fc(out[-1, :])
        else:
            out = self.fc(out[:, -1, :])

        return out


def train_model(model, optimizer, criterion, train_loader):
    model.train()

    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        if i % 42 == 0:
            wandb.log({"loss": loss})

        loss.backward()
        optimizer.step()

    return model


def eval_model(model, criterion, seq_length, test_loader):
    # Evaluate on test data
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_acc = 0.0  # To accumulate accuracy scores
    total_samples = 0  # To keep track of total samples

    with torch.no_grad():  # Disable gradient calculation
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_test_pred = model(x_test)
            test_loss += criterion(y_test_pred, y_test).item()
            x_test = x_test.cpu().numpy()

            # Assuming the last feature in x_test is 'close'
            # Adjust the feature index accordingly
            prev_np = x_test[:, -1, 1]  # Shape: (batch_size,)

            pred_np = y_test_pred.squeeze().cpu().numpy()[:, 1]
            actual_np = y_test.cpu().numpy()[:, 1]

            # Initialize accuracy for this batch
            batch_accuracy = 0.0

            for pred, actual, prev in zip(pred_np, actual_np, prev_np):
                if (
                    (actual > prev and pred > prev)
                    or (actual < prev and pred < prev)
                    or (actual == prev and pred == prev)
                ):
                    batch_accuracy += 1.0
                elif (actual != prev and pred == prev) or (
                    actual == prev and pred != prev
                ):
                    batch_accuracy += 0.5
                else:
                    batch_accuracy += 0.0

                total_samples += 1

            test_acc += batch_accuracy

    test_loss /= len(test_loader)  # Average test loss
    test_acc /= total_samples / 100  # Percentage
    return test_loss, test_acc


criterions = [
    nn.L1Loss(),
    nn.MSELoss(),
    nn.HuberLoss(),
    nn.SmoothL1Loss(),
    nn.SoftMarginLoss(),
]


def main(model):
    criterion_id = 3
    seq_length = 47
    batch_size = 56
    lr = 1.7202934434750757e-05
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = criterions[criterion_id]
    criterion_test = nn.MSELoss()
    epochs = 262 * 2 * seq_length

    wandb.init(
        # set the wandb project where this run will be logged
        project="io-ug2024-cnn_lstm_cloud_test",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": "CNN-LSTM",
            "dataset": "CRYPTO",
            "epochs": epochs,
        },
    )

    # 1/(2^n) sequence to account for the higher importance of newer values
    test_loss_over_time = 50

    increase_dropout_counter = 0
    input_dropout = 0

    def calc_dropout(input_val):
        dropout = (math.atan(input_val / 100) * 2 * 0.98) / math.pi
        return dropout

    test_loader = get_dataloader(
        get_normalized_data(None, True).values, seq_length, batch_size
    )

    for epoch in range(epochs):
        if increase_dropout_counter == 5:
            increase_dropout_counter -= 1
            if epoch > 50:
                input_dropout += 1
                model.dropout_rate = calc_dropout(input_dropout)
                print(f"Dropout changed: {model.dropout_rate}")

        pair = None
        # if epoch < 21:
        #     pair = "btcusd.csv"
        train_loader = get_dataloader(
            get_normalized_data(pair).values, seq_length, batch_size, epoch
        )
        model = train_model(model, optimizer, criterion, train_loader)
        test_loss, test_acc = eval_model(
            model, criterion_test, seq_length, test_loader)
        if test_loss > test_loss_over_time:
            increase_dropout_counter += 1
        else:
            increase_dropout_counter = 0
        test_loss_over_time += test_loss
        test_loss_over_time /= 2
        wandb.log({"val_loss": test_loss, "val_acc": test_acc})
        print(f"Test loss: %s; Test accuracy: %s" % (test_loss, test_acc))


if __name__ == "__main__":
    hidden_dim = 1718
    layer_dim = 7
    conv1 = 9
    conv2 = 7
    model = LSTMModel(
        input_dim=5,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=5,
        conv1=conv1,
        conv2=conv2,
    ).to(device)
    try:
        main(model)
    except KeyboardInterrupt:
        print("Interrupted")
    torch.save(model.state_dict(), "./models/cnn_lstm_trained.model")
    wandb.finish()
