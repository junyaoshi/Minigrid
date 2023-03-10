"""
References: 
https://towardsdatascience.com/
predicting-probability-distributions-using-neural-networks-abef7db10eac
"""

import math

from tqdm.auto import tqdm
import torch
from torch import nn

BN_TRACK_STATS = True
AFFINE = True


class block(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        """
        A 2-layer residual learning building block as illustrated by Fig.2
        in "Deep Residual Learning for Image Recognition"
        Parameters:
        - in_features:  int
                        the number of input features of this block
        - out_features: int
                        the number of output features of this block
        Attributes:
        - residual: boolean
                     When false the residual shortcut is removed
                     resulting in a 'plain' block.
        """
        # Setup layers
        self.fc1 = nn.Linear(num_features, num_features, bias=False)
        self.bn1 = nn.BatchNorm1d(
            num_features, track_running_stats=BN_TRACK_STATS, affine=AFFINE
        )
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_features, num_features, bias=False)
        self.bn2 = nn.BatchNorm1d(
            num_features, track_running_stats=BN_TRACK_STATS, affine=AFFINE
        )
        self.relu2 = nn.ReLU()

    def shortcut(self, z, x):
        """
        Implements parameter free shortcut connection by identity mapping.
        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        return z + x

    def forward(self, x, residual=False):
        z = self.fc1(x)
        z = self.bn1(z)
        z = self.relu1(z)

        z = self.fc2(z)
        z = self.bn2(z)

        # Shortcut connection
        # This if statement is the only difference between
        # a fully connected net and a resnet!
        if residual:
            z = self.shortcut(z, x)

        z = self.relu2(z)

        return z


class EndtoEndNet(nn.Module):
    def __init__(
        self, in_features, out_features, n_blocks, residual=True, density=True
    ):
        """Architecture for fully connected network
        Input -> ResNet -> Action
        Parameters:
        - in_features:  int
                        the number of input features

        - out_features: int
                        the number of output features

        - n_blocks:     int
                        the number of (residual) blocks in the network

        - residual:     boolean
                        When false the residual shortcut is removed
                        resulting in a 'plain' block.

        - density:      boolean
                        When true the output is a density function
        """
        super().__init__()
        self.residual = residual
        self.density = density

        # Input
        self.bn0 = nn.BatchNorm1d(
            in_features, track_running_stats=BN_TRACK_STATS, affine=AFFINE
        )
        self.fcIn = nn.Linear(in_features, 256, bias=False)
        self.bnIn = nn.BatchNorm1d(
            256, track_running_stats=BN_TRACK_STATS, affine=AFFINE
        )
        self.relu = nn.ReLU()

        self.stack = nn.ModuleList([block(256) for _ in range(n_blocks)])

        # Output
        self.fcOut = nn.Linear(
            256, out_features * 2 if density else out_features, bias=True
        )

    def forward(self, x):
        z = self.bn0(x)
        z = self.fcIn(z)
        z = self.bnIn(z)
        z = self.relu(z)

        for l in self.stack:
            z = l(z, residual=self.residual)

        z = self.fcOut(z)

        if self.density:
            # Split into mean and log variance
            mean, logvar = z.chunk(2, dim=1)
            # Return mean and variance
            return mean, logvar
        else:
            return z


def train_model(reward_data, feedback_data, all_reward, all_feedback, args, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_data = torch.from_numpy(reward_data).unsqueeze(1).float().to(device)
    feedback_data = torch.from_numpy(feedback_data).unsqueeze(1).float().to(device)
    all_reward = torch.from_numpy(all_reward).float().unsqueeze(1).to(device)
    all_feedback = torch.from_numpy(all_feedback).unsqueeze(1).float().to(device)

    bsize = args.batch_size
    train_n_batches = math.ceil(len(reward_data) / bsize)
    test_n_batches = math.ceil(len(all_reward) / bsize)
    global_step = 0

    model = EndtoEndNet(1, 1, args.n_blocks, residual=args.residual, density=False)
    model.to(device)

    criterion = nn.MSELoss()
    optimzier = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.n_epochs), desc="Training Epochs", leave=True):
        # Train
        model.train()
        train_r_pred = torch.empty((0, 1), device=device)
        test_r_pred = torch.empty((0, 1), device=device)
        for batch in tqdm(
            range(train_n_batches), desc=f"epoch={epoch} train", leave=False
        ):
            start = batch * bsize
            end = start + bsize
            if end > len(reward_data):
                end = len(reward_data)
            r = reward_data[start:end]
            f = feedback_data[start:end]

            optimzier.zero_grad()
            r_pred = model(f)
            loss = criterion(r_pred, r)

            loss.backward()
            optimzier.step()

            train_r_pred = torch.cat((train_r_pred, r_pred.detach()), dim=0)
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Test
        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                range(test_n_batches), desc=f"epoch={epoch} test", leave=False
            ):
                start = batch * bsize
                end = start + bsize
                if end > len(all_reward):
                    end = len(all_reward)
                r = all_reward[start:end]
                f = all_feedback[start:end]

                r_pred = model(f)
                loss = criterion(r_pred, r)

                test_r_pred = torch.cat((test_r_pred, r_pred), dim=0)
                writer.add_scalar("test/loss", loss.item(), global_step)

        writer.add_histogram("reward/reward_data_pred", train_r_pred, global_step)
        writer.add_histogram("reward/all_reward_pred", test_r_pred, global_step)

    return model


if __name__ == "__main__":
    test_density = False
    test_binary = True

    import torch
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from feedback import noisy_sigmoid_feedback

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="model_test")

    # train data
    train_data_size = 2048
    train_x = np.ones((train_data_size, 1))
    train_x[: train_data_size // 2] *= -1
    np.random.shuffle(train_x)
    train_y = noisy_sigmoid_feedback(train_x)
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    writer.add_histogram("train_y_data", train_y, 0)

    # test data
    test_data_size = 2048
    test_x = np.ones((test_data_size, 1))
    test_x[: test_data_size // 2] *= -1
    np.random.shuffle(test_x)
    test_y = noisy_sigmoid_feedback(test_x)
    test_x = torch.from_numpy(test_x).float().to(device)
    test_y = torch.from_numpy(test_y).float().to(device)
    writer.add_histogram("test_y_data", test_y, 0)

    batch_size = 64
    train_n_batches = train_data_size // batch_size
    test_n_batches = test_data_size // batch_size
    global_step = 0

    if test_density:
        net = EndtoEndNet(1, 1, 0, residual=False, density=True)
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        for epoch in tqdm(range(100), desc="Epochs", leave=True):
            # Train
            net.train()
            for batch in tqdm(
                range(train_n_batches), desc=f"epoch={epoch} train", leave=False
            ):
                x_batch = train_x[batch * batch_size : (batch + 1) * batch_size]
                y_batch = train_y[batch * batch_size : (batch + 1) * batch_size]

                optimizer.zero_grad()
                mean, logvar = net(x_batch)
                dist = torch.distributions.Normal(mean, torch.exp(logvar))
                loss = -dist.log_prob(y_batch).mean()

                loss.backward()
                optimizer.step()

                writer.add_scalar("loss", loss.item(), global_step)
                global_step += 1

            # Test
            net.eval()
            with torch.no_grad():
                y_samples = torch.empty((0,), device=device)
                for batch in tqdm(
                    range(test_n_batches), desc=f"epoch={epoch} test", leave=False
                ):
                    x_batch = test_x[batch * batch_size : (batch + 1) * batch_size]
                    y_batch = test_y[batch * batch_size : (batch + 1) * batch_size]

                    mean, logvar = net(x_batch)
                    dist = torch.distributions.Normal(mean, torch.exp(logvar))
                    y_sample = dist.sample((10,)).flatten()
                    y_samples = torch.cat((y_samples, y_sample), dim=0)

                writer.add_histogram("y_test", y_samples.squeeze(), global_step)

        print("Done")

    if test_binary:
        net = EndtoEndNet(1, 1, 0, residual=False, density=False)
        net.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        for epoch in tqdm(range(100), desc="Epochs", leave=True):
            # Train
            net.train()
            for batch in tqdm(
                range(train_n_batches), desc=f"epoch={epoch} train", leave=False
            ):
                x_batch = train_x[batch * batch_size : (batch + 1) * batch_size]
                x_batch[x_batch == -1] = 0
                y_batch = train_y[batch * batch_size : (batch + 1) * batch_size]

                optimizer.zero_grad()
                x_pred = net(y_batch)
                loss = criterion(x_pred, x_batch)
                accuracy = (x_pred > 0).float() == x_batch

                loss.backward()
                optimizer.step()

                writer.add_scalar("train_loss", loss.item(), global_step)
                writer.add_scalar(
                    "train_accuracy", accuracy.float().mean().item(), global_step
                )
                global_step += 1

            # Test
            net.eval()
            with torch.no_grad():
                for batch in tqdm(
                    range(test_n_batches), desc=f"epoch={epoch} test", leave=False
                ):
                    x_batch = test_x[batch * batch_size : (batch + 1) * batch_size]
                    x_batch[x_batch == -1] = 0
                    y_batch = test_y[batch * batch_size : (batch + 1) * batch_size]

                    x_pred = net(y_batch)
                    loss = criterion(x_pred, x_batch)
                    accuracy = (x_pred > 0).float() == x_batch

                    writer.add_scalar("test_loss", loss.item(), global_step)
                    writer.add_scalar(
                        "test_accuracy", accuracy.float().mean().item(), global_step
                    )
