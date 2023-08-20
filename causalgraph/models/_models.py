import logging
import random
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from causalgraph.common import utils
from ._base_models import MLP
from ._columnar import ColumnsDataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")
logging.getLogger('pytorch-lightning').setLevel(logging.ERROR)
torch_log = logging.getLogger("pytorch-lightning")
torch_log.propagate = False
torch_log.setLevel(logging.ERROR)


class BaseModel(object):

    model = None
    all_columns = None
    callbacks = None
    columns = None
    logger = None
    extra_trainer_args = None
    scaler = None
    train_loader = None
    val_loader = None
    n_rows = 0

    # device = torch.device(utils.select_device("cpu"))
    device = utils.select_device("cpu")

    def __init__(
        self,
        target: str,
        dataframe: pd.DataFrame,
        test_size: float,
        batch_size: int,
        tb_suffix: str,
        seed: int,
        early_stop: bool = True,
        patience: int = 10,
        min_delta: float = 0.001
    ):
        super(BaseModel, self).__init__()

        self.target = target
        self.dataframe = dataframe
        self.test_size = test_size
        self.batch_size = batch_size
        self.seed = seed
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta

        logging.getLogger("pytorch_lightning").propagate = False
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # pl.seed_everything(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        torch.set_printoptions(precision=3, linewidth=150, sci_mode=False)
        np.set_printoptions(precision=3, linewidth=150)

        self.init_logger(tb_suffix)
        self.init_callbacks(self.early_stop, self.min_delta, self.patience)
        self.init_data()

    def init_logger(self, suffix: str):
        # Init logger. Without this, TensorBoard doesn't work.
        self.logger = TensorBoardLogger(
            "tb_logs", name=f"{self.target}_{suffix}")
        TensorBoardLogger(
            save_dir="tb_logs",
            name=f"run_{self.target}",
            version="fixed_version",
            default_hp_metric=False,
            log_graph=True,
        )

    def init_callbacks(
        self,
        early_stop: bool = True,
        min_delta: float = 0.001,
        patience: int = 10,
        prog_bar: bool = False,
    ):
        self.callbacks = []
        if early_stop:
            earlyStopping = EarlyStopping(
                monitor="val_loss",
                min_delta=min_delta,
                patience=patience)
            self.callbacks.append(earlyStopping)
        if prog_bar:
            class MeterlessProgressBar(TQDMProgressBar):
                def init_train_tqdm(self):
                    l_bar = super().init_train_tqdm()
                    l_bar.dynamic_ncols = False
                    l_bar.ncols = 0
                    return l_bar

            bar = MeterlessProgressBar()
            self.callbacks.append(bar)

    def init_data(self):
        self.all_columns = list(self.dataframe)
        self.columns = list(self.dataframe)
        self.columns.remove(self.target)
        self.columns.append("Noise")

        # Compute the sizes of the splits to have round batch sizes on both.
        self.n_rows = self.dataframe.shape[0]
        train_split_size = np.floor((1-self.test_size) * self.n_rows)
        val_split_size = np.floor(self.test_size * self.n_rows)
        if self.batch_size > val_split_size:
            self.batch_size = int(val_split_size)
        reminder = val_split_size % self.batch_size
        val_split_size = int(val_split_size - reminder)
        train_split_size = int(self.n_rows - val_split_size)
        train_df, test_df = train_test_split(
            self.dataframe,
            train_size=train_split_size,
            test_size=val_split_size,
            shuffle=True,
            random_state=1234)
        self.train_loader = DataLoader(
            ColumnsDataset(self.target, train_df),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            ColumnsDataset(self.target, test_df),
            batch_size=self.batch_size,
        )

    def override_extras(self, **kwargs):
        real_steps = np.floor(self.n_rows/self.batch_size)-1
        log_every_steps = int(min(real_steps, self.batch_size))
        # log_every_steps = np.min(int(np.floor(float(self.n_rows)/float(self.batch_size)))-1., self.batch_size)
        self.extra_trainer_args = dict(
            {
                "enable_model_summary": False,
                # "auto_lr_find": True,
                "log_every_n_steps": log_every_steps,
                "fast_dev_run": False,
                "enable_progress_bar": False,
            }
        )
        for k, default_value in self.extra_trainer_args.items():
            self.extra_trainer_args[k] = kwargs.get(k, default_value)


class MLPModel(BaseModel):
    def __init__(
        self,
        target: str,
        input_size: int,
        hidden_dim: List[int],
        activation: nn.Module,
        learning_rate: float,
        batch_size: int,
        loss_fn: str,
        dropout: float,
        num_epochs: int,
        dataframe: pd.DataFrame,
        test_size: float,
        device: Union[int, str],
        seed: int,
        early_stop: bool = True,
        patience: int = 10,
        min_delta: float = 0.001,
        **kwargs,
    ):
        super(MLPModel, self).__init__(
            target=target,
            dataframe=dataframe,
            test_size=test_size,
            batch_size=batch_size,
            tb_suffix="MLP",
            seed=seed,
            early_stop=early_stop,
            patience=patience,
            min_delta=min_delta)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.override_extras(**kwargs)

        self.model = MLP(
            self.input_size,
            self.hidden_dim,
            self.activation,
            self.batch_size,
            self.learning_rate,
            self.loss_fn,
            self.dropout)

        self.trainer = Trainer(
            max_epochs=self.num_epochs,
            logger=self.logger,
            callbacks=self.callbacks,
            accelerator=self.device,
            **self.extra_trainer_args)

    def train(self):
        print(f"Calling trainer.fit(), on target {self.target}")
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )


if __name__ == "__main__":
    data = pd.read_csv("~/phd/data/generated_linear_10.csv")
    mlp = MLPModel(
        target='V0',
        input_size=data.shape[1],
        hidden_dim=[64, 128, 64],
        learning_rate=0.05,
        batch_size=32,
        loss_fn="mse",
        dropout=0.05,
        num_epochs=200,
        dataframe=data,
        test_size=0.1,
        device="auto",
        seed=1234,
        early_stop=False)
    mlp.train()
