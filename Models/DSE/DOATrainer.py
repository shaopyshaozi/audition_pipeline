from models.utils.base_cli import BaseCLI
# import BaseCLI at the beginning

import os
from typing import *
import numpy as np
import random
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jsonargparse import lazy_instance
from packaging.version import Version
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as sdr

from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import Resample

import models.utils.general_steps as GS
from models.io.loss import *
from models.io.loss import neg_sa_sdr
from models.io.norm import Norm
from models.io.stft import STFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.base_cli import BaseCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from models.utils.my_earlystopping import MyEarlyStopping as EarlyStopping
from models.arch.DSENet import DSENet

import soundfile as sf
from scipy.signal import resample_poly



def load_audio_file(path: str, target_sr: int):
    wav, sr = sf.read(path, always_2d=True)   # [T, C]
    wav = wav.T.astype(np.float32)            # [C, T]

    if sr != target_sr:
        resampled = []
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        for ch in wav:
            resampled_ch = resample_poly(ch, up, down).astype(np.float32)
            resampled.append(resampled_ch)
        wav = np.stack(resampled, axis=0)
        sr = target_sr

    return torch.from_numpy(wav), sr


class MultimicDataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int = 16000, cut_len: int = 4):
        self.data_dir = data_dir
        self.clean_dir = os.path.join(data_dir, "clean") # data_dir:/data/private/datasets/dataset_3mic_6spk/train   /clean/clean_fileid_0_doa0_spk1.wav
        self.noisy_dir = os.path.join(data_dir, "mic")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.cut_len = cut_len*sample_rate
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        parts = self.clean_wav_name[idx].split("_") #clean_fileid_0_doa0_spk1.wav

        parent_name = os.path.basename(os.path.dirname(self.data_dir))
        dataset_tag = parent_name.split('_')[-1]

        if parts[4][0:5] == 'width':
            noisy_file = os.path.join(
                self.noisy_dir,
                f"mic_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{dataset_tag}.wav"
            )
        else:
            noisy_file = os.path.join(
                self.noisy_dir,
                f"mic_{parts[1]}_{parts[2]}_{parts[3]}_{dataset_tag}.wav"
            )

        clean_ds, sr = load_audio_file(clean_file, self.sample_rate)
        noisy_ds, sr = load_audio_file(noisy_file, self.sample_rate)

        length = len(clean_ds[0])
        assert length == len(noisy_ds[0])
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[ : , : self.cut_len%length])
            noisy_ds_final.append(noisy_ds[ : , : self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 4 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[ : , wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[ : , wav_start:wav_start + self.cut_len]


        if len(clean_ds)==1:
            clean_ds = clean_ds.repeat(len(noisy_ds),1)


        prefix = "doa"
        doa = int (parts[3][len(prefix):])

        if parts[4][0:5] == 'width':
            w_prefix = "width"
            width = int (parts[4][len(w_prefix):])
        else:
            width = 30 # default width is 30

        target_name = f"enhance_{parts[1]}_{parts[2]}_doa_width.wav"

        paras = {
                'index': idx,
                'wavname': self.clean_wav_name[idx],
                'savename': target_name,
                # 'target': self.target,
        }

        return noisy_ds, clean_ds, doa, width, paras


class MyDataModule(LightningDataModule):

    def __init__(self, train_dir: str, test_dir: str, sample_rate: int, num_workers: int = 0, batch_size: Tuple[int, int] = (2, 2)):
        super().__init__()
        
        self.num_workers = num_workers
        self.batch_size = batch_size  # train: batch_size[0]; test: batch_size[1]
        self.train_ds = MultimicDataset(train_dir, sample_rate=sample_rate)
        self.test_ds = MultimicDataset(test_dir, sample_rate=sample_rate)


    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size[0], num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size[1], num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True)



class TrainModule(pl.LightningModule):
    """Network Lightning Module, which controls the training, testing, and inference of given arch and io
    """
    name: str  # used by CLI for creating logging dir
    import_path: str = 'SharedTrainer.TrainModule'

    def __init__(
        self,
        arch: DSENet = lazy_instance(DSENet),
        stft: STFT = STFT(n_fft=256, n_hop=128, win_len=256),
        norm: Norm = Norm(mode='utterance'),
        loss: Loss = Loss(loss_func=neg_sa_sdr, pit=False),
        optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {
            "lr": 0.001
        }),
        lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = ('ReduceLROnPlateau', {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-4
        }),
        metrics: List[str] = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'eSTOI'],
        mchunk: Optional[Tuple[float, float]] = None,  # chunk for cal_metrics_functional
        val_metric: str = 'loss',
        write_examples: int = 200,
        ensemble: Union[int, str, List[str], Literal[None]] = None,
        compile: bool = False,
        exp_name: str = "exp",
        reset: Optional[List[str]] = None,
        sample_rate: int = 16000,
        ref_channel: int = 0,
        test_doa: int = -1,
        test_beam: int = -1,
        freeze: bool = False,
    ):
        """
        Args:
            exp_name: set exp_name to notag when debug things. Defaults to "exp".
            metrics: metrics used at test time. Defaults to ['SNR', 'SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ'].
            write_examples: write how many examples at test.
            reset: reset the items in checkpoint when loading e.g. ['optimizer', 'lr_scheduler'].
        """

        super().__init__()

        args = locals().copy()  # capture the parameters passed to this function or their edited values

        if compile != False:
            assert Version(torch.__version__) >= Version('2.0.0'), torch.__version__
            self.arch = torch.compile(arch, dynamic=Version(torch.__version__) >= Version('2.1.0'))
        else:
            self.arch = arch

        self.stft = stft
        self.norm = norm
        self.loss = loss
        self.compile_model = compile
        self.freeze = freeze

        self.val_cpu_metric_input = []
        self.norm_if_exceed_1 = True
        self.name = type(arch).__name__
        self.reset = reset

        self.sample_rate = sample_rate
        self.test_doa = test_doa
        self.test_beam = test_beam


        # save other parameters to `self`
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)



    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
        GS.on_train_start(self=self, exp_name=self.exp_name, model_name=self.name, nfft=self.stft.n_fft, model_class_path=self.import_path)

    def forward(self, x: Tensor, DOA: Tensor, width: Tensor, istft: bool = True) -> Tensor:
        """
        Args:
            x: [B,C,T]

        Returns:
            Tensor: ys_hat
        """
        # obtain STFT X
        X, stft_paras = self.stft.stft(x)  # [B,C,F,T], complex
        B, C, F, T = X.shape

        X, norm_paras = self.norm.norm(X, ref_channel=self.ref_channel)

        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C

        # network process
        out = self.arch(X, DOA, width)
        if not torch.is_complex(out):
            out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2)) 
        out = out.permute(0, 3, 1, 2)  # [B,M,F,T]

        # to time domain
        Yr_hat = out
        if norm_paras is not None:
            Yr_hat = self.norm.inorm(Yr_hat, norm_paras)
        yr_hat = self.stft.istft(Yr_hat, stft_paras) if istft else torch.view_as_real(Yr_hat)

        return yr_hat 

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, yr, DOA, width, paras = batch  # x: [B,M,T], yr: [B,M,T]

        yr_hat = self.forward(x, DOA, width) #[B,1,T]
        yr = yr[:, self.ref_channel, :].unsqueeze(dim=1)  #[B,1,T]

        # float32 loss calculation
        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True, stft_func = self.stft)  # convert to float32 to avoid numerical problem in loss calculation
        else:
            loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True, stft_func = self.stft)

        si_sdr_train = si_sdr(preds=yr_hat, target=yr)
        self.log('train/si_sdr', si_sdr_train.mean(), batch_size=yr[0].shape[0], prog_bar=True)
        self.log('train/' + self.loss.name, loss, batch_size=yr[0].shape[0], prog_bar=True)

        if torch.isnan(loss):
            print("\n"*5)
            print("loss is nan")
            print("x: \n", x)
            print("yr_hat: \n", yr_hat)
            print(DOA)
            print("\n"*5)
            return None

        return loss

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, yr, DOA, width, paras = batch

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        # forward & loss
        yr_hat = self.forward(x, DOA, width)
        yr = yr[:, self.ref_channel, :].unsqueeze(dim=1)
        loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True, stft_func = self.stft)

        # metrics
        sdr_val = sdr(yr_hat, yr).mean()
        

        si_sdr_val = si_sdr(preds=yr_hat, target=yr).mean()

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # logging
        self.log('val/' + self.loss.name, loss, sync_dist=True, batch_size=yr.shape[0])
        val_metric = {'loss': loss, 'si_sdr': si_sdr_val, 'sdr': sdr_val}[self.val_metric]
        self.log('val/metric', val_metric, sync_dist=True, batch_size=yr.shape[0])  # log val/metric for checkpoint picking

        # always computes the sdr/sisdr for the comparison of different runs
        self.log('val/sdr', sdr_val, sync_dist=True, batch_size=yr.shape[0])
        if self.loss.name != 'neg_si_sdr':
            # always computes the neg_si_sdr for the comparison of different runs in Tensorboard
            self.log('val/neg_si_sdr', -si_sdr_val, sync_dist=True, batch_size=yr.shape[0])


    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=5)

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []
        

    def on_test_epoch_end(self):
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)

    def test_step(self, batch, batch_idx):
        x, yr, DOA, width, paras = batch

        if self.test_doa != -1:
            DOA = DOA.fill_(self.test_doa)
        if self.test_beam != -1 :
            width = width.fill_(self.test_beam)

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()


        yr_hat = self.forward(x, DOA, width)
        yr = yr[:, self.ref_channel, :].unsqueeze(dim=1)
        x_ref = x[:, self.ref_channel, :].unsqueeze(dim=1)

        loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True, stft_func = self.stft)
        self.log('test/' + self.loss.name, loss, batch_size=yr.shape[0], prog_bar=True, sync_dist=True) # 

        # write results & infos
        wavname = os.path.basename(paras['savename'][0])
        wavname = wavname.replace('_doa_width.wav',f'_doa{str(int(DOA))}_width{str(int(width))}.wav')
        result_dict = {'id': batch_idx, 'wavname': wavname, self.loss.name: loss.item()}

        # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
        if self.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(preds=yr_hat, mixture=x[:, self.ref_channel, :], scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)


        # calculate metrics, input_metrics, improve_metrics on GPU
        metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0], yr[0], x_ref[0], self.sample_rate, device_only='gpu')
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        self.cpu_metric_input.append((self.metrics, yr_hat[0].detach().cpu(), yr[0].detach().cpu(), x_ref[0].detach().cpu(), self.sample_rate, 'cpu'))


        # write examples
        if self.write_examples < 0 or batch_idx < self.write_examples:
            GS.test_setp_write_example(
                self=self,
                x=x_ref,
                yr=yr,
                yr_hat=yr_hat,
                sample_rate=self.sample_rate,
                DOA=DOA,
                paras=paras,
                result_dict=result_dict,
                wavname=wavname,
                exp_save_path=self.exp_save_path,
            )

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # return metrics, which will be collected, saved in test_epoch_end
        if 'metrics' in paras:
            del paras['metrics']  # remove circular reference
        result_dict['paras'] = paras.copy()

        self.results.append(result_dict)
        return result_dict


    def configure_optimizers(self):
        """configure optimizer and lr_scheduler"""
        return GS.configure_optimizers(
            self=self,
            optimizer=self.optimizer[0],
            optimizer_kwargs=self.optimizer[1],
            monitor='val/metric',
            lr_scheduler=self.lr_scheduler[0] if self.lr_scheduler is not None else None,
            lr_scheduler_kwargs=self.lr_scheduler[1] if self.lr_scheduler is not None else None,
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        GS.on_load_checkpoint(self=self, checkpoint=checkpoint, ensemble_opts=self.ensemble, compile=self.compile_model, freeze=self.freeze, reset=self.reset,)


    def on_after_backward(self) -> None:
        super().on_after_backward()
        if self.current_epoch != 0:
            return
        # This function is useful for debuging the following error:
        # RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
        for name, p in self.named_parameters():
            if p.grad is None:
                print('unused parameter (check code or freeze it):', name)


class TrainCLI(BaseCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # # EarlyStopping
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        early_stopping_defaults = {
            "early_stopping.enable": False,
            "early_stopping.monitor": "val/metric",
            "early_stopping.patience": 10,
            "early_stopping.mode": "max",
            "early_stopping.min_delta": 0.1,
        }
        parser.set_defaults(early_stopping_defaults)

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_metric{val/metric:.4f}",
            "model_checkpoint.monitor": "val/metric",
            "model_checkpoint.mode": "max",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": -1,  # save all checkpoints
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        self.add_model_invariant_arguments_to_parser(parser)


if __name__ == '__main__':
    # python SharedTrainer.py --help
    cli = TrainCLI(
        TrainModule,
        MyDataModule,
        seed_everything_default=2,  
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        # subclass_mode_data=True,
    )
    
