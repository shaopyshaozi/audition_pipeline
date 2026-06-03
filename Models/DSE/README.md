# End-to-End DOA-Guided Speech Extraction in Noisy Multi-Talker Scenarios


## Requirements

```bash
pip install -r requirements.txt
```

## Train & Test

This project is built on the `pytorch-lightning` package.

**Train** 

```bash
python DOATrainer.py fit \
 --config=configs/DOASE.yaml \ # network config
 --model.arch.dim_input=6 \ # input dim per T-F point, i.e. 2 * the number of channels
 --model.arch.dim_output=2 \ # output dim per T-F point
 --model.arch.num_freqs=129 \ # the number of frequencies, related to model.stft.n_fft
 --data.train_dir=/datasets/train \ # the path of train dataset
 --data.test_dir=/datasets/val \ # the path of val dataset
 --trainer.precision=bf16-mixed \ # mixed precision training
 --data.batch_size=[4,4] \ # batch size for train and val
 --trainer.devices=0, \
 --trainer.max_epochs=100 # better performance may be obtained if more epochs are given
```



**Test** the model trained:

```bash
python DOATrainer.py test --config=logs/DSENet/version_x/config.yaml \ 
 --ckpt_path=logs/DSENet/version_x/checkpoints/epochY.ckpt \ 
 --trainer.devices=0,
```

## Test Result Demonstration (Demo)


We provide an intuitive demonstration of the test results in the code repository under [test_wav](./test_wav)  to showcase the core functionality of this project in multi-source speech processing.The mixed audio consists of three speakers with Directions of Arrival (DOAs) at 0°, 61°, and 317°. The detail information is described in [file](./test_wav/file.txt).

By inputting these DOA values with different beamwidths (15°, 30°, and 45°), the model successfully extracts independent audio streams corresponding to each DOA. This demonstrates the model's ability to separate and extract the voices of multiple speakers from a mixed audio signal.



