sudo mkdir -p /mnt/e
sudo mount -t drvfs E: /mnt/e

python Simu.py --train --sources 3



python runIPDnetOn.py fit \
  --data.batch_size=[16,16] \
  --trainer.devices=1 \
  --trainer.strategy=auto

python runIPDnetOn.py test \
  --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet/logs/MyModel/version_8/checkpoints/last-v1.ckpt \
  --data.batch_size=[16,16] \
  --trainer.devices=1 \
  --trainer.strategy=auto


python runIPDnetOn.py predict \
  --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet/logs/MyModel/version_8/checkpoints/last-v1.ckpt \
  --trainer.devices=1 \
  --trainer.strategy=auto