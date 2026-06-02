cd /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet2
source venv_wsl/bin/activate
python run_IPDnet2.py test   --ckpt_path "/mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet2/checkpoints/ipdnet2_small.ckpt"   --trainer.enable_progress_bar false


python run_IPDnet2_3spk.py predict  --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet2/checkpoints/last.ckpt

python run_IPDnet2_3spk.py validate --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet2/checkpoints/last.ckpt
python run_IPDnet2_3spk.py validate --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet2/checkpoints/epoch23_valid_loss0.1336.ckpt


python run_IPDnet2_3spk.py predict  --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/SSL/checkpoints/ipdnet2_23.ckpt