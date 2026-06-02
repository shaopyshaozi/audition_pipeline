(venv_wsl) shaozi@LAPTOP-SROFMGUA:/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/SSL$ python run_IPDnet2_3spk.py predict  --ckpt_path /mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/SSL/checkpoints/ipdnet2_23.ckpt   # Predict DOA using SSL baseline

(venv_wsl) shaozi@LAPTOP-SROFMGUA:/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/DSE/eval$ python eval_all_val_4mics.py    # Process predicted DOA into three unique DOA integers

(venv_wsl) shaozi@LAPTOP-SROFMGUA:/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/SSL/DOA_eval$ python DOA_eval_acc_all.py    # Inject DOA and do Directional Speech Enhancement with DSENET

(venv_wsl) shaozi@LAPTOP-SROFMGUA:/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/ASR$ python test_inference_WER_all.py    # Turn enchanced audio and capture its word error rate