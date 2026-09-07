odaslive -c /home/shaozi/ucl/code/audition_pipeline/ablation_study/baseline/ODAS/ours.cfg

ffmpeg -f s16le -ar 16000 -ac 3 -i postfiltered_fileid_0_doa73_3spk.raw -c:a pcm_s16le postfiltered_all_slots.wav

for slot in 0 1 2 ; do ffmpeg -i postfiltered_all_slots.wav -filter:a "pan=mono|c0=c${slot}" -c:a pcm_s16le "odas_postfiltered_slot$((slot + 1)).wav"; done