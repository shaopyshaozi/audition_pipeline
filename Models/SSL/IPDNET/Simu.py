import os
from Opt import opt
from utils_ import set_seed,save_file,load_file
opts = opt()
args = opts.parse()
dirs = opts.dir()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
from Dataset import Parameter
import Dataset as at_dataset
import tqdm
if args.train:
    data_num = 300000
    stage = 'train'
    snr_range = Parameter(-5,15)
    rt_range = Parameter(0.2,1.3)
    set_seed(100)

if args.test:
    data_num = 4000
    stage = 'test'
    snr_range = Parameter(0,15)
    rt_range = Parameter(0.2,1)
    set_seed(101)

if args.dev:
    data_num = 4000
    stage = 'dev'
    snr_range = Parameter(0,15)
    rt_range = Parameter(0.2,1)
    set_seed(102)

speed = 343.0	
fs = 16000
T = 4 # Trajectory length (s) 
traj_points = 40 # number of RIRs per trajectory
array_setup = at_dataset.respeaker4_array_setup

# Transform
win_len = 512
win_shift_ratio = 0.5
seg_fra_ratio = 12

seg_len = int(win_len * win_shift_ratio * (seg_fra_ratio + 1))  # 3328
seg_shift = int(win_len * win_shift_ratio * seg_fra_ratio)      # 3072

print(seg_len, seg_shift)

segmenting = at_dataset.Segmenting_SRPDNN(
    K=seg_len,
    step=seg_shift,
    window=None
)

# Source signal
sourceDataset = at_dataset.LibriSpeechDataset(
	path = dirs['sousig_'+stage], 
	T = T, 
	fs = fs,
	num_source = max(args.sources), 
	return_vad = True, 
	clean_silence = True,
	stage = stage,)

# Noise signal
noiseDataset = at_dataset.NoiseDataset(
	T = T, 
	fs = fs, 
	nmic = array_setup.mic_pos.shape[0], 
	noise_type = Parameter(['diffuse'], discrete=True), 
	noise_path = dirs['noisig_'+stage], 
	c = speed)

dataset = at_dataset.RandomTrajectoryDataset(
	sourceDataset = sourceDataset,
	num_source = Parameter(args.sources, discrete=True), # Random number of sources from list-args.sources
	source_state = args.source_state,
	room_sz = Parameter([6,6,2.5], [10,8,6]),  	# Random room sizes from 6x6x2.5 to 10x8x6 meters
	T60 = rt_range,					# Random reverberation times
	abs_weights = Parameter([0.5]*6, [1.0]*6),  # Random absorption weights ratios between walls
	array_setup = array_setup,
	array_pos = Parameter([0.35,0.35,0.3], [0.65,0.65,0.5]), # Ensure a minimum separation between the array and the walls
	noiseDataset = noiseDataset,
	SNR = snr_range, 	# Start the simulation with a low level of omnidirectional noise
	nb_points = traj_points,	# Simulate RIRs per trajectory
	min_dis = Parameter(0.5),
	c = speed, 
	transforms = [segmenting]
	)

# Data generation
save_dir = os.path.join("/mnt/e/generated_data", stage)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('make dir: ' + save_dir)

print(data_num)

for idx in tqdm.tqdm(range(data_num)):
    sig_path = os.path.join(save_dir, f"{idx}.wav")
    acous_path = os.path.join(save_dir, f"{idx}.npz")

    if os.path.exists(sig_path) and os.path.exists(acous_path):
        continue

    tmp_sig_path = sig_path + ".tmp.wav"
    tmp_acous_path = acous_path + ".tmp"

    mic_signals, acoustic_scene = dataset[idx]
    save_file(mic_signals, acoustic_scene, tmp_sig_path, tmp_acous_path)

    os.replace(tmp_sig_path, sig_path)
    os.replace(tmp_acous_path, acous_path)
