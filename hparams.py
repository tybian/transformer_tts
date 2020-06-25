from text import symbols


################################
# Experiment Parameters        #
################################
epochs = 1000
iters_per_checkpoint = 1000
seed = 1234
cudnn_enabled = True
cudnn_benchmark = False

################################
# Data Parameters             #
################################
load_mel_from_disk = True
training_files = "filelists/ljs_audio_text_train_filelist.txt"
validation_files = "filelists/ljs_audio_text_val_filelist.txt"
test_files = "filelists/ljs_audio_text_test_filelist.txt"
text_cleaners = ["english_cleaners"]

################################
# Audio Parameters             #
################################

# if use tacotron 1's feature normalization
tacotron1_norm = False
preemphasis = 0.97
ref_level_db = 20.0
min_level_db = -100.0

sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################
n_symbols = len(symbols)

# body
d_embed = 512
d_mel = 80
d_model = 512
d_inner = 2048
n_head = 4
n_layers = 3
n_position = 1024
n_frames_per_step = 1   # currently, only 1 is supported
max_decoder_steps = 1000
stop_threshold = 0.5

# Encoder prenet parameters
eprenet_chans = 512
eprenet_kernel_size = 5
eprenet_n_convolutions = 3

# Decoder prenet parameters
dprenet_size = 256

# Mel-post processing network parameters
dpostnet_chans = 512
dpostnet_kernel_size = 5
dpostnet_n_convolutions = 5

################################
# Optimization Hyperparameters #
################################
learning_rate = 0.0442  # 0.0442 is 1 / sqrt(d_model)
adam_beta1 = 0.9
adam_beta2 = 0.98
adam_eps = 1e-9
weight_decay = 0.0
warmup_step = 4000
grad_clip_thresh = 1.0
batch_size = 16
