# All parameters are defined in params.py

################################ Task to run ################################
task_generate_samples = False
task_train_discriminator = False
task_train_ensemble = True
# task_eval = False

################################ file paths ################################
diffusion_mPath = './model/edm-cifar10-32x32-uncond-vp.pkl'
classifier_mPath = './model/32x32_classifier.pt'
discriminator_mPath = './model/discriminator_60_uncond_pretrained.pt'
outdir_gen = './data/generated_samples_cifar10_unconditional'
outdir_discriminator = './model'
outdir_eval = './evaluation'


################################ DG diffusion ################################
nbr_diff_steps=35 # number of diffusion steps
min_dis=10e-5
max_dis=1-10e-5
img_size=32
dg_weight_1order=2.0    # dg weight 1st order
dg_weight_2order=0    # dg weight 2nd order

time_min = 0.01 # [0,1]
time_max = 1.0  # [0,1]
boosting = True

batch_size = 64
nbr_samples = 50000


################################ Discriminator training ################################
importance_sampling = True
nbr_epochs = 20
lr = 3e-4   # 0.0003
weight_decay = 1e-7
min_diff_time = 1e-5


################################ Ensemble training ################################
# nbr_epochs from figure 3: [10,35] and try some new ones, 1000 for double decent?
# lr: default 3e-4=0.0003 try some new ones [1e-5, 1e-3]
# weight_decay: default 1e-7 try some new ones [0, 1e-9]
# min_diff_time: default 1e-5 try some new ones [0.01, 1e-5]
nbr_epochs_e0 = 60        # paper case
lr_e0 = 0.0003
weight_decay_e0 = 1e-7
min_diff_time_e0 = 1e-5

nbr_epochs_e1 = 40
lr_e1 = 0.001
weight_decay_e1 = 1e-7
min_diff_time_e1 = 1e-5

nbr_epochs_e2 = 40
lr_e2 = 0.0001
weight_decay_e2 = 0
min_diff_time_e2 = 0.01

nbr_epochs_e3 = 40
lr_e3 = 0.0001
weight_decay_e3 = 1e-3
min_diff_time_e3 = 1e-3

nbr_epochs_e4 = 40
lr_e4 = 0.00005
weight_decay_e4 = 1e-9
min_diff_time_e4 = 1e-3

nbr_epochs_e5 = 200
lr_e5 = 0.00001
weight_decay_e5 = 1e-11
min_diff_time_e5 = 1e-5