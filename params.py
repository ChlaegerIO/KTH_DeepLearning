# All parameters are defined in params.py

################################ Task to run ################################

task_generate_samples = False
task_generate_samples_ensemble = False
task_train_discriminator = False
task_train_ensemble = False
task_eval = True

################################ file paths ################################

diffusion_mPath = './model/pretrained/edm-cifar10-32x32-uncond-vp.pkl'
classifier_mPath = './model/pretrained/32x32_classifier.pt'
discriminator_mPath = './model/ensemble/discriminator_ensemble_0_55.pth'
outdir_gen_path = './data/generated_samples_cifar10_paperWithoutDG'
outdir_discriminator = './model/ensemble'
outdir_eval = './evaluation/paperWithoutDG'


# eval_load_images_path = './data/generated_samples_cifar10_unconditional_paperWithoutDG'
# FID_stats_path = '/evaluation/FID_stats_file/cifar10-32x32.npz'
eval_load_images_path = '\data\generated_samples_cifar10_paperWithoutDG'                # for windows path
FID_stats_path = '\evaluation\FID_stats_file\cifar10-32x32.npz'                         # for windows path

# ensemble paths
discriminator_mPath_e0 = './model/ensemble/discriminator_ensemble_0_55.pth'
discriminator_mPath_e1 = './model/ensemble/discriminator_ensemble_1_35.pth'
discriminator_mPath_e2 = './model/ensemble/discriminator_ensemble_2_35.pth'
discriminator_mPath_e3 = './model/ensemble/discriminator_ensemble_3_35.pth'
discriminator_mPath_e4 = './model/ensemble/discriminator_ensemble_4_35.pth'
discriminator_mPath_e5 = './model/ensemble/discriminator_ensemble_5_195.pth'

################################ DG diffusion ################################
discriminator_type = 'pretrained'  # 'own' or 'pretrained'
nbr_diff_steps=35       # number of diffusion steps
img_size=32
dg_weight_1order=2.0    # dg weight 1st order
dg_weight_2order=0    # dg weight 2nd order

time_min = 0.01  # [0,1]
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