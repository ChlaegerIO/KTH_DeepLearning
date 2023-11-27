# All parameters are defined in params.py

################################ Task to run ################################
task_generate_samples = True
task_train_discriminator = False
task_eval = False


################################ file paths ################################
diffusion_mPath = './model/edm-cifar10-32x32-uncond-vp.pkl'
classifier_mPath = './model/32x32_classifier.pt'
discriminator_mPath = './model/discriminator_60_uncond_pretrained.pt'
outdir_gen = './data/generated_samples_cifar10_unconditional'


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

batch_size = 8
nbr_samples = 50#000


################################ Discriminator training ################################
nbr_epochs = 20
lr = 0.0001
