# All parameters are defined in params.py

################################ Task to run ################################
task_generate_samples = False
task_train_discriminator = True
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
lr = 0.0001

