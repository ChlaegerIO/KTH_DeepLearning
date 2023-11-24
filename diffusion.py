import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class Diffusion:
    '''
    All that we need for the discriminator guided Diffusion model to generate samples
    score_model: score-based generative model
    dg_model: discriminator-guided discriminator model
    nbr_diff_steps: number of diffusion steps to run the model
    min_dis: minimum discriminator level ????
    max_dis: maximum discriminator level ????
    img_size: image size
    dg_weight_1order: weight of 1st order DG correction
    dg_weight_2order: weight of 2nd order DG correction
    device: device to run the model
    '''
    def __init__(self, score_model, dg_model, nbr_diff_steps=35, min_dis=10e-5, max_dis=1-10e-5, img_size=32, dg_weight_1order=2.0, dg_weight_2order=0, device='cuda'):
        self.score_model = score_model
        self.dg_model = dg_model
        self.nbr_diff_steps = nbr_diff_steps
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.img_size = img_size
        self.dg_weight_1order = dg_weight_1order
        self.dg_weight_2order = dg_weight_2order
        self.device = device
        self.beta_min = 0.1
        self.beta_max = 20.0


    def sample(self, x_latent, boosting, time_min, time_max, sigma_min=0.002, sigma_max=80, rho=7, periode=5, period_weight=2,
               S_churn=0, S_churn_manual = 4.0, S_noise_manual = 1.0, S_min=0, S_max=float("inf"), S_noise=1):
        '''
        Stochastic Differential Equations (SDE) solver to generate samples

        x_latent: latent space noisy image
        boosting: whether to use boosting
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: noise schedule
        periode: periode of boosting
        period_weight: period of weight boosting
        S_churn: churn rate
        S_churn_manual: churn rate for boosting
        S_noise_manual: noise level for boosting
        S_min: minimum churn rate
        S_max: maximum churn rate
        S_noise: noise level

        return: generated samples
        '''
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.score_model.sigma_min)
        sigma_max = min(sigma_max, self.score_model.sigma_max)

        # Time step discretization (sigma_t_wve) i.e. noise schedule at each time step
        step_indices = torch.arange(self.nbr_diff_steps, dtype=torch.float64, device=self.device)
        # TODO: min und max vertauscht???
        t_steps = (sigma_max ** (1 / rho) + step_indices / (self.nbr_diff_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.score_model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0, reverse order

        # S_churn settings
        log_ratio = torch.tensor([np.inf] * x_latent.shape[0], device=self.device)
        S_churn = torch.tensor([S_churn] * x_latent.shape[0], device=self.device)
        S_churn_max = torch.tensor([np.sqrt(2) -1] * x_latent.shape[0], device=self.device)
        S_noise = torch.tensor([S_noise] * x_latent.shape[0], device=self.device)

        # noise image
        x_next = x_latent * t_steps[0]
        # Sampling loop from x_T (i=0) to x_0 (i=N)
        for i, (t_cur, t_next) in (enumerate(zip(t_steps[:-1], t_steps[1:]))):
            x_cur = x_next

            # sample noise mean=0, std=S_noise
            S_churn_ = S_churn.clone()
            S_noise_ = S_noise.clone()
            if boosting and i % periode == 0:
                S_churn_[log_ratio < 0.] = S_churn_manual
                S_noise_[log_ratio < 0.] = S_noise_manual

            # noise modulation
            gamma = torch.minimum(S_churn_ / self.nbr_diff_steps, S_churn_max) if S_min <= t_cur <= S_max else torch.zeros_like(S_churn_)
            t_hat = self.score_model.round_sigma(t_cur + gamma * t_cur)

            # Add step noise to image
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt()[:,None,None,None] * S_noise_[:,None,None,None] * torch.randn_like(x_cur)

            # Denoise prediction of diffusion model
            predicted_noise = self.score_model(x_hat, t_hat).to(torch.float64)
            predicted_noise_hat = (x_hat - predicted_noise) / t_hat[:,None,None,None]

            # DG correction 1st order
            if self.dg_weight_1order != 0:
                dg_correction, log_ratio = self.get_grad_log_ratio(x_hat, t_hat, time_min, time_max)
                if boosting and i % period_weight == 0:
                    dg_correction[log_ratio < 0.] *= 2
                dg_correction_hat = self.dg_weight_1order * dg_correction / t_hat[:,None,None,None]

            # Euler step
            current_step = predicted_noise_hat + dg_correction_hat
            x_next = x_hat + (t_next - t_hat)[:,None,None,None] * current_step

            if i < self.nbr_diff_steps - 1:
                # Denoise next prediction of diffusion model
                predicted_noise_next = self.score_model(x_next, t_next).to(torch.float64)
                predicted_noise_hat_next = (x_next - predicted_noise_next) / t_next
                # DG correction 2nd order
                if self.dg_weight_2order != 0:
                    # DG correction 2nd order
                    dg_correction_next, log_ratio = self.get_grad_log_ratio(x_next, t_next, time_min, time_max)
                    dg_correction_hat_next = self.dg_weight_2order * (dg_correction_next / t_next)

                # Heun step
                next_step = predicted_noise_hat_next + dg_correction_hat_next
                x_next = x_hat + (t_next - t_hat)[:,None,None,None] * 0.5 * (current_step + next_step)

        return x_next
    
    def get_grad_log_ratio(self, input, sigma_t_wve, time_min, time_max):
        # normalize time embedding to right format
        mean_vp_tau, tau = self.transform_unnormalized_wve_to_normalized_vp(sigma_t_wve)
        if tau.min() > time_max or tau.min() < time_min:
            raise ValueError(f'tau.min()={tau.min()} is out of range [{time_min}, {time_max}]')
        if self.dg_model == None:
            raise ValueError(f'dg_model is None')
        
        with torch.enable_grad(): # why?
            input_ = mean_vp_tau[:,None,None,None] * input
            x_t = input_.float().clone().detach().requires_grad_()
            tau = torch.ones(input_.shape[0], device=tau.device) * tau

            # compute gradient of log ratio
            logits = self.dg_model(x_t, timesteps=tau)
            prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
            log_ratio = torch.log(prediction / (1. - prediction))

            # compute gradient of log ratio
            dg_score = torch.autograd.grad(log_ratio.sum(), x_t, retain_graph=False)[0]
            dg_score *= -((sigma_t_wve[:,None,None,None] ** 2) * mean_vp_tau[:,None,None,None])

        return dg_score, log_ratio
    
    def compute_tau(self, sigma_t_wve):
        tau = - self.beta_min + torch.sqrt(self.beta_min ** 2 + 2 * (self.beta_max - self.beta_min) * torch.log(1 + sigma_t_wve ** 2))
        tau /= (self.beta_max - self.beta_min)
        return tau
    
    def transform_unnormalized_wve_to_normalized_vp(self, t):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        return mean_vp_tau, tau
    
    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


# combine classifier_model and discriminator_model to one model, where only discriminator_model is trained
class Discriminator(nn.Module):
    def __init__(self, classifier, discriminator):
        super().__init__()
        self.classifier = classifier
        self.discriminator = discriminator

    def forward(self, x, timesteps, sigmoid=True):
        adm_features = self.classifier(x, timesteps=timesteps)
        x = self.discriminator(adm_features, timesteps, sigmoid=sigmoid, condition=None).view(-1)
        return x