import torch
import tqdm


class Diffusion:
    def __init__(self, T_nfe=35, min_dis=10e-5, max_dis=1-10e-5, img_size=32, weight_DG=2.0, device='cuda'):
        self.T_nfe = T_nfe
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.img_size = img_size
        self.weight_DG = weight_DG
        self.device = device


    def sample(self, score_model, discriminator, batch):
        '''
        Stochastic Differential Equations (SDE) solver
        inputImage: 2D array of image data
        min_dis: minimum discriminator value
        max_dis: maximum discriminator value
        t_mid: time point ???
        nfe: number of forward Euler steps to take
        weight_DG: weight of the diffusion gradient term
        '''
        # noise image
        x_T = torch.randn(batch, 3, self.img_size, self.img_size, device=self.device)
        for i in tqdm(reversed(range(self.T_nfe))):
            # sample noise mean=0, std=S_1?
            eps = torch.randn_like(x_T)

            # sample time
            t = (torch.ones(batch) * i).to(self.device)
            # time shifted?
            # t_hat

            # x_ti_hat?

            # f(x_t,t) = x_t

            # g(t) = sqrt(beta_t)

            # dt_hat = 1 / sqrt(alpha_t(1-alpha_t_hat)))

            # dw_hat = 1 / sqrt(alpha) eps

            # s_theta(x_t,t)
            predicted_noise = score_model(x_T, t)


            # forward 2nd order Heun method for s_theta


            # forward 1st order Euler method for c_phi


# combine classifier_model and discriminator_model to one model, where only discriminator_model is trained
class Discriminator(nn.Module):
    def __init__(self, classifier, discriminator):
        super().__init__()
        self.classifier = classifier
        self.discriminator = discriminator

    def forward(self, x):
        x = self.classifier(x)
        x = self.discriminator(x)
        return x

