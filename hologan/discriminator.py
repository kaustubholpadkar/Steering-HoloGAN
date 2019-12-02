import torch
import torch.nn as nn
import torch.nn.functional as F
from hologan.util import calc_mean_std
from torch.nn.utils import spectral_norm
from torch.distributions.normal import Normal


class Discriminator(nn.Module):

    def __init__(self, cont_dim):
        super(Discriminator, self).__init__()

        self.cont_dim = cont_dim

        self.noise_generator = Normal(loc=0.0, scale=0.02)

        self.convolve0 = spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2))
        nn.init.normal_(self.convolve0.weight, std=0.02)
        torch.nn.init.zeros_(self.convolve0.bias)

        self.convolve1 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2))
        nn.init.normal_(self.convolve1.weight, std=0.02)
        torch.nn.init.zeros_(self.convolve1.bias)
        self.normalize1 = nn.InstanceNorm2d(num_features=128)

        self.convolve2 = spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2))
        nn.init.normal_(self.convolve2.weight, std=0.02)
        torch.nn.init.zeros_(self.convolve2.bias)
        self.normalize2 = nn.InstanceNorm2d(num_features=256)

        self.convolve3 = spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2))
        nn.init.normal_(self.convolve3.weight, std=0.02)
        torch.nn.init.zeros_(self.convolve3.bias)
        self.normalize3 = nn.InstanceNorm2d(num_features=512)

        self.convolve4 = spectral_norm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2))
        nn.init.normal_(self.convolve4.weight, std=0.02)
        torch.nn.init.zeros_(self.convolve4.bias)
        self.normalize4 = nn.InstanceNorm2d(num_features=1024)

        self.linear_classifier1 = nn.Linear(256, 1)
        self.linear_classifier2 = nn.Linear(512, 1)
        self.linear_classifier3 = nn.Linear(1024, 1)
        self.linear_classifier4 = nn.Linear(2048, 1)
        self.linear_classifier5 = nn.Linear(9216, 1)

        nn.init.normal_(self.linear_classifier1.weight, std=0.02)
        nn.init.normal_(self.linear_classifier2.weight, std=0.02)
        nn.init.normal_(self.linear_classifier3.weight, std=0.02)
        nn.init.normal_(self.linear_classifier4.weight, std=0.02)
        nn.init.normal_(self.linear_classifier5.weight, std=0.02)

        torch.nn.init.zeros_(self.linear_classifier1.bias)
        torch.nn.init.zeros_(self.linear_classifier2.bias)
        torch.nn.init.zeros_(self.linear_classifier3.bias)
        torch.nn.init.zeros_(self.linear_classifier4.bias)
        torch.nn.init.zeros_(self.linear_classifier5.bias)

        self.linear_projector1 = nn.Linear(9216, 128)
        self.linear_projector2 = nn.Linear(128, self.cont_dim)

        nn.init.normal_(self.linear_projector1.weight, std=0.02)
        nn.init.normal_(self.linear_projector2.weight, std=0.02)

        torch.nn.init.zeros_(self.linear_projector1.bias)
        torch.nn.init.zeros_(self.linear_projector2.bias)

    def forward(self, x, negative_slope=0.2):
        print(x.shape)
        x = x / 127.5 - 1.
        if torch.cuda.is_available():
            x = x + self.noise_generator.sample(sample_shape=x.shape).cuda()
        else:
            x = x + self.noise_generator.sample(sample_shape=x.shape)

        h0 = F.leaky_relu(self.convolve0(x), negative_slope=negative_slope)

        h1 = self.convolve1(h0)
        h1_mean, h1_var = calc_mean_std(h1)
        h1 = self.normalize1(h1)
        d_h1_style = torch.cat((h1_mean, h1_var), 1)
        d_h1_style = d_h1_style.view(d_h1_style.shape[0], -1)
        d_h1_logits = self.linear_classifier1(d_h1_style)
        # d_h1_logits = torch.sigmoid(d_h1)
        h1 = F.leaky_relu(h1, negative_slope=negative_slope)

        h2 = self.convolve2(h1)
        h2_mean, h2_var = calc_mean_std(h2)
        h2 = self.normalize2(h2)
        d_h2_style = torch.cat((h2_mean, h2_var), 1)
        d_h2_style = d_h2_style.view(d_h2_style.shape[0], -1)
        d_h2_logits = self.linear_classifier2(d_h2_style)
        # d_h2_logits = torch.sigmoid(d_h2)
        h2 = F.leaky_relu(h2, negative_slope=negative_slope)

        h3 = self.convolve3(h2)
        h3_mean, h3_var = calc_mean_std(h3)
        h3 = self.normalize3(h3)
        d_h3_style = torch.cat((h3_mean, h3_var), 1)
        d_h3_style = d_h3_style.view(d_h3_style.shape[0], -1)
        d_h3_logits = self.linear_classifier3(d_h3_style)
        # d_h3_logits = torch.sigmoid(d_h3)
        h3 = F.leaky_relu(h3, negative_slope=negative_slope)

        h4 = self.convolve4(h3)
        h4_mean, h4_var = calc_mean_std(h4)
        h4 = self.normalize4(h4)
        d_h4_style = torch.cat((h4_mean, h4_var), 1)
        d_h4_style = d_h4_style.view(d_h4_style.shape[0], -1)
        d_h4_logits = self.linear_classifier4(d_h4_style)
        # d_h4_logits = torch.sigmoid(d_h4)
        h4 = F.leaky_relu(h4, negative_slope=negative_slope)
        h4f = h4.view(h4.shape[0], -1)

        h5 = self.linear_classifier5(h4f)
        h5s = torch.sigmoid(h5)

        encoding = F.leaky_relu(self.linear_projector1(h4f), negative_slope=negative_slope)
        cont_vars = torch.tanh(self.linear_projector2(encoding))

        return h5s, h5, cont_vars, d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits
