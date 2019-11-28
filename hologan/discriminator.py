import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, cont_dim):
        super(Discriminator, self).__init__()

        self.cont_dim = cont_dim

        self.convolve0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=2)

        self.convolve1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.normalize1 = nn.InstanceNorm2d(num_features=128)

        self.convolve2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.normalize2 = nn.InstanceNorm2d(num_features=256)

        self.convolve3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.normalize3 = nn.InstanceNorm2d(num_features=512)

        self.convolve4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2)
        self.normalize4 = nn.InstanceNorm2d(num_features=1024)

        self.linear_classifier1 = nn.Linear(256, 1)
        self.linear_classifier2 = nn.Linear(512, 1)
        self.linear_classifier3 = nn.Linear(1024, 1)
        self.linear_classifier4 = nn.Linear(2048, 1)
        self.linear_classifier5 = nn.Linear(9216, 1)

        self.linear_projector1 = nn.Linear(9216, 128)
        self.linear_projector2 = nn.Linear(128, self.cont_dim)

    def moments(self, x):
        mean = torch.mean(x.view(x.shape[0], x.shape[1], 1, -1, ), dim=3, keepdim=True)
        variance = torch.var(x.view(x.shape[0], x.shape[1], 1, -1, ), dim=3, keepdim=True)
        return mean, variance

    def forward(self, x, negative_slope=0.2):

        h0 = F.leaky_relu(self.convolve0(x), negative_slope=negative_slope)

        h1 = self.convolve1(h0)
        h1_mean, h1_var = self.moments(h1)
        h1 = self.normalize1(h1)
        d_h1_style = torch.cat((h1_mean, h1_var), 1)
        d_h1_style = d_h1_style.view(d_h1_style.shape[0], -1)
        d_h1 = self.linear_classifier1(d_h1_style)
        d_h1_logits = torch.sigmoid(d_h1)
        h1 = F.leaky_relu(h1, negative_slope=negative_slope)

        h2 = self.convolve2(h1)
        h2_mean, h2_var = self.moments(h2)
        h2 = self.normalize2(h2)
        d_h2_style = torch.cat((h2_mean, h2_var), 1)
        d_h2_style = d_h2_style.view(d_h2_style.shape[0], -1)
        d_h2 = self.linear_classifier2(d_h2_style)
        d_h2_logits = torch.sigmoid(d_h2)
        h2 = F.leaky_relu(h2, negative_slope=negative_slope)

        h3 = self.convolve3(h2)
        h3_mean, h3_var = self.moments(h3)
        h3 = self.normalize3(h3)
        d_h3_style = torch.cat((h3_mean, h3_var), 1)
        d_h3_style = d_h3_style.view(d_h3_style.shape[0], -1)
        d_h3 = self.linear_classifier3(d_h3_style)
        d_h3_logits = torch.sigmoid(d_h3)
        h3 = F.leaky_relu(h3, negative_slope=negative_slope)

        h4 = self.convolve4(h3)
        h4_mean, h4_var = self.moments(h4)
        h4 = self.normalize4(h4)
        d_h4_style = torch.cat((h4_mean, h4_var), 1)
        d_h4_style = d_h4_style.view(d_h4_style.shape[0], -1)
        d_h4 = self.linear_classifier4(d_h4_style)
        d_h4_logits = torch.sigmoid(d_h4)
        h4 = F.leaky_relu(h4, negative_slope=negative_slope)
        h4f = h4.view(h4.shape[0], -1)

        h5 = self.linear_classifier5(h4f)
        h5s = torch.sigmoid(h5)

        encoding = F.leaky_relu(self.linear_projector1(h4f), negative_slope=negative_slope)
        cont_vars = F.tanh(self.linear_projector2(encoding))

        return h5s, h5, cont_vars, d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits
