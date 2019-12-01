import torch
import torch.nn as nn
import torch.nn.functional as F
from hologan import util


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.const = nn.Parameter(torch.empty(size=(1, 512, 4, 4, 4)), requires_grad=True)
        nn.init.normal_(self.const, std=0.02)

        self.mlp0 = nn.Linear(128, 512 * 2)
        nn.init.normal_(self.mlp0.weight, std=0.02)
        torch.nn.init.zeros_(self.mlp0.bias)

        self.mlp1 = nn.Linear(128, 256 * 2)
        self.dcv3d1 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)

        nn.init.normal_(self.mlp1.weight, std=0.02)
        torch.nn.init.zeros_(self.mlp1.bias)
        nn.init.normal_(self.dcv3d1.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv3d1.bias)

        self.mlp2 = nn.Linear(128, 128 * 2)
        self.dcv3d2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)

        nn.init.normal_(self.mlp2.weight, std=0.02)
        torch.nn.init.zeros_(self.mlp2.bias)
        nn.init.normal_(self.dcv3d2.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv3d2.bias)

        self.dcv3d3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dcv3d4 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        nn.init.normal_(self.dcv3d3.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv3d3.bias)
        nn.init.normal_(self.dcv3d4.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv3d4.bias)

        self.dcv2d1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)

        nn.init.normal_(self.dcv2d1.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv2d1.bias)

        self.mlp3 = nn.Linear(128, 256 * 2)
        self.dcv2d2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2)

        nn.init.normal_(self.mlp3.weight, std=0.02)
        torch.nn.init.zeros_(self.mlp3.bias)
        nn.init.normal_(self.dcv2d2.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv2d2.bias)

        self.mlp4 = nn.Linear(128, 64 * 2)
        self.dcv2d3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2)

        nn.init.normal_(self.mlp4.weight, std=0.02)
        torch.nn.init.zeros_(self.mlp4.bias)
        nn.init.normal_(self.dcv2d3.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv2d3.bias)

        self.mlp5 = nn.Linear(128, 32 * 2)
        self.dcv2d4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=1)

        nn.init.normal_(self.mlp5.weight, std=0.02)
        torch.nn.init.zeros_(self.mlp5.bias)
        nn.init.normal_(self.dcv2d4.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv2d4.bias)

        self.dcv2d5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=1)
        nn.init.normal_(self.dcv2d5.weight, std=0.02)
        torch.nn.init.zeros_(self.dcv2d5.bias)

    def forward(self, z, pose, negative_slope=0.2):

        w_tile = self.const.repeat((z.shape[0], 1, 1, 1, 1))
        z0 = F.leaky_relu(self.mlp0(z), negative_slope=negative_slope)
        h0 = util.adaptive_instance_normalization(w_tile, z0)
        h0 = F.leaky_relu(h0, negative_slope=negative_slope)

        h1 = self.dcv3d1(h0)
        z1 = F.leaky_relu(self.mlp1(z), negative_slope=negative_slope)
        h1 = util.adaptive_instance_normalization(h1, z1)
        h1 = F.leaky_relu(h1, negative_slope=negative_slope)

        h2 = self.dcv3d2(h1)
        z2 = F.leaky_relu(self.mlp2(z), negative_slope=negative_slope)
        h2 = util.adaptive_instance_normalization(h2, z2)
        h2 = F.leaky_relu(h2, negative_slope=negative_slope)

        h2_rotated = F.grid_sample(h2, F.affine_grid(pose, h2.size(), align_corners=True), align_corners=True)

        h2_proj1 = F.leaky_relu(self.dcv3d3(h2_rotated), negative_slope=negative_slope)
        h2_proj2 = F.leaky_relu(self.dcv3d4(h2_proj1), negative_slope=negative_slope)

        h2_2d = h2_proj2.view(h2_proj2.shape[0], h2_proj2.shape[1] * h2_proj2.shape[2], h2_proj2.shape[3], h2_proj2.shape[4])

        h3 = F.leaky_relu(self.dcv2d1(h2_2d), negative_slope=negative_slope)

        h4 = self.dcv2d2(h3)
        z3 = F.leaky_relu(self.mlp3(z), negative_slope=negative_slope)
        h4 = util.adaptive_instance_normalization(h4, z3)
        h4 = F.leaky_relu(h4, negative_slope=negative_slope)

        h5 = self.dcv2d3(h4)
        z4 = F.leaky_relu(self.mlp4(z), negative_slope=negative_slope)
        h5 = util.adaptive_instance_normalization(h5, z4)
        h5 = F.leaky_relu(h5, negative_slope=negative_slope)

        h6 = self.dcv2d4(h5)
        z5 = F.leaky_relu(self.mlp5(z), negative_slope=negative_slope)
        h6 = util.adaptive_instance_normalization(h6, z5)
        h6 = F.leaky_relu(h6, negative_slope=negative_slope)

        h7 = self.dcv2d5(h6)
        output = torch.tanh(h7)

        return output
