import os
import time
import torch
from torch import optim
from hologan import util
from itertools import chain
from hologan.generator import Generator
from hologan.discriminator import Discriminator
from torchvision.utils import save_image


class HoloGAN:

    def __init__(self,
                 angles=[-30, +30, 0, 0, 0, 0],
                 cont_dim=128,
                 batch_size=32,
                 use_cuda='detect',
                 use_multiple_gpus=True,
                 update_g_every=5,
                 style_lambda=1.,
                 latent_lambda=1.,
                 opt_d_args={'lr': 0.00005, 'betas': (0.5, 0.999)},
                 opt_g_args={'lr': 0.00005, 'betas': (0.5, 0.999)},
                 data_dir="./images"
                 ):

        self.use_multiple_gpus = use_multiple_gpus
        self.latent_lambda = latent_lambda
        self.style_lambda = style_lambda
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            self.use_cuda = torch.cuda.is_available()

        self.cont_dim = cont_dim
        self.batch_size = batch_size

        self.update_g_every = update_g_every

        self.angles = util.angles_to_dict(angles)

        self.G = Generator()
        self.D = Discriminator(self.cont_dim)

        self.g_optim = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), **opt_g_args)
        self.d_optim = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), **opt_d_args)

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

            if self.use_multiple_gpus:
                self.G = torch.nn.DataParallel(self.G)
                self.D = torch.nn.DataParallel(self.D)

        self.last_epoch = 0

        self.data_loader, self.dataset_size = util.get_data_loader(data_dir, self.batch_size)

    def train_batch(self, z, x, itr, epoch, plot_batch_gradients=False):

        self.G.train()
        self.D.train()

        self.g_optim.zero_grad()
        self.d_optim.zero_grad()

        angles = util.sample_angles(z.shape[0], **self.angles)
        thetas = util.get_theta(angles)

        if self.use_cuda:
            z = z.cuda()
            thetas = thetas.cuda()

        fake = self.G(z, thetas)
        h5s, h5, cont_vars, _, _, _, _ = self.D(fake)
        aux_loss = self.latent_lambda * util.mse_loss(z, cont_vars)
        # Train Generator
        gen_loss = util.binary_loss(h5, 1.) + aux_loss
        gen_loss.backward()
        self.g_optim.step()

        # 2nd time
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()

        angles = util.sample_angles(z.shape[0], **self.angles)
        thetas = util.get_theta(angles)

        if self.use_cuda:
            z = z.cuda()
            thetas = thetas.cuda()

        fake = self.G(z, thetas)
        h5s, h5, cont_vars, _, _, _, _ = self.D(fake)
        aux_loss = self.latent_lambda * util.mse_loss(z, cont_vars)
        # Train Generator
        gen_loss = util.binary_loss(h5, 1.) + aux_loss
        gen_loss.backward()
        self.g_optim.step()


        # Train Discriminator
        self.d_optim.zero_grad()

        h5s_f, h5_f, cont_vars, d_h1_logits_f, d_h2_logits_f, d_h3_logits_f, d_h4_logits_f = self.D(fake.detach())
        h5s_r, h5_r, _, d_h1_logits_r, d_h2_logits_r, d_h3_logits_r, d_h4_logits_r = self.D(x)

        d_r_loss = util.binary_loss(h5_r, 1.)
        d_f_loss = util.binary_loss(h5_f, 0.)
        d_h1_loss = self.style_lambda * (util.binary_loss(d_h1_logits_r, 1.) + util.binary_loss(d_h1_logits_f, 0.))
        d_h2_loss = self.style_lambda * (util.binary_loss(d_h2_logits_r, 1.) + util.binary_loss(d_h2_logits_f, 0.))
        d_h3_loss = self.style_lambda * (util.binary_loss(d_h3_logits_r, 1.) + util.binary_loss(d_h3_logits_f, 0.))
        d_h4_loss = self.style_lambda * (util.binary_loss(d_h4_logits_r, 1.) + util.binary_loss(d_h4_logits_f, 0.))
        aux_loss = self.latent_lambda * util.mse_loss(z, cont_vars)

        d_loss = d_r_loss + d_f_loss + d_h1_loss + d_h2_loss + d_h3_loss + d_h4_loss + aux_loss

        d_loss.backward()
        self.d_optim.step()

        # plot
        if plot_batch_gradients:
            util.plot_grad_flow(self.G.named_parameters(), "generator", itr, epoch)
            util.plot_grad_flow(self.D.named_parameters(), "discriminator", itr, epoch)

        losses = {
            'g_loss': gen_loss.item(),
            'd_loss': d_loss.item() / 2.,
            'z_loss': aux_loss.item()
        }

        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }

        return losses, outputs

    def prepare_batch(self, batch):
        if len(batch) != 1:
            raise Exception("Expected batch to only contain X")
        X_batch = batch[0].float()
        if self.use_cuda:
            X_batch = X_batch.cuda()
        return [X_batch]

    def sample(self, const_z=False, seed=None):
        """Return a sample G(z)"""
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            z = util.sample_z(self.batch_size, const_z=const_z, dist="normal")
            angles = util.sample_angles(z.shape[0], **self.angles, sequential=True)
            thetas = util.get_theta(angles)
            if self.use_cuda:
                z = z.cuda()
                thetas = thetas.cuda()
            gz = self.G(z, thetas)
        return gz

    def train(self,
              epochs=10,
              model_dir="./checkpoints",
              result_dir="./results",
              save_every=5,
              sample_every=5,
              verbose=True,
              plot_gradients=True,
              plot_batch_gradients=False):

        loss = {"g": [], "d": [], "z": []}

        for epoch in range(self.last_epoch, epochs):
            epoch_start_time = time.time()

            running_loss_g = 0.0
            running_loss_d = 0.0
            running_loss_z = 0.0

            b = 1
            for batch, _ in self.data_loader:
                if type(batch) not in [list, tuple]:
                    batch = [batch]
                batch = self.prepare_batch(batch)
                z_batch = util.sample_z(batch[0].size()[0])
                losses, outputs = self.train_batch(z_batch, *batch, itr=b, epoch=epoch, plot_batch_gradients=plot_batch_gradients)
                b += batch[0].size()[0]

                running_loss_g += losses['g_loss']
                running_loss_d += losses['d_loss']
                running_loss_z += losses['z_loss']

                if verbose:
                    print("{} / {}".format(b, self.dataset_size))

            epoch_loss_g = running_loss_g / self.dataset_size
            epoch_loss_d = running_loss_d / self.dataset_size
            epoch_loss_z = running_loss_z / self.dataset_size

            loss["g"].append(epoch_loss_g)
            loss["d"].append(epoch_loss_d)
            loss["z"].append(epoch_loss_z)

            print("Epoch:", epoch+1, "| g loss:", epoch_loss_g, "| d loss:", epoch_loss_d, "| z loss:", epoch_loss_z, "| time:", round(time.time() - epoch_start_time, 2), "sec")

            if plot_gradients:
                util.plot_grad_flow(self.G.named_parameters(), "epoch_generator", 9999, epoch)
                util.plot_grad_flow(self.D.named_parameters(), "epoch_discriminator", 9999, epoch)

            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)

            if (epoch+1) % sample_every == 0 and model_dir is not None:
                gz = self.sample()
                filename = str(epoch + 1) + ".jpg"
                save_image(gz, os.path.join(result_dir, filename))

    def save(self, filename, epoch, legacy=False):
        dd = dict()
        dd['g'] = self.G.state_dict()
        dd['d'] = self.D.state_dict()
        dd['d_optim'] = self.d_optim.state_dict()
        dd['g_optim'] = self.g_optim.state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename, legacy=False, ignore_d=False):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        dd = torch.load(filename, map_location=map_location)
        self.G.load_state_dict(dd['g'])
        if not ignore_d:
            self.D.load_state_dict(dd['d'])
            self.d_optim.load_state_dict(dd['d_optim'])

        self.g_optim.load_state_dict(dd['g_optim'])
        self.last_epoch = dd['epoch']
