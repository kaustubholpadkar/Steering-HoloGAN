from hologan import HoloGAN

angles = [-30, +30, 0, 0, 0, 0]
cont_dim = 128
batch_size = 32
use_cuda = 'detect'
use_multiple_gpus = True
update_g_every = 5
style_lambda = 1.
latent_lambda = 1.
opt_d_args = {'lr': 0.00005, 'betas': (0.5, 0.999)}
opt_g_args = {'lr': 0.00005, 'betas': (0.5, 0.999)}
data_dir = "./images"

gan = HoloGAN(
    angles, cont_dim, batch_size, use_cuda, use_multiple_gpus, update_g_every, style_lambda, latent_lambda, opt_d_args, opt_g_args, data_dir
)

epochs = 10,
model_dir = "./checkpoints",
result_dir = "./results",
save_every = 5,
sample_every = 5,
verbose = True,
plot_gradients = True,
plot_batch_gradients = False

gan.train(
    epochs, model_dir, result_dir, save_every, sample_every, verbose, plot_gradients, plot_batch_gradients
)

save_file = "./trained_hologan.pkl"

gan.save(save_file, epochs)
