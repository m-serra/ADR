import os
import math
from scipy.stats import norm
import pickle
import numpy as np
import tensorflow as tf
from termcolor import colored
import moviepy.editor as mpy
from data_readers.bair_data_reader import BairDataReader
from data_readers.google_push_data_reader import GooglePushDataReader
from models.encoder_decoder import repeat_skips, slice_skips
from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset
from adr import kl_unit_normal
import tensorflow.python.keras.backend as K


def get_data(dataset, mode, dataset_dir, batch_size=32, sequence_length_train=12, sequence_length_test=12,
             shuffle=True, initializable=False):

    assert dataset in ['bair', 'google', 'robonet']
    assert mode in ['train', 'val', 'test']

    if dataset == 'bair':
        d = BairDataReader(dataset_dir=dataset_dir,
                           batch_size=batch_size,
                           use_state=1,
                           sequence_length_train=sequence_length_train,
                           sequence_length_test=sequence_length_test,
                           shuffle=shuffle,
                           batch_repeat=1,
                           initializable=initializable)
    elif dataset == 'google':
        d = GooglePushDataReader(dataset_dir=dataset_dir,  # '/media/Data/datasets/google_push/push/',
                                 batch_size=batch_size,
                                 sequence_length_train=sequence_length_train,
                                 sequence_length_test=sequence_length_test,
                                 shuffle=shuffle,
                                 train_dir_name='push_train',
                                 test_dir_name='push_train',
                                 batch_repeat=1)
    elif dataset == 'robonet':
        train_database = load_metadata(os.path.expanduser('~/'), 'RoboNet/hdf5/train')
        val_database = load_metadata(os.path.expanduser('~/'), 'RoboNet/hdf5/val2')
        train_database = train_database[train_database['robot'] == 'fetch']
        d_train = RoboNetDataset(batch_size=batch_size, dataset_files_or_metadata=train_database,
                                 hparams={'img_size': [64, 64], 'target_adim': 2, 'target_sdim': 3})
        d_val = RoboNetDataset(batch_size=batch_size, dataset_files_or_metadata=val_database,
                               hparams={'img_size': [64, 64], 'target_adim': 2, 'target_sdim': 3})

    d.train_filenames = ['/media/Data/datasets/bair/softmotion30_44k/train/traj_10174_to_10429.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_1024_to_1279.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_10430_to_10685.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_10686_to_10941.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_10942_to_11197.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_11198_to_11453.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_11454_to_11709.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_11710_to_11965.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_11966_to_12221.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_12222_to_12477.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_12478_to_12733.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_12734_to_12989.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_1280_to_1535.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_12990_to_13245.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_13341_to_13596.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_13597_to_13852.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_13853_to_14108.tfrecords',
                         '/media/Data/datasets/bair/softmotion30_44k/train/traj_14109_to_14364.tfrecords']

    d.val_filenames = ['/media/Data/datasets/bair/softmotion30_44k/train/traj_5983_to_6238.tfrecords',
                       '/media/Data/datasets/bair/softmotion30_44k/train/traj_6239_to_6494.tfrecords',
                       '/media/Data/datasets/bair/softmotion30_44k/train/traj_6495_to_6750.tfrecords']

    if dataset == 'robonet':
        frames = tf.squeeze(d_train['images'])  # images, states, and actions are from paired
        actions = d_train['actions']
        states = d_train['states']
        val_frames = tf.squeeze(d_val['images'])
        val_actions = d_val['actions']
        val_states = d_val['states']
        steps = 545
        val_steps = 545
    else:
        steps = d.num_examples_per_epoch(mode) // d.batch_size
        iterator = d.build_tf_iterator(mode=mode)
        input_get_next_op = iterator.get_next()
        frames = input_get_next_op['images']
        actions = input_get_next_op['actions'][:, :, :4]
        states = input_get_next_op['states'][:, :, :3]

    return frames, actions, states, steps, iterator


class ModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, models, criteria, ckpt_dir, filenames, neptune_ckpt=False, keep_all=False):
        super(tf.keras.callbacks.Callback, self).__init__()
        super().__init__()
        self.models = models
        self.criteria = criteria
        self.ckpt_dir = ckpt_dir
        self.filenames = filenames
        self.best_loss = 9999
        self.best_train_loss = 9999
        self.best_train_epoch = 0
        self.best_val_loss = 9999
        self.best_val_epoch = 0
        self.saver = None
        self.neptune_ckpt = neptune_ckpt
        self.keep_all = keep_all

        if type(self.models) is not list:
            self.models = [self.models]
        if type(self.filenames) is not list:
            self.filenames = [self.filenames]

        assert len(models) == len(filenames), 'models and filenames must have the same length'
        assert criteria in ['train_rec', 'val_rec'], 'criteria must be either train_rec or val_rec'

    def on_epoch_end(self, epoch, logs=None):
        # train_loss = logs.get('rec_pred')
        train_loss = logs.get('rec_loss')
        # val_loss = logs.get('val_rec_pred')
        val_loss = logs.get('val_rec_loss')
        self.model_checkpoint(train_loss, val_loss, epoch)

    def model_checkpoint(self, train_loss, val_loss, epoch):

        new_best_train, new_best_val = False, False

        # criteria_map = {'train_rec': train_loss[1], 'val_rec': val_loss[1]}
        criteria_map = {'train_rec': train_loss, 'val_rec': val_loss}

        loss = criteria_map.get(self.criteria)

        if loss < self.best_loss:
        #  if loss < self.best_loss or train_loss < self.best_train_loss:  # --> !!!!
            for m, f in zip(self.models, self.filenames):
                if self.keep_all:
                    f = f.replace('.h5', '') + '_t' + str(train_loss).replace('0.', '') + \
                        '_v' + str(val_loss).replace('0.', '') + '.h5'
                tf.keras.models.save_model(m, os.path.join(self.ckpt_dir, f))
                # m.save_weights(os.path.join(self.ckpt_dir, f))
                if self.neptune_ckpt:
                    neptune.log_artifact(os.path.join(self.ckpt_dir, f))
            self.best_loss = loss

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.best_train_epoch = epoch + 1
            new_best_train = True

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch + 1
            new_best_val = True

        if new_best_train:
            print(colored('Best train loss: %.7f, epoch  %d' %
                          (self.best_train_loss, self.best_train_epoch), 'magenta'))
        if new_best_val:
            print(colored('Best val loss: %.7f, epoch %d' % (self.best_val_loss, self.best_val_epoch), 'green'))
        return


def npy_to_gif(npy, filename, fps=10):
    clip = mpy.ImageSequenceClip(list(npy), fps=fps, )
    clip.write_gif(filename)


def save_gifs(sequence, name, save_dir):

    SEQ_LEN = sequence.shape[1]
    BATCH_SIZE = sequence.shape[0]

    for b in range(BATCH_SIZE):
        k = []
        for i in range(SEQ_LEN):
            k.append(sequence[b, i, :, :, :])

        video = np.stack(k, ) * 255
        npy_to_gif(video, os.path.join(save_dir, name + '_batch_' + str(b) + '.gif'))
    return


def print_loss(loss, loss_names, title=None):
    assert len(loss) == len(loss_names)
    if title is not None:
        print('\n===== ' + title + ' =====')
    c = '  '.join('%s: %.6f' % t for t in zip(loss_names, loss))
    print(c)


def plot_multiple(model, model_decoder, bs, grid_z, iter_, ckpt_dir, nz, aggressive, steps=8):

    loc = np.zeros(nz)
    scale = np.ones(nz)
    # prior = torch.distributions.normal.Normal(loc=loc, scale=scale)
    prior = norm(loc=loc, scale=scale)

    infer_posterior_mean = []
    report_loss_kl = report_mi = report_num_sample = 0

    for s in range(steps):
        pred, data, mu, logvar, skips_0, skips_1, skips_2, skips_3 = model.predict(x=None, steps=1)
        _skips = [skips_0[:, :1], skips_1[:, :1], skips_2[:, :1], skips_3[:, :1]]
        skips = [np.repeat(s, repeats=400, axis=1) for s in _skips]
        print('SKIPS', skips[0].shape)

        grid_z_batched = np.repeat(np.expand_dims(grid_z, axis=0), repeats=bs, axis=0)
        grid_z_batched = np.repeat(grid_z_batched, repeats=154, axis=-1)  # --> ???????????????

        print('grid z', grid_z_batched.shape)
        # print('OUT', grid_z_batched, skips)
        grid_pred = model_decoder.predict(x=[grid_z_batched, skips[0], skips[1], skips[2], skips[3]])
        print('GRID PRED', grid_pred.shape)
        # --> THIS DOESNT INFLUENCE THE PLOTS
        report_loss_kl += np.sum(KL(mu, logvar))
        # report_loss_kl += model.KL(data).sum().item()

        report_num_sample += bs

        # --> THIS DOESNT INFLUENCE THE PLOTS
        # report_mi += calc_mi_q(model, data) * bs
        report_mi += calc_mi(mu, logvar) * bs   # --> was getting negative MI

        # [batch, 1]
        # --> ===============================================
        posterior_mean = calc_model_posterior_mean(prior, grid_pred, data, grid_z, bs)  # should output shape [bs, nz]
        # print('Posterior mean:', posterior_mean)
        print('Posterior mean:', posterior_mean.shape, np.max(posterior_mean), np.min(posterior_mean), np.mean(posterior_mean))

        # print('Posterior Mean:', posterior_mean.shape)
        posterior_mean = np.mean(posterior_mean, axis=1)  # # --> mean added by me

        # infer_mean = calc_infer_mean(model, data)  # shape [bs, nz]
        infer_mean = mu  # shape [bs, steps, nz]
        infer_mean = np.mean(infer_mean, axis=1)  # --> mean added by me to average steps
        # print('Infer Mean', infer_mean.shape)

        infer_posterior_mean.append(np.concatenate([posterior_mean, infer_mean], axis=1))

    # [*, 2]
    infer_posterior_mean = np.concatenate(infer_posterior_mean, axis=0)

    # save_path = os.path.join(ckpt_dir, 'aggr%d_iter%d_multiple.pickle' % (int(aggressive), iter_))
    save_path = os.path.join(os.path.expanduser('~/'), 'adr/plot_data/multiple/aggr%d_iter%d_multiple.pickle'
                             % (int(aggressive), iter_))

    # print('True posterior:', infer_posterior_mean[:, 0])
    # print('Infer posterior:', infer_posterior_mean[:, 1])
    save_data = {'posterior': infer_posterior_mean[:, 0],
                 'inference': infer_posterior_mean[:, 1],
                 'kl': report_loss_kl / report_num_sample,
                 'mi': report_mi / report_num_sample}
    pickle.dump(save_data, open(save_path, 'wb'))


def KL(mu, logvar):
    kl = - 0.5 * np.sum(1.0 + logvar - np.square(mu) - np.exp(logvar), axis=(-1, -2))
    return kl


def calc_infer_mean(model, data):
    _, _, mu, _ = model.predict(data, steps=1)
    return mu


def reparameterize(mu, logvar, z_samples=1):

    std = np.exp(0.5 * logvar)

    if z_samples > 1:
        mu = np.repeat(np.expand_dims(mu, axis=1), z_samples, axis=1)
        std = np.repeat(np.expand_dims(std, axis=1), z_samples, axis=1)

    epsilon = np.random.normal(size=np.shape(logvar), loc=0.0, scale=1.0)

    return mu + std * epsilon


def calc_mi(mu, logvar):
    """Approximate the mutual information between x and z
    I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
    Returns: Float
    """
    # [x_batch, nz]
    z_batch = 1

    mu = np.mean(mu, axis=1)
    logvar = np.mean(logvar, axis=1)

    bs, nz = mu.shape
    # print('MU SHAPE', mu.shape)  # correct shape

    # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
    neg_entropy = np.mean(-0.5 * nz * np.log(2 * np.pi) - 0.5 * np.sum(1 + logvar, axis=-1))
    # print('NEG ENTROPY:', neg_entropy.shape)
    # print('NEG ENTROPY:', neg_entropy)

    # [z_batch, nz]
    z_samples = reparameterize(mu, logvar, z_samples=z_batch)
    # print('Z_SAMPLES', z_samples.shape)  # correct shape

    # [1, x_batch, nz]
    mu, logvar = np.expand_dims(mu, axis=0), np.expand_dims(logvar, axis=0)
    var = np.exp(logvar)

    # print('MU', mu.shape)
    # print('LOGVAR', logvar.shape)

    # (z_batch, x_batch, nz)
    dev = z_samples - mu
    # print('DEV', dev.shape)

    # (z_batch, x_batch)
    # log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
    log_density = -0.5 * np.sum(np.square(dev) / var, axis=-1) - 0.5 * (nz * np.log(2 * np.pi) + np.sum(logvar, axis=-1))
    # print('LOG DENS', log_density.shape)

    # log q(z): aggregate posterior
    # [z_batch]
    log_qz = log_sum_exp(log_density, dim=1) - math.log(bs)
    # print('LOG QZ', log_qz)

    return neg_entropy - log_qz.mean(-1)


def calc_mii(mu, logvar):
    x_batch, nz = mu.size()

    # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
    neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

    # [z_batch, 1, nz]
    z_samples = reparameterize(mu, logvar, 1)

    # [1, x_batch, nz]
    mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
    var = logvar.exp()

    # (z_batch, x_batch, nz)
    dev = z_samples - mu

    # (z_batch, x_batch)
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
        0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

    # log q(z): aggregate posterior
    # [z_batch]
    log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

    return (neg_entropy - log_qz.mean(-1)).item()


def calc_model_posterior_mean(prior, pred, data, grid_z, bs):
    """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
    Args:
        grid_z: different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        data: [batch, *]
    Returns: Tensor1
        Tensor1: the mean value tensor with shape [batch, nz]
    """

    # [batch, K^2]
    log_posterior = eval_log_model_posterior(prior, pred, data, grid_z, bs)
    posterior = np.exp(log_posterior)

    # [batch, nz]
    return np.expand_dims(posterior, axis=2) * np.sum(np.expand_dims(grid_z, axis=0), axis=1)
    # return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)


def eval_log_model_posterior(prior, pred, data, grid_z, bs):
    """perform grid search to calculate the true posterior
     this function computes p(z|x)
    Args:
        grid_z: tensor
            different z points that will be evaluated, with
            shape (k^2, nz), where k=(zmax - zmin)/pace
    Returns: Tensor
        Tensor: the log posterior distribution log p(z|x) with
                shape [batch_size, K^2]
    """
    print('===== grid_z', grid_z.shape)
    # (k^2, nz) -> (batch_size, k^2, nz)
    grid_z = np.repeat(np.expand_dims(grid_z, axis=0), bs, axis=0)

    # (batch_size, k^2)
    log_comp = eval_complete_ll(prior, pred, data, grid_z)

    # normalize to posterior
    log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

    return log_posterior


def eval_complete_ll(prior, pred, data, z):
    """compute log p(z,x)
       data: is the true target image
       z: a grid of latent points from which to make a reconstruction
    """
    # [batch, nsamples]
    log_prior = eval_prior_dist(prior, z)
    log_gen = eval_cond_ll(pred, data)  # for each point in z reconstruct and compare with data
    print('log_prior:', log_prior.shape, log_prior.mean(), log_prior.max(), log_prior.min())
    print('log_gen:', log_gen.shape, log_gen.mean(), log_gen.max(), log_gen.min())
    return log_prior + log_gen


def eval_cond_ll(pred, data):
    """compute log p(x|z)
    """
    # return self.decoder.log_probability(x, z)
    return np.mean(np.square(pred - data), (2, 3, 4))


def eval_prior_dist(prior, zrange):
    """perform grid search to calculate the true posterior
    Args:
        prior:
        zrange: tensor
            different z points that will be evaluated, with
            shape (k^2, nz), where k=(zmax - zmin)/space
    """
    # (k^2)
    # return prior.log_prob(zrange).sum(dim=-1)
    return np.sum(prior.logpdf(zrange), axis=-1)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        # m, _ = torch.max(value, dim=dim, keepdim=True)
        m = np.max(value, axis=dim, keepdims=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        # return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        return m + np.log(np.sum(np.exp(value0), axis=dim, keepdims=keepdim))
    else:
        # m = torch.max(value)
        m = np.max(value)
        # sum_exp = torch.sum(torch.exp(value - m))
        sum_exp = np.sum(np.exp(value - m))
        # return m + torch.log(sum_exp)
        return m + np.log(sum_exp)


def generate_grid(zmin, zmax, dz, ndim=2):
    """generate a 1- or 2-dimensional grid
    Returns: Tensor, int
        Tensor: The grid tensor with shape (k^2, 2),
            where k=(zmax - zmin)/dz
        int: k
    """
    # import torch
    # cuda = False
    if ndim == 2:  # --> this not adapted to TF
        # x = torch.arange(zmin, zmax, dz)
        x = np.arange(start=zmin, stop=zmax, step=dz)
        k = x.shape(0)

        x1 = np.expand_dims(x, axis=1).repeat(1, k).view(-1)
        x2 = x.repeat(k)

        # return torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1).to(device), k
    elif ndim == 1:
        return np.expand_dims(np.arange(start=zmin, stop=zmax, step=dz), axis=1)
