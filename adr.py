import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import RepeatVector
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.metrics import binary_accuracy
from models.encoder_decoder import repeat_skips
from models.encoder_decoder import slice_skips
from models.lstm import lstm_initial_state_zeros
import tensorflow.python.keras.backend as K


def get_ins(frames, actions, states, gaussian=True, units=0, layers=0):

    initial_state = None
    frame_inputs = Input(tensor=frames)
    ins = [frame_inputs]

    if actions is not None:
        action_inputs = Input(tensor=actions, name='a_input')
        ins.append(action_inputs)
        action_state = action_inputs  # only using actions
    if states is not None:
        state_inputs = Input(tensor=states, name='s_input')
        ins.append(state_inputs)
        action_state = state_inputs  # only using states
    if actions is not None and states is not None:
        action_state = K.concatenate([action_inputs, state_inputs], axis=-1)  # using actions and states

    if gaussian:
        bs = frames.shape[0]
        initial_state = lstm_initial_state_zeros(units=units, n_layers=layers, batch_size=bs)
        ins.append(initial_state)

    return frame_inputs, action_state, initial_state, ins


def adr_ao(frames, actions, states, context_frames, Ec, A, D, learning_rate, gaussian=False, kl_weight=None, L=None,
           lstm_units=None, lstm_layers=None, training=True, reconstruct_random_frames=False):

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]

    frame_inputs, action_state, initial_state, ins = get_ins(frames, actions, states, gaussian=gaussian,
                                                             units=lstm_units, layers=lstm_layers)

    # random context frames
    rand_index = tf.random.uniform(shape=(), minval=0, maxval=seq_len - context_frames + 1, dtype='int32')
    xc_0 = tf.slice(frame_inputs, (0, rand_index, 0, 0, 0), (-1, context_frames, -1, -1, -1))
    xc_1 = tf.slice(frame_inputs, (0, 0, 0, 0, 0), (-1, context_frames, -1, -1, -1))
    x_to_recover = frame_inputs
    n_frames = seq_len

    # ===== Build the model
    hc_0, skips_0 = Ec(xc_0)
    hc_1, _ = Ec(xc_1)

    hc_0 = tf.slice(hc_0, (0, context_frames - 1, 0), (-1, 1, -1))
    hc_1 = tf.slice(hc_1, (0, context_frames - 1, 0), (-1, 1, -1))
    skips = slice_skips(skips_0, start=context_frames - 1, length=1)

    if reconstruct_random_frames:
        a_s_dim = action_state.shape[-1]
        rand_index_1 = tf.random.uniform(shape=(), minval=0, maxval=seq_len-context_frames+1, dtype='int32')
        action_state = tf.slice(action_state, (0, 0, 0), (bs, rand_index_1+1, a_s_dim))
        x_to_recover = tf.slice(frames, (0, rand_index_1+1, 0, 0, 0), (bs, 1, w, h, c))
        n_frames = rand_index_1 + 1
    else:
        skips = repeat_skips(skips, seq_len)

    ha = A(action_state)
    hc_repeat = RepeatVector(n_frames)(tf.squeeze(hc_0, axis=1))

    if gaussian:
        hc_ha = K.concatenate([hc_repeat, ha], axis=-1)
        z, mu, logvar, state = L([hc_ha, initial_state])
        z = mu if training is False else z
        hc_ha = K.concatenate([hc_repeat, ha, z], axis=-1)
    else:
        hc_ha = K.concatenate([hc_repeat, ha], axis=-1)

    if reconstruct_random_frames:
        _, hc_ha = tf.split(hc_ha, [-1, 1], axis=1)
        if gaussian:
            _, mu = tf.split(mu, [-1, 1], axis=1)
            _, logvar = tf.split(logvar, [-1, 1], axis=1)

    x_recovered = D([hc_ha, skips])

    rec_loss = mean_squared_error(x_to_recover, x_recovered)
    sim_loss = mean_squared_error(hc_0, hc_1)

    model = Model(inputs=ins, outputs=x_recovered)
    model.add_metric(rec_loss, name='rec_loss', aggregation='mean')
    model.add_metric(sim_loss, name='sim_loss', aggregation='mean')

    if gaussian:
        kl_loss = kl_unit_normal(mu, logvar)
        model.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        model.add_loss(K.mean(rec_loss) + K.mean(sim_loss) + kl_weight * K.mean(kl_loss))
    else:
        model.add_loss(K.mean(rec_loss) + K.mean(sim_loss))

    model.compile(optimizer=Adam(lr=learning_rate))

    return model


def build_O_model(self, frames, actions, states, context_frames, Ec, Eo, A, Do, Da, La=None, Lo=None, AD=None,
                  gaussian_a=False, gaussian_o=True, lstm_units=256, lstm_o_units=256, lstm_layers=1,
                  ae_learning_rate=0.0, disc_learning_rate=0.0, disc_weight=0.0, kl_weight=0.0, L=None):

    batch_size, seq_len, w, h, c = [int(s) for s in frames.shape]
    half = batch_size // 2
    a_dim = actions.shape[-1]
    s_dim = states.shape[-1]

    frame_inputs = Input(tensor=frames, name='frame_inputs')
    if actions is not None:
        action_inputs = Input(tensor=actions, name='action_inputs')
    if states is not None:
        state_inputs = Input(tensor=states, name='state_inputs')

    # ===== Select the frames to be used
    rand_index_0 = tf.Variable(tf.random_uniform((), minval=0,
                                                 maxval=seq_len - context_frames,
                                                 dtype='int32')).initialized_value()  # --> !!!!!
    xc_0 = tf.slice(frame_inputs, (0, rand_index_0, 0, 0, 0), (-1, context_frames, -1, -1, -1))
    # xc_0 = tf.slice(frame_inputs, (0, 0, 0, 0, 0), (-1, context_frames, -1, -1, -1))

    # --> change minval to 0
    rand_index = tf.Variable(
        tf.random_uniform((), minval=0, maxval=seq_len, dtype='int32')).initialized_value()  # --> !!!!!

    if gaussian_o:
        # xo = tf.slice(frame_inputs, (0, 0, 0, 0, 0), (-1, rand_index+1, -1, -1, -1))
        xo = frame_inputs
    else:
        # xo = tf.slice(frame_inputs, (0, rand_index, 0, 0, 0), (-1, 1, -1, -1, -1))
        xo = frame_inputs

    if actions is not None:
        # a = tf.slice(action_inputs, (0, 0, 0), (batch_size, rand_index+1, a_dim))
        a = action_inputs
        a_in = a
        ins = [frame_inputs, action_inputs]
    if states is not None:
        # s = tf.slice(state_inputs, (0, 0, 0), (batch_size, rand_index + 1, s_dim))
        s = state_inputs
        a_in = s
        ins = [frame_inputs, state_inputs]
    if actions is not None and states is not None:
        a_in = K.concatenate([a, s], axis=-1)
        ins = [frame_inputs, action_inputs, state_inputs]

    if gaussian_a:
        initial_state_a = lstm_initial_state_zeros(units=lstm_units, n_layers=lstm_layers, batch_size=batch_size)
        ins.append(initial_state_a)
    if gaussian_o:
        initial_state_o = lstm_initial_state_zeros(units=lstm_o_units, n_layers=lstm_layers, batch_size=batch_size)
        ins.append(initial_state_o)

    # initial_state = lstm_initial_state_zeros(units=512, n_layers=2, batch_size=batch_size)  # --> !!!
    # ins.append(initial_state)

    # == Autoencoder
    hc, skips = Ec(xc_0)
    hc_0 = tf.slice(hc, (0, context_frames - 1, 0), (batch_size, 1, -1))

    # hc_repeat = RepeatVector(rand_index + 1)(tf.squeeze(hc_0, axis=1))
    hc_repeat = RepeatVector(seq_len)(tf.squeeze(hc_0, axis=1))

    skips_last = slice_skips(skips, start=context_frames - 1, length=1)

    ha = A(a_in)

    hc_ha = K.concatenate([hc_repeat, ha], axis=-1)

    if gaussian_a:
        _, za, _, _ = La([hc_ha, initial_state_a])  # here z is taken as the mean
        # hc_ha = K.concatenate([hc_0, ha, za], axis=-1)
        hc_ha = K.concatenate([hc_repeat, ha, za], axis=-1)

    if gaussian_o:
        skips_repeat = repeat_skips(skips_last, ntimes=seq_len)
        skips_repeat = slice_skips(skips_repeat, start=0, length=rand_index + 1)
    else:  # all x_rec_a are needed to initialize LSTM hidden state
        # _, hc_ha = tf.split(hc_ha, [-1, 1], axis=1)
        # skips_repeat = skips_last
        skips_repeat = repeat_skips(skips_last, ntimes=seq_len)

    x_rec_a = Da([hc_ha, skips_repeat])

    x_rec_a_pos = K.relu(xo - x_rec_a)
    x_rec_a_neg = K.relu(x_rec_a - xo)
    xo_rec_a = K.concatenate([x_rec_a_pos, x_rec_a_neg], axis=-1)

    ho, _ = Eo(xo_rec_a)

    if gaussian_o:
        h = K.concatenate([hc_repeat, ha, ho], axis=-1)
        zo, mu, logvar, _ = Lo([h, initial_state_o])
        h = K.concatenate([hc_repeat, ha, ho, zo], axis=-1)

        # for computing the loss use only the last timestep
        _, h = tf.split(h, [-1, 1], axis=1)
        _, x_rec_a = tf.split(x_rec_a, [-1, 1], axis=1)
        _, x_rec_a_pos = tf.split(x_rec_a_pos, [-1, 1], axis=1)
        _, x_rec_a_neg = tf.split(x_rec_a_neg, [-1, 1], axis=1)
        _, xo = tf.split(xo, [-1, 1], axis=1)
        _, mu = tf.split(mu, [-1, 1], axis=1)
        _, logvar = tf.split(logvar, [-1, 1], axis=1)
    else:
        # _, ha = tf.split(ha, [-1, 1], axis=1)                                        # single reconstruction
        # h = K.concatenate([hc_0, ha, ho], axis=-1)                                   # single reconstruction
        h = K.concatenate([hc_repeat, ha, ho], axis=-1)  # multiple reconstruction

    # x_err = Do([h, skips_last])
    x_err = Do([h, skips_repeat])

    x_err_pos = x_err[:, :, :, :, :3]
    x_err_neg = x_err[:, :, :, :, 3:]
    x_recovered = x_err_pos - x_err_neg
    x_target = xo - x_rec_a
    x_target_pos = x_rec_a_pos
    x_target_neg = x_rec_a_neg

    # == Discriminator
    # == Shuffle half of hp_sd_1 in the batch_size dim to create video mismatch in relation to hp_sd_0
    _, ha = tf.split(ha, [-1, 1], axis=1)  # --> delete later
    _, ho = tf.split(ho, [-1, 1], axis=1)  # --> delete later

    first_half_indices_shuffled = Lambda(lambda _x: tf.random_shuffle(tf.range(0, _x)))(half)
    second_half_indices_ordered = K.arange(half, batch_size)
    indices = K.concatenate([first_half_indices_shuffled, second_half_indices_ordered], axis=-1)
    ha_shuffled = Lambda(lambda _x: tf.gather(_x[0], _x[1], axis=0))([ha, indices])

    ad_decisions = AD([ho, ha])
    ad_decisions_shuffled = AD([ho, ha_shuffled])

    ad_target = K.constant(0.5, shape=[batch_size, ])
    # label 1 means mismatch, label 0 means match
    ad_target_shuffled = K.concatenate([K.constant(1, shape=[half, ]), K.constant(0, shape=[half, ])], axis=0)

    # == Instance the two models
    # == Autoencoder
    autoencoder = Model(inputs=ins, outputs=x_recovered)

    rec_loss = mean_squared_error(x_target, x_recovered)
    autoencoder.add_metric(K.mean(rec_loss), name='rec_loss', aggregation='mean')

    rec_loss_pos = mean_squared_error(x_target_pos, x_err_pos)
    autoencoder.add_metric(rec_loss_pos, name='rec_loss_pos', aggregation='mean')

    rec_loss_neg = mean_squared_error(x_target_neg, x_err_neg)
    autoencoder.add_metric(rec_loss_neg, name='rec_loss_neg', aggregation='mean')

    rec_action_only_loss = mean_squared_error(x_rec_a, xo)
    autoencoder.add_metric(rec_action_only_loss, name='rec_A', aggregation='mean')

    # --> make all combinations
    if disc_weight > 0.0:
        autoencoder.add_metric(K.sigmoid(ad_decisions), name='AD_decisions', aggregation='mean')
        ad_loss = binary_crossentropy(y_true=ad_target, y_pred=ad_decisions, from_logits=True) - 0.6931472  # --> !
        autoencoder.add_metric(ad_loss, name='bce_loss', aggregation='mean')
        autoencoder.add_loss(K.mean(rec_loss) + (K.mean(rec_loss_pos) + K.mean(rec_loss_neg)) +
                             disc_weight * K.mean(ad_loss))
    elif gaussian_o:
        kl_loss = self.kl_unit_normal(_mean=mu, _logvar=logvar)
        autoencoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        autoencoder.add_loss(K.mean(rec_loss) + (K.mean(rec_loss_pos) + K.mean(rec_loss_neg)) +
                             kl_weight * K.mean(kl_loss))
    else:
        autoencoder.add_loss(K.mean(rec_loss) + (K.mean(rec_loss_pos) + K.mean(rec_loss_neg)))

    autoencoder.compile(optimizer=Adam(lr=ae_learning_rate))

    # == Discriminator
    discriminator = Model(inputs=ins, outputs=ad_decisions_shuffled)
    disc_ad_loss = binary_crossentropy(y_true=ad_target_shuffled, y_pred=ad_decisions_shuffled, from_logits=True)
    disc_ad_accuracy = binary_accuracy(y_true=ad_target_shuffled, y_pred=K.sigmoid(ad_decisions_shuffled))
    discriminator.add_loss(K.mean(disc_ad_loss))
    discriminator.add_metric(disc_ad_loss, name='disc_ad_loss', aggregation='mean')
    discriminator.add_metric(disc_ad_accuracy, name='disc_ad_acc', aggregation='mean')
    discriminator.compile(optimizer=Adam(lr=disc_learning_rate))

    return autoencoder, discriminator


def kl_unit_normal(_mean, _logvar):
    # KL divergence has a closed form solution for unit gaussian
    # See: https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian
    _kl_loss = - 0.5 * K.sum(1.0 + _logvar - K.square(_mean) - K.exp(_logvar), axis=[-1, -2])
    return _kl_loss
