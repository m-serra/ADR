import os
import tensorflow as tf
from utils.utils import get_data
from adr import adr_ao
from models.lstm import load_lstm
from models.encoder_decoder import load_decoder
from models.encoder_decoder import load_recurrent_encoder
from models.action_net import load_action_net
from models.action_net import load_recurrent_action_net
import tensorflow.python.keras.backend as K


def main():

    bs = 32
    seq_len = 12
    mode = 'train'
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'
    ckpt_dir = os.path.join('/home/mandre/adr/trained_models/bair')

    frames, actions, states, steps, _ = get_data(dataset='bair', mode=mode, batch_size=bs, shuffle=False,
                                                 dataset_dir=dataset_dir, sequence_length_train=seq_len,
                                                 sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='1')
    config = tf.ConfigProto(gpu_options=gpu_options)

    evaluate_autoencoder_A(frames,
                           actions,
                           states,
                           ckpt_dir=ckpt_dir,
                           context_frames=2,
                           gaussian=True,
                           hc_dim=128,
                           ha_dim=16,
                           z_dim=10,
                           units=256,
                           lstm_units=256,
                           lstm_layers=1,
                           steps=steps,
                           ec_filename='Ec_a_test.h5',
                           d_filename='D_a_test.h5',
                           a_filename='A_a_test.h5',
                           l_filename='L_a_test.h5',
                           set_states=False,
                           config=config,
                           evaluate=True,
                           predict=False,
                           random_context_frames=False)


def evaluate_autoencoder_A(frames, actions, states=None, ckpt_dir=None, context_frames=5, gaussian=False,
                           hc_dim=128, ha_dim=16, z_dim=10, units=256, config=None, steps=None,
                           lstm_units=256, lstm_layers=1, ec_filename='Ec.h5', d_filename='Da.h5',
                           a_filename='A.h5', l_filename='L.h5', set_states=False, evaluate=False, predict=True,
                           random_context_frames=False):

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]
    a_dim = int(actions.shape[-1]) if actions is not None else 0
    s_dim = int(states.shape[-1]) if states is not None else 0
    if set_states:
        seq_len = 24
    z_dim = 0 if gaussian is False else z_dim

    sess = tf.Session(config=config)
    K.set_session(sess)

    Ec = load_recurrent_encoder([bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir, filename=ec_filename,
                                trainable=False, load_model_state=False)
    D = load_decoder(h_dim=hc_dim+ha_dim+z_dim, model_name='D', ckpt_dir=ckpt_dir, filename=d_filename,
                     output_activation='sigmoid', trainable=False, load_model_state=False)
    if not gaussian:
        L = None
        A = load_recurrent_action_net([bs, seq_len-context_frames, a_dim+s_dim], units, ha_dim, ckpt_dir,
                                      filename=a_filename, trainable=False, load_model_state=False)
    else:
        A = load_action_net(batch_shape=[bs, seq_len, a_dim+s_dim], units=units, ha_dim=ha_dim,
                            ckpt_dir=ckpt_dir, filename=a_filename, trainable=False, load_model_state=False)
        L = load_lstm(batch_shape=[bs, seq_len, hc_dim+ha_dim], output_dim=z_dim, lstm_units=lstm_units,
                      n_layers=lstm_layers, ckpt_dir=ckpt_dir, filename=l_filename, lstm_type='gaussian',
                      load_model_state=False)
    if set_states:
        states = K.variable([[[0.525, -0.175, 0.2], [0.5666, -0.1333, 0.2],   [0.60833, -0.09166, 0.2],
                            [0.65, -0.05, 0.2],   [0.69166, -0.00833, 0.2], [0.7333, 0.03333, 0.2],
                            [0.775, 0.075, 0.2],  [0.7333, 0.11666, 0.2],   [0.69166, 0.158333, 0.2],
                            [0.65, 0.2, 0.2],     [0.60833, 0.158333, 0.2], [0.5666, 0.11666, 0.2],
                            [0.525, 0.075, 0.2],  [0.5666, 0.03333, 0.2],   [0.60833, -0.00833, 0.2],
                            [0.65, -0.05, 0.2],   [0.69166, -0.09166, 0.2], [0.7333, -0.1333, 0.2],
                            [0.775, -0.175, 0.2], [0.7333, -0.21666, 0.2],  [0.69166, -0.25833, 0.2],
                            [0.65, -0.3, 0.2],    [0.60833, -0.25833, 0.2], [0.5666, -0.21666, 0.2]]]*bs,
                            dtype='float32').initialized_value()

    model = adr_ao(frames,
                   actions,
                   states,
                   context_frames,
                   Ec=Ec,
                   A=A,
                   D=D,
                   L=L,
                   learning_rate=0.0,
                   gaussian=gaussian,
                   kl_weight=0.0,
                   lstm_units=lstm_units,
                   lstm_layers=lstm_layers,
                   training=False,
                   random_context_frames=random_context_frames)

    if evaluate:
        model.evaluate(x=None, steps=steps)

    if predict:
        x, imgs = model.predict(x=None, steps=steps)
        return x, imgs
    return


if __name__ == '__main__':
    main()
