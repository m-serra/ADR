import os
import numpy as np
import tensorflow as tf
from utils.utils import get_data
from models.lstm import load_lstm
from models.encoder_decoder import load_encoder
from models.encoder_decoder import load_decoder
from models.encoder_decoder import load_recurrent_encoder
from models.action_net import load_action_net
from models.action_net import load_recurrent_action_net
import tensorflow.python.keras.backend as K
from adr import adr_vp_teacher_forcing, adr_vp_feedback, adr_vp_feedback_frames
from utils.utils import save_gifs


def main():

    bs = 32
    use_seq_len = 12
    seq_len = 30
    mode = 'train'  # --> !!!
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'
    ckpt_dir = os.path.join('/home/mandre/adr/trained_models/bair/random_window')

    frames, actions, states, steps, _ = get_data(dataset='bair', mode=mode, batch_size=bs, shuffle=False,
                                                 dataset_dir=dataset_dir, sequence_length_train=seq_len,
                                                 sequence_length_test=seq_len)

    gpu_options = tf.GPUOptions(visible_device_list='0')
    config = tf.ConfigProto(gpu_options=gpu_options)

    evaluate_adr_vp(frames,
                    actions,
                    states,
                    ckpt_dir=ckpt_dir,
                    context_frames=2,
                    use_seq_len=use_seq_len,
                    gaussian_a=True,
                    hc_dim=128,
                    ha_dim=16,
                    ho_dim=32,
                    za_dim=10,
                    lstm_units=256,
                    lstm_a_units=256,
                    lstm_layers=2,
                    lstm_a_layers=1,
                    action_net_units=256,
                    steps=steps,
                    eo_load_name='Eo_t00103_v0011_tf10tv2.h5',
                    do_load_name='D_o_t00103_v0011_tf10tv2.h5',
                    l_load_name='L_t00103_v0011_tf10tv2.h5',
                    ec_load_name='Ec_a_t00243_v0023_tf10tv2.h5',
                    a_load_name='A_a_t00243_v0023_tf10tv2.h5',
                    da_load_name='D_a_t00243_v0023_tf10tv2.h5',
                    la_load_name='L_a_t00243_v0023_tf10tv2.h5',
                    config=config,
                    evaluate=True,
                    predict=False,
                    feedback_predictions=True,
                    random_window=True)


def evaluate_adr_vp(frames, actions, states, context_frames, ckpt_dir=None, hc_dim=128, ha_dim=16, ho_dim=32, za_dim=10,
                    gaussian_a=True, use_seq_len=30, lstm_layers=2, lstm_units=256, lstm_a_layers=1, lstm_a_units=256,
                    action_net_units=256, ec_load_name='Ec.h5', a_load_name='A.h5', eo_load_name='Eo.h5',
                    da_load_name='Da.h5', do_load_name='Do.h5', la_load_name='La.h5', l_load_name='L.h5', config=None,
                    evaluate=False, predict=True, steps=1, feedback_predictions=True, random_window=False):

    bs, seq_len, w, h, c = [int(s) for s in frames.shape]
    a_dim = int(actions.shape[-1]) if actions is not None else 0
    s_dim = int(states.shape[-1]) if states is not None else 0
    za_dim = 0 if gaussian_a is False else za_dim

    sess = tf.Session(config=config)
    K.set_session(sess)

    # == Instance and load the models
    Ec = load_recurrent_encoder([bs, context_frames, w, h, c], h_dim=hc_dim, ckpt_dir=ckpt_dir, name='Ec',
                                filename=ec_load_name, trainable=False, load_model_state=True)

    Da = load_decoder(batch_shape=[bs, seq_len, hc_dim+ha_dim+za_dim], model_name='Da', ckpt_dir=ckpt_dir,
                      output_channels=3, filename=da_load_name, output_activation='sigmoid', trainable=False,
                      load_model_state=True)

    Do = load_decoder(batch_shape=[bs, 1, hc_dim + ha_dim + ho_dim], model_name='Do', ckpt_dir=ckpt_dir,
                      output_channels=6, filename=do_load_name, output_activation='sigmoid', trainable=False,
                      load_model_state=True)

    Eo = load_encoder(batch_shape=[bs, 1, w, h, c * 2], h_dim=ho_dim, model_name='Eo', ckpt_dir=ckpt_dir,
                      filename=eo_load_name, trainable=False, load_model_state=True)

    L = load_lstm(batch_shape=[bs, 1, hc_dim+ha_dim*2+ho_dim], h_dim=ho_dim, n_layers=lstm_layers, name='L',
                  lstm_units=lstm_units, ckpt_dir=ckpt_dir, filename=l_load_name, lstm_type='simple',
                  trainable=False, load_model_state=True)

    if gaussian_a:
        A = load_action_net(batch_shape=[bs, seq_len, a_dim + s_dim], units=action_net_units, h_dim=ha_dim, name='A',
                            ckpt_dir=ckpt_dir, filename=a_load_name, trainable=False, load_model_state=True)
        La = load_lstm(batch_shape=[bs, seq_len, hc_dim + ha_dim], h_dim=za_dim, lstm_units=lstm_a_units,
                       n_layers=lstm_a_layers, ckpt_dir=ckpt_dir, filename=la_load_name, lstm_type='gaussian',
                       name='La', trainable=False, load_model_state=True)
    else:
        A = load_recurrent_action_net([bs, seq_len, a_dim + s_dim], action_net_units, ha_dim, ckpt_dir=ckpt_dir,
                                      filename=a_load_name, trainable=False, load_model_state=False)

    if feedback_predictions:
        model = adr_vp_feedback_frames(frames, actions, states, context_frames, Ec=Ec, Eo=Eo, A=A, Do=Do, Da=Da, L=L, La=La,
                                gaussian_a=gaussian_a, use_seq_len=use_seq_len, lstm_a_units=lstm_a_units,
                                lstm_a_layers=lstm_a_layers, lstm_units=lstm_units, lstm_layers=lstm_layers,
                                learning_rate=0.0, random_window=random_window)
    else:
        model = adr_vp_teacher_forcing(frames, actions, states, context_frames, Ec=Ec, Eo=Eo, A=A, Do=Do, Da=Da, L=L,
                                       La=La, gaussian_a=gaussian_a, use_seq_len=use_seq_len, lstm_a_units=lstm_a_units,
                                       lstm_a_layers=lstm_a_layers, lstm_units=lstm_units, lstm_layers=lstm_layers,
                                       learning_rate=0.0, random_window=random_window)
    if evaluate:
        model.evaluate(x=None, steps=steps)

    if predict:
        ho_pred, x_curr, x, x_a, imgs = model.predict(x=None, steps=steps)
        save_gifs(sequence=np.clip(x[:bs], a_min=0.0, a_max=1.0), name='pred', save_dir=os.path.join('/home/mandre/adr/gifs'))
        save_gifs(sequence=np.clip(x_a[:bs], a_min=0.0, a_max=1.0), name='pred_a', save_dir=os.path.join('/home/mandre/adr/gifs'))
        save_gifs(sequence=imgs[:bs], name='gt', save_dir=os.path.join('/home/mandre/adr/gifs'))

    return None, None


if __name__ == '__main__':
    main()
