import os
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_absolute_error
from adr import action_inference_model
from utils.utils import get_data
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def main():

    epochs = 100
    shuffle = True
    bs = 32
    seq_len = 30
    shuffle = True
    dataset_dir = '/media/Data/datasets/bair/softmotion30_44k/'
    save_path = os.path.join(os.path.expanduser('~/'), 'adr/trained_models/bair/')

    frames, _, states, steps, train_iterator = get_data(dataset='bair', mode='train', batch_size=bs,
                                                        shuffle=shuffle, dataset_dir=dataset_dir,
                                                        sequence_length_train=seq_len, initializable=False)

    val_frames, _, val_states, val_steps, val_iterator = get_data(dataset='bair', mode='val', batch_size=bs,
                                                                  shuffle=False, dataset_dir=dataset_dir,
                                                                  sequence_length_test=seq_len)

    history = train_inference_model(frames, states, val_frames, val_states, epochs, steps, val_steps, save_path)


def train_inference_model(frames, states, val_frames, val_states, epochs, steps, val_steps, save_path):

    C = action_inference_model(frames)

    C.compile(optimizer=Adam(),
              loss=mean_absolute_error,
              target_tensors=states)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=epochs)
    ckpt = ModelCheckpoint(filepath=os.path.join(save_path, 'C.h5'), monitor='val_loss',
                           save_best_only=True)

    history = C.fit(epochs=epochs,
                    steps_per_epoch=steps,
                    callbacks=[es, ckpt],
                    validation_data=(val_frames, val_states),
                    validation_steps=val_steps)

    return history


if __name__ == '__main__':
    main()
