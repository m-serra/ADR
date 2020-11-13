import os
import sys
import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


class BaseDataReader(object):

    def __init__(self,
                 batch_size,
                 sequence_length_train,
                 sequence_length_test,
                 shuffle=False,
                 shuffle_files=True,
                 dataset_repeat=None,
                 train_dir_name='train',
                 test_dir_name='test',
                 batch_repeat=1,
                 block_length=1,
                 initializable=False):
        """
        Dataset class for the BAIR and Google Push datasets.

        inputs
        ------
        - batch_size: (int) size of the mini batches the iterator will provide
        - sequence_length_train: (int) number of timesteps to use for training and validation
        - sequence_length_test: (int) number of timesteps to use for test
        - shuffle: (boolean) whether to shuffle the train/val filenames and tfrecord samples. Defaults
                        to FLAGS.shuffle.
        - dataset_repeat: (int) number of times the dataset can be iterated. If None indefinite iteration is
                                allowed.
        """
        self.dataset_name = None
        self.COLOR_CHAN = None
        self.IMG_WIDTH = None
        self.IMG_HEIGHT = None
        self.STATE_DIM = None
        self.ACTION_DIM = None
        self.ORIGINAL_WIDTH = None
        self.ORIGINAL_HEIGHT = None
        self.filenames = None
        self.data_dir = None
        self.train_val_split = None
        self.train_filenames = None
        self.val_filenames = None
        self.test_filenames = None
        self.shuffle = shuffle
        self.shuffle_files = shuffle_files
        self.n_threads = multiprocessing.cpu_count()
        self.train_dir_name = train_dir_name
        self.test_dir_name = test_dir_name

        self.dataset_repeat = dataset_repeat
        self.batch_size = batch_size
        self.sequence_length_train = sequence_length_train
        self.sequence_length_test = sequence_length_test
        self.sequence_length_to_use = None
        self.batch_repeat = batch_repeat
        self.block_length = block_length
        self.initializable = initializable

    def set_filenames(self):

        train_val_dir = os.path.join(self.data_dir, self.train_dir_name)
        test_dir = os.path.join(self.data_dir, self.test_dir_name)

        try:
            train_val_filenames = gfile.Glob(os.path.join(train_val_dir, '*'))
            idx_train_val_split = int(np.floor(self.train_val_split * len(train_val_filenames)))
            train_filenames = train_val_filenames[:idx_train_val_split]
            val_filenames = train_val_filenames[idx_train_val_split:]

        except:
            train_filenames = None
            val_filenames = None
            print("No train/val data files found.")

        try:
            test_filenames = gfile.Glob(os.path.join(test_dir, '*'))
        except:
            test_filenames = None
            print("No test data files found.")

        return train_filenames, val_filenames, test_filenames

    def build_tf_iterator(self, mode='train'):
        """Create input tfrecord iterator
        Args:
        -----
            mode: 'train', 'val' or 'test'

        Returns:
        --------
            An iterator that returns images, actions, states, distances, angles
        """
        assert mode in ['train', 'val', 'test'], 'Mode must be one of "train", "val" or "test"'
        filenames = None

        if mode == 'train' and self.train_filenames is not None:
            filenames = self.train_filenames
            self.sequence_length_to_use = self.sequence_length_train
        elif mode == 'val' and self.val_filenames is not None:
            filenames = self.val_filenames
            self.sequence_length_to_use = self.sequence_length_train
        elif self.test_filenames:
            filenames = self.test_filenames
            self.sequence_length_to_use = self.sequence_length_test

        if filenames is None:
            print('No files found')
            sys.exit(0)

        filename_queue = tf.data.Dataset.from_tensor_slices(filenames)

        if self.shuffle and self.shuffle_files and mode is not 'test':
            filename_queue = filename_queue.shuffle(buffer_size=len(filenames))

        dataset = tf.data.TFRecordDataset(filename_queue)

        if self.shuffle and mode is not 'test':
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=2048, count=self.dataset_repeat))
        else:
            dataset = dataset.repeat(count=self.dataset_repeat)  # to allow multiple epochs with one_shot_iterator

        # ==== Trying to avoid corrupted record error
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        dataset = dataset.map(lambda x: self._parse_sequences(x))

        # This is useful for training adversarial generator/discriminator as the iterator will output a batch for
        # the generator on a call and then the same batch for the discriminator on a second call
        if self.batch_repeat > 1:
            dataset = dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(self.batch_repeat),
                                         cycle_length=self.batch_size, block_length=self.block_length)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        if self.initializable:
            iterator = dataset.make_initializable_iterator()
        else:
            iterator = dataset.make_one_shot_iterator()

        return iterator

    def _parse_sequences(self, serialized_example):
        # return {'images': image_seq,
        #         'actions': zeros_action,
        #         'states': zeros_state,
        #         'action_targets': zeros_targets}
        raise NotImplementedError

    def get_seq(self):
        raise NotImplementedError

    @staticmethod
    # in the future replace state by actions for generality
    def save_tfrecord_example(writer, example_id, gen_images, gt_state, save_dir):
        raise NotImplementedError

    def num_examples_per_epoch(self, mode):
        raise NotImplementedError
