import sys
from os import listdir
from os.path import isfile, join
import os
import pathlib

#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import tensorflow as tf
import time

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
#from IPython import display

np.set_printoptions(threshold=sys.maxsize)

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
time_start = time.perf_counter()

data_dir = pathlib.Path('data/speech_commands_v0.02')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'speech_commands_v0.02.tar.gz',
      origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
      extract=True,
      cache_dir='.', cache_subdir='data/speech_commands_v0.02')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))



train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)


def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return parts[-2]


def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)


def get_spectrogram(waveform):
  sample_rate = 16000.0

  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=1024, frame_step=512)

  spectrogram = tf.abs(spectrogram)
  
  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = spectrogram.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,  upper_edge_hertz)
  mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
  mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  spectrogram = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :13]

  return spectrogram



def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)




def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds



num_labels = len(commands)

time_preprocess = time.perf_counter()

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
model.summary()


time_end=time.perf_counter()





x=-1
for command in commands:
  y = 0
  x=x+1
  mypath = str(data_dir) + "/" + str(command) + "/"
  from os import listdir
  from os.path import isfile, join
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  for file in onlyfiles:
    sample_ds = preprocess_dataset([mypath + str(file)])

    for spectrogram, label in sample_ds.batch(1):
      prediction = model(spectrogram)
      confidence = tf.nn.softmax(prediction[0])
      if confidence[x] < .1:
        os.remove(mypath + str(file))
        print(f'Remove {mypath + str(file) + "           "}')
        print(commands)
        print(confidence)
        y = y +1
  print("Removed ", y)

