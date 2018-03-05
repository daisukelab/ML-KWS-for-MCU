# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against live recorded sound and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None

# For realtime handling
import pyaudio
import queue
import time
import numpy as np
empty_1s_wav = 'empty-mono1s.wav'
WAV_HEADER_BYTES = 44
THRESH = 0.5 # Print results higher than this only


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      if score < THRESH or human_string == '_silence_':
        break
      print('%s (score = %.5f)' % (human_string, score))

    return 0

frame_buffer = queue.Queue(maxsize=30)

def audio_callback(in_data, frame_count, time_info, status):
    #wave = np.fromstring(in_data, dtype=np.int16)
    wave = bytes(in_data)
    #spec = librosa_wave_to_melspectrogram(wave)
    #print(len(wave), frame_buffer.qsize())
    #spec = spec.reshape(6, 40)
    frame_buffer.put(wave, True)
    return (None, pyaudio.paContinue)

def label_realtime(input_device_index, labels, graph, input_name, output_name, how_many_labels):
  """Starts online audio, loads the model and labels, and runs the inference"""
  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  # load labels
  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  # open audio
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  SAMPLING_RATE = 44100 # 16000
  CHUNK = 441 * 5 # 160 * 5

  audio = pyaudio.PyAudio()
  stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLING_RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK,
            start=False,
            stream_callback=audio_callback
        )

  print('Mic is ready to listen, every 1 sec recording will be processed.')
  print('^C to stop program. Here we go.')

  # Prepare fixed wav header, and empty data
  with open(empty_1s_wav, mode='rb') as f:
    tmpl_wav_data = f.read()
    print('Circular buffer size = ', len(tmpl_wav_data))
    wav_data = bytearray(len(tmpl_wav_data))
    wav_data[:WAV_HEADER_BYTES] = tmpl_wav_data[:WAV_HEADER_BYTES]
    wav_data[WAV_HEADER_BYTES:] = [0] * (len(tmpl_wav_data) - WAV_HEADER_BYTES)

  stream.start_stream()
  try:
    while stream.is_active():
      if frame_buffer.qsize() < 1:
        time.sleep(0.005)
        continue
      frame = frame_buffer.get(True)
      # Shift FIFO
      wav_data[WAV_HEADER_BYTES:] = wav_data[WAV_HEADER_BYTES + len(frame):] + frame
      # predict
      run_graph(bytes(wav_data), labels_list, input_name, output_name, how_many_labels)
  except KeyboardInterrupt:
    pass

  # Todo: fixme to come here
  print('Finished.')
  stream.stop_stream()
  stream.close()
  audio.terminate()

def main(_):
  """Entry point for script, converts flags to arguments."""
  label_realtime(FLAGS.input_device_index, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_device_index', type=int, default='0', help='Audio device index to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
