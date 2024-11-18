#Original script: https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_params.py
#Licensed under Apache License 2.0

# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Global parameters for the VGGish model.

See vggish_slim.py for more information.
"""

# Architectural constants.
NUM_FRAMES = 96 # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 8000 #originally set as 16000
STFT_WINDOW_LENGTH_SECONDS = 0.05 #originally set as 0.25
STFT_HOP_LENGTH_SECONDS = 0.01 
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 2000 #originally set as 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96 # each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96 # with zero overlap

#originally contained a few more lines that are not relevant to the present repository
