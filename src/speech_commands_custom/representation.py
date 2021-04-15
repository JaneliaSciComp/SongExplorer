import tensorflow as tf

# Allow the sound's volume to be adjusted.
def scale_foreground(foreground_data, foreground_volume, model_settings):
    return tf.multiply(foreground_data, foreground_volume)

# Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.

def compute_spectrograms(foreground_data, foreground_volume, model_settings):
    scaled_foreground = scale_foreground(foreground_data, foreground_volume, model_settings)

    # tf.square is needed to get the same output as with tf.contrib.signal, but
    # should be omitted when input to tf.signal.mfcc
    # given channel X time, returns channel X time X freq
    return tf.math.abs(tf.signal.stft(scaled_foreground,
                                      model_settings['window_tics'],
                                      model_settings['stride_tics']))

#  output is similar but not the same as with tf.contrib.signal
def compute_mfccs(foreground_data, foreground_volume, model_settings):
    spectrograms = compute_spectrograms(foreground_data, foreground_volume, model_settings)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = model_settings['window_tics']//2+1
    lower_edge_hertz = 0.0
    upper_edge_hertz = model_settings['audio_tic_rate']//2
    num_mel_bins = model_settings['filterbank_nchannels']
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, model_settings['audio_tic_rate'],
      lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first few
    return tf.signal.mfccs_from_log_mel_spectrograms(
      log_mel_spectrograms)[..., :model_settings['dct_ncoefficients']]
