import math

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def create_model(fingerprint_input, model_settings, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  if model_settings['representation']=='waveform':
    input_frequency_size = 1
    input_time_size = model_settings['desired_samples']
  elif model_settings['representation']=='spectrogram':
    input_frequency_size = model_settings['window_size_samples']//2+1
    input_time_size = model_settings['spectrogram_length']
  elif model_settings['representation']=='mel-cepstrum':
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
  input_channel_size = model_settings['channel_count']
  filter_counts = model_settings['filter_counts']
  filter_sizes = model_settings['filter_sizes']
  final_filter_len = model_settings['final_filter_len']
  filter_sizes = model_settings['filter_sizes']
  batch_size = model_settings['batch_size']
  nwindows = model_settings['nwindows']
  dilate_after_layer = model_settings['dilate_after_layer']
  stride_after_layer = model_settings['stride_after_layer']
  residual = True if model_settings['connection_type']=='residual' else False

  fingerprint_4d = tf.reshape(fingerprint_input,
                              [batch_size, input_time_size,
                               input_frequency_size, input_channel_size])
  fingerprint_4d_shape = fingerprint_4d.get_shape().as_list()

  iconv=0
  hidden_layers = []

  inarg = fingerprint_4d
  hidden_layers.append(inarg)
  inarg_shape = inarg.get_shape().as_list()
  output_time_size1 = inarg_shape[1]-nwindows+1
  filter_count_prev = input_channel_size

  while inarg_shape[2]>=filter_sizes[0]:
    if residual and iconv%2==0:
      bypass = inarg
    weights = tf.Variable( tf.truncated_normal(
              [filter_sizes[0], filter_sizes[0], filter_count_prev, filter_counts[0]],
              stddev=0.01))
    filter_count_prev = filter_counts[0]
    bias = tf.Variable(tf.zeros([filter_counts[0]]))
    dilation = [1,2**max(0,iconv-dilate_after_layer+1),1,1]
    strides = [1,1+(iconv>=stride_after_layer),1,1]
    conv = tf.nn.conv2d(inarg, weights, strides, 'VALID', dilations=dilation) + bias
    bypassadded=False
    if residual and iconv%2!=0:
      bypassadded=True
      bypass_shape = bypass.get_shape().as_list()
      output_shape = conv.get_shape().as_list()
      woffset = (bypass_shape[1] - output_shape[1]) // 2
      hoffset = (bypass_shape[2] - output_shape[2]) // 2
      conv += bypass[:, hoffset:hoffset+output_shape[1], woffset:woffset+output_shape[2], :]
    hidden_layers.append(conv)
    relu = tf.nn.relu(conv)
    if is_training:
      dropout = tf.nn.dropout(relu, dropout_prob)
    else:
      dropout = relu
    tf.logging.info('conv layer %d: in_shape = %s, conv_shape = %s, strides = %s, dilation = %s, bypass = %s' %
          (iconv, inarg.get_shape(), weights.get_shape(), str(strides), str(dilation), str(bypassadded)))
    inarg = dropout
    inarg_shape = inarg.get_shape().as_list()
    output_time_size1 = math.ceil((output_time_size1 - filter_sizes[0] + 1) / strides[1])
    iconv += 1

  while inarg_shape[2]>=filter_sizes[1]:
    if residual and iconv%2==0:
      bypass = inarg
    weights = tf.Variable( tf.truncated_normal(
              [filter_sizes[1], filter_sizes[1], filter_count_prev, filter_counts[1]],
              stddev=0.01))
    filter_count_prev = filter_counts[1]
    bias = tf.Variable(tf.zeros([filter_counts[1]]))
    dilation = [1,2**max(0,iconv-dilate_after_layer+1),1,1]
    strides = [1,1+(iconv>=stride_after_layer),1,1]
    conv = tf.nn.conv2d(inarg, weights, strides, 'VALID', dilations=dilation) + bias
    bypassadded=False
    if residual and iconv%2!=0:
      bypassadded=True
      bypass_shape = bypass.get_shape().as_list()
      output_shape = conv.get_shape().as_list()
      woffset = (bypass_shape[1] - output_shape[1]) // 2
      hoffset = (bypass_shape[2] - output_shape[2]) // 2
      conv += bypass[:, hoffset:hoffset+output_shape[1], woffset:woffset+output_shape[2], :]
    hidden_layers.append(conv)
    relu = tf.nn.relu(conv)
    if is_training:
      dropout = tf.nn.dropout(relu, dropout_prob)
    else:
      dropout = relu
    tf.logging.info('conv layer %d: in_shape = %s, conv_shape = %s, strides = %s, dilation = %s, bypass = %s' %
          (iconv, inarg.get_shape(), weights.get_shape(), str(strides), str(dilation), str(bypassadded)))
    inarg = dropout
    inarg_shape = inarg.get_shape().as_list()
    output_time_size1 = math.ceil((output_time_size1 - filter_sizes[1] + 1) / strides[1])
    iconv += 1

  assert inarg_shape[2]==1

  #inarg = tf.squeeze(inarg,[2])
  output_time_size = inarg.get_shape().as_list()[1]
  while output_time_size1>final_filter_len:
    if residual and iconv%2==0:
      bypass = inarg
    weights = tf.Variable( tf.truncated_normal(
              [filter_sizes[2], 1, filter_count_prev, filter_counts[2]],
              stddev=0.01))
    filter_count_prev = filter_counts[2]
    bias = tf.Variable(tf.zeros([filter_counts[2]]))
    dilation = [1,2**max(0,iconv-dilate_after_layer+1),1,1]
    strides = [1,1+(iconv>=stride_after_layer),1,1]
    conv = tf.nn.conv2d(inarg, weights, strides, 'VALID', dilations=dilation) + bias
    output_time_size = conv.get_shape().as_list()[1]
    bypassadded=False
    if residual and iconv%2!=0:
      bypassadded=True
      offset = (bypass.get_shape().as_list()[1] - output_time_size) // 2
      conv += bypass[:, offset:offset+output_time_size, :, :]
    hidden_layers.append(conv)
    relu = tf.nn.relu(conv)
    if is_training:
      dropout = tf.nn.dropout(relu, dropout_prob)
    else:
      dropout = relu
    tf.logging.info('conv layer %d: in_shape = %s, conv_shape = %s, strides = %s, dilation = %s, bypass = %s' %
          (iconv, inarg.get_shape(), weights.get_shape(), str(strides), str(dilation), str(bypassadded)))
    inarg = dropout
    output_time_size1 = math.ceil((output_time_size1 - filter_sizes[2] + 1) / strides[1])
    iconv += 1

  #assert output_time_size==(nwindows+final_filter_len)

  inarg = tf.squeeze(inarg,[2])
  label_count = model_settings['label_count']
  weights = tf.Variable( tf.truncated_normal(
            [output_time_size1, filter_count_prev, label_count],
            stddev=0.01))
  bias = tf.Variable(tf.zeros([label_count]))
  strides = 1+(iconv>=stride_after_layer)
  final = tf.nn.conv1d(inarg, weights, strides, 'VALID') + bias

  tf.logging.info('final layer: in_shape = %s, conv_shape = %s, strides = %s' %
        (inarg.get_shape(), weights.get_shape(), str(strides)))
  if is_training:
    return hidden_layers, tf.squeeze(final), dropout_prob
  else:
    return hidden_layers, tf.squeeze(final)
