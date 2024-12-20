import numpy as np

def augmentation_parameters():
    return [
          ["volume",  "volume",  "",            "1,1", 1, [], None, True],
          ["noise",   "noise",   "",            "0,0", 1, [], None, True],
          ["dc",      "dc",      "",            "0,0", 1, [], None, True],
          ["reverse", "reverse", ["no", "yes"], "no",  1, [], None, True],
          ["invert",  "invert",  ["no", "yes"], "no",  1, [], None, True],
    ]

def augment(audio_slice, parameters):
    volume_range = [float(x) for x in parameters['volume'].split(',')]
    noise_range = [float(x) for x in parameters['noise'].split(',')]
    dc_range = [float(x) for x in parameters['dc'].split(',')]
    reverse_bool = parameters['reverse'] == 'yes'
    invert_bool = parameters['invert'] == 'yes'
    nsounds = audio_slice.shape[0]
    audio_nchannels = audio_slice.shape[2]
    if volume_range != [1,1]:
        volume_ranges = np.random.uniform(*volume_range, (nsounds,1,audio_nchannels))
        audio_slice *= volume_ranges
    if noise_range != [0,0]:
        noise_ranges = np.random.uniform(*noise_range, (nsounds,1,audio_nchannels))
        noises = np.random.normal(0, noise_ranges, audio_slice.shape)
        audio_slice += noises
    if dc_range != [0,0]:
        dc_ranges = np.random.uniform(*dc_range, (nsounds,1,audio_nchannels))
        audio_slice += dc_ranges
    if reverse_bool:
        ireverse = np.random.choice([False,True], nsounds)
        audio_slice[ireverse] = np.flip(audio_slice[ireverse], axis=1)
    if invert_bool:
        iinvert = np.random.choice([-1,1], (nsounds,1,1))
        audio_slice *= iinvert

    return audio_slice
