import numpy as np

# a list of lists specifying the doubleclick-specific hyperparameters in the GUI
def doubleclick_parameters(time_units, freq_units, time_scale, freq_scale):
    return [
        ["search", "search ("+time_units+")", "", str(0.010/time_scale), 1, [], None, True],
        ["range",  "range ("+time_units+")",  "", str(0.001/time_scale), 1, [], None, True],
    ]

# a function which returns the start and stop times of the annotation given the location the user double clicked.
def doubleclick_annotation(context_data, context_data_istart, audio_tic_rate, doublclick_parameters, x_tic):
    search_sec = float(doublclick_parameters["search"].value) * time_units
    range_sec = float(doublclick_parameters["range"].value) * time_units

    search_tic = int(search_sec * audio_tic_rate)
    half_range_tic = int(range_sec * audio_tic_rate / 2)

    if search_tic==0:
        imax = 0
    else:
        ileft = max(0, x_tic - context_data_istart - search_tic)
        iright = min(len(context_data[0]), x_tic - context_data_istart + search_tic)
        imax = np.argmax(np.abs(context_data[0][ileft:iright]))
    
    return [context_data_istart + ileft + imax - half_range_tic,
            context_data_istart + ileft + imax + half_range_tic]
