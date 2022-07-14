import numpy as np

# a list of lists specifying the doubleclick-specific hyperparameters in the GUI
doubleclick_parameters = [
    ["search", "search (msec)", "", "10", 1, [], None, True],
    ["width",  "width (msec)",  "", "1",  1, [], None, True],
    ]

# a function which returns the start and stop times of the annotation given the location the user double clicked.
def doubleclick_annotation(context_data, context_data_istart, audio_tic_rate, doublclick_parameters, x_tic):
    search_ms = float(doublclick_parameters["search"].value)
    width_ms = float(doublclick_parameters["width"].value)

    search_tic = int(search_ms/1000*audio_tic_rate)
    half_width_tic = int(width_ms/1000*audio_tic_rate/2)

    if search_tic==0:
        imax = 0
    else:
        ileft = max(0, x_tic - context_data_istart - search_tic)
        iright = min(len(context_data[0]), x_tic - context_data_istart + search_tic)
        imax = np.argmax(np.abs(context_data[0][ileft:iright]))
    
    return [context_data_istart + ileft + imax - half_width_tic,
            context_data_istart + ileft + imax + half_width_tic]
