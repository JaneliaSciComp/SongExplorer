# a list of lists specifying the doubleclick-specific hyperparameters in the GUI.
def doubleclick_parameters(time_units, freq_units, time_scale, freq_scale):
    return []

# a function which returns the start and stop times of the annotation given the location the user double clicked.
def doubleclick_annotation(context_data, context_data_istart, audio_tic_rate, doublclick_parameters, x_tic):
    return [x_tic, x_tic]
