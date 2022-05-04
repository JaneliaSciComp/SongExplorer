# use bokehlog.info() to print debugging messages
import logging 
bokehlog = logging.getLogger("songexplorer") 

# optional callbacks can be used to validate user input
def _callback(p,M,V,C):
    C.time.sleep(0.5)
    V.model_parameters[p].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def callback(n,M,V,C):
    # M, V, C are the model, view, and controller in src/gui
    # access the hyperparameters below with the V.model_parameters dictionary
    # the value is stored in .value, and the appearance can be controlled with .css_classes
    if int(V.model_parameters['a-bounded-value'].value) < 0:
        #bokehlog.info("a-bounded-value = "+str(V.model_parameters['a-bounded-value'].value))  # uncomment to debug
        V.model_parameters['a-bounded-value'].css_classes = ['changed']
        V.model_parameters['a-bounded-value'].value = "0"
        if V.bokeh_document:  # if interactive
            V.bokeh_document.add_next_tick_callback(lambda: _callback('a-bounded-value',M,V,C))
        else:  # if scripted
            _callback('a-bounded-value',M,V,C)

# a list of lists specifying the doubleclick-specific hyperparameters in the GUI.
doubleclick_parameters = [
  # [key in `detect_parameters`, title in GUI, "" for textbox or [] for pull-down, default value, enable logic, callback, required]
  ["my-simple-textbox",    "h-parameter 1",    "",              "32",   [],                  None,     True],
  ["a-bounded-value",      "can't be < 0",     "",              "3",    [],                  callback, True],
  ["a-menu",               "choose one",       ["this","that"], "this", [],                  None,     True],
  ["a-conditional-param",  "that's parameter", "",              "8",    ["a-menu",["that"]], None,     True],
  ["an-optional-param",    "can be blank",     "",              "0.5",  [],                  None,     False],
  ]

# a function which returns the start and stop times of the annotation given the location the user double clicked.
# `context_data` is a list of the waveforms displayed in the context window for each channel.
# `context_data_istart` is the index into the WAV file of the first sample in context_data.
# `doubleclick_parameters` is a dictionary of the doubleclick_parameters list above.
def doubleclick_annotation(context_data, context_data_istart, audio_tic_rate, doublclick_parameters, x_tic):
    hyperparameter1 = int(detect_parameters["my-simple-textbox"])
    return [x_tic, x_tic]
