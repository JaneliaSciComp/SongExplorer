# all imported packages must be in the container
import tensorflow as tf
from tensorflow.keras.layers import *

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

# a list of lists specifying the architecture-specific hyperparameters in the GUI
model_parameters = [
  # [key, title in GUI, "" for textbox or [] for pull-down, default value, width, enable logic, callback, required]
  ["my-simple-textbox",    "h-parameter 1",    "",              "32",   1, [],                  None,     True],
  ["a-bounded-value",      "can't be < 0",     "",              "3",    1, [],                  callback, True],
  ["a-menu",               "choose one",       ["this","that"], "this", 1, [],                  None,     True],
  ["a-conditional-param",  "that's parameter", "",              "8",    1, ["a-menu",["that"]], None,     True],
  ["an-optional-param",    "can be blank",     "",              "0.5",  1, [],                  None,     False],
  ]

# define custom keras layers by sub-classing Layer and wrapping tf functions
# call with MyLayer(arg1, arg2)(previous_layer) as usual
# class MyLayer(tf.keras.layers.Layer):
#     def __init__(self, arg1, arg2, **kwargs):
#         super(MyLayer, self).__init__(**kwargs)
#         self.arg1 = arg1
#         self.arg2 = arg2
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'arg1': self.arg1,
#             'arg2': self.arg2,
#         })
#         return config
#     def call(self, inputs, training=None):  # training is boolean
#         return tf.some_tensorflow_function(inputs, self.arg1, self.arg2)

# a function which returns a keras model
def create_model(model_settings, model_parameters):
    # `model_settings` is a dictionary of additional hyperparameters
    hyperparameter1 = int(model_parameters["my-simple-textbox"])
    nonnegative = int(model_parameters["a-bounded-value"])

    # hidden_layers is used to visualize intermediate clusters in the GUI
    hidden_layers = []

    # 'parallelize' specifies the number of output tics to evaluate
    # simultaneously when classifying.  stride (from e.g. spectrograms)
    # and downsampling (from e.g. conv kernel strides) must be taken into
    # account to get the corresponding number of input tics
    ninput_tics = model_settings["context_tics"] + model_settings["parallelize"] - 1
    input_layer = Input(shape=(ninput_tics, model_settings["audio_nchannels"]))

    x = Conv1D(hyperparameter1, nonnegative)(input_layer);
    hidden_layers.append(x)
    if model_parameters["an-optional-param"]!="":
        x = Dropout(float(model_parameters["an-optional-param"]))(x)
    if model_parameters["a-menu"]=="this":
        x = BatchNormalization()(x)
    elif model_parameters["a-menu"]=="that":
        x = GroupNormalization(float(model_parameters["a-conditional-param"]))(x)

    # add more layers, e.g. x = ReLU(x)
    # append interesting ones to hidden_layers

    # last layer must be convolutional with nlabels as the output size
    output_layer = Conv1D(model_settings['nlabels'], 1)(x)

    print('architecture-plugin.py version = 0.1')
    return tf.keras.Model(inputs=input_layer, outputs=[hidden_layers, output_layer],
                          name='architecture-plugin')
