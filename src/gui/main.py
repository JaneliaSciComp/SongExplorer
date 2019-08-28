# interactively browse clusters and annotate sounds
 
# visualize.py --args <sample-rate> <snippets-ms> <nx-snippets> <ny-snippets> <gui-width-pix> <context-width-ms> <context-offset-ms>
# http://<hostname>:<port>/visualize

# e.g.
# bokeh serve --allow-websocket-origin=`hostname`:5006 --allow-websocket-origin=localhost:5006 --port 5006 --show /opt/users/deepsong/src/visualize.py --args 5000 40 10 10 8 1200 400 0

import os
from bokeh.plotting import curdoc
from bokeh.io import show, output_notebook
from bokeh.layouts import column, row, widgetbox, Spacer
import threading
import logging 
from sys import argv

log = logging.getLogger("deepsong") 
#log.info(...) 

_, configuration_inarg, Fs, snippets_ms, nx, ny, nlabels, gui_width_pix, context_width_ms, context_offset_ms = argv

import model as M
import view as V
import controller as C

M.init(configuration_inarg, Fs, snippets_ms, nx, ny, nlabels, gui_width_pix, context_width_ms, context_offset_ms)
V.init()

curdoc().add_periodic_callback(M.save_annotations, 5000)

tsne_buttons = widgetbox(V.which_layer, V.which_species, V.which_word, V.which_nohyphen, V.which_kind, width=M.gui_width_pix)
panzoom_buttons = row(column(Spacer(height=41,width=40), widgetbox(V.panleft, width=50)),
                      widgetbox(V.zoomin,V.reset,V.zoomout, width=50),
                      column(Spacer(height=41,width=40), widgetbox(V.panright, width=50)))
label_widgets = row(column(V.label_count_widgets[:M.nlabels//2]),
                    column(V.label_text_widgets[:M.nlabels//2], width=200),
                    Spacer(width=40),
                    column(V.label_count_widgets[M.nlabels//2:]),
                    column(V.label_text_widgets[M.nlabels//2:], width=200))
main_content = row(column(
                      tsne_buttons,
                      row(V.p_tsne, V.p_snippets),
                      row(widgetbox(V.radius_size, width=100),
                          Spacer(width=(M.gui_width_pix-100-300-170)//2),
                          widgetbox(V.save_indicator,width=50, align='center'),
                          widgetbox(V.undo, align='center'),
                          widgetbox(V.redo, align='center'),
                          Spacer(width=(M.gui_width_pix-100-300-170)//2),
                          widgetbox(V.zoom_context, width=150),
                          widgetbox(V.zoom_offset, width=150)),
                      V.p_context,
                      Spacer(height=10),
                      row(panzoom_buttons,
                          Spacer(width=40),
                          label_widgets,
                          Spacer(width=40),
                          V.wordcounts, width=M.gui_width_pix)),
                  column(
                      row(row(V.labelsounds, V.makepredictions, V.fixfalsepositives, V.fixfalsenegatives, V.leaveoneout, V.tunehyperparameters, V.findnovellabels, V.examineerrors, width=M.gui_width_pix-60), row(V.doit,width=60)),
                      row(V.detect, V.misses, V.train, V.generalize, V.xvalidate, V.hidden, V.cluster, V.visualize, V.accuracy, V.freeze, V.classify, V.ethogram, V.compare, V.dense, width=M.gui_width_pix),
                      Spacer(height=20),
                      row(V.configuration, widgetbox(V.configuration_file, width=M.gui_width_pix-120)),
                      row(V.logs, widgetbox(V.logs_folder, width=M.gui_width_pix-120)),
                      row(V.model, widgetbox(V.model_file, width=M.gui_width_pix-120)),
                      row(V.wavtfcsvfiles, widgetbox(V.wavtfcsvfiles_string, width=M.gui_width_pix-120)),
                      row(V.groundtruth, widgetbox(V.groundtruth_folder, width=M.gui_width_pix-120)),
                      row(V.validationfiles, widgetbox(V.validationfiles_string, width=M.gui_width_pix-120)),
                      row(V.testfiles, widgetbox(V.testfiles_string, width=M.gui_width_pix-120)),
                      row(V.wantedwords, widgetbox(V.wantedwords_string, width=M.gui_width_pix-120)),
                      row(V.labeltypes, widgetbox(V.labeltypes_string, width=M.gui_width_pix-120)),
                      row(widgetbox(V.time_sigma_string, width=100),
                          widgetbox(V.time_smooth_ms_string, width=100),
                          widgetbox(V.frequency_n_ms_string, width=100),
                          widgetbox(V.frequency_nw_string, width=100),
                          widgetbox(V.frequency_p_string, width=100),
                          widgetbox(V.frequency_smooth_ms_string, width=100),
                          widgetbox(V.nsteps_string, width=100),
                          widgetbox(V.save_and_validate_period_string, width=100),
                          widgetbox(V.validate_percentage_string, width=100),
                          widgetbox(V.mini_batch_string, width=100),
                          widgetbox(V.kfold_string, width=100)),
                      row(widgetbox(V.context_ms_string, width=100),
                          widgetbox(V.shiftby_ms_string, width=100),
                          widgetbox(V.window_ms_string, width=100),
                          widgetbox(V.mel_dct_string, width=100),
                          widgetbox(V.stride_ms_string, width=100),
                          widgetbox(V.dropout_string, width=100),
                          widgetbox(V.optimizer, width=100),
                          widgetbox(V.learning_rate_string, width=100),
                          widgetbox(V.kernel_sizes_string, width=100),
                          widgetbox(V.last_conv_width_string, width=100),
                          widgetbox(V.nfeatures_string, width=120)),
                      row(widgetbox(V.cluster_equalize_ratio_string, width=100),
                          widgetbox(V.cluster_max_samples_string, width=100),
                          widgetbox(V.pca_fraction_variance_to_retain_string, width=100),
                          widgetbox(V.tsne_perplexity_string, width=100),
                          widgetbox(V.tsne_exaggeration_string, width=100),
                          widgetbox(V.precision_recall_ratios_string, width=150),
                          Spacer(width=M.gui_width_pix-710),
                          column(Spacer(height=40), V.editconfiguration)),
                      row(column(V.file_dialog_string, V.file_dialog_table),
                          V.configuration_contents)),
                  V.readme_contents)
curdoc().add_root(main_content)
