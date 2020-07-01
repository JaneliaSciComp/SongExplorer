# interactively browse clusters and annotate sounds
 
# visualize.py --args <audio-tic-rate> <audio-nchannels> <video-frame-rate> <snippets-ms> <nx-snippets> <ny-snippets> <gui-width-pix> <context-width-ms> <context-offset-ms> <cluster-background-color> <cluster-circle-color> <cluster-dot-colormap> <snippet-colormap>
# http://<hostname>:<port>/visualize

# e.g.
# bokeh serve --allow-websocket-origin=`hostname`:5006 --allow-websocket-origin=localhost:5006 --port 5006 --show /opt/users/deepsong/src/visualize.py --args 5000 1 15 40 10 10 8 1200 400 0 #FFFFFF #FF00FF Category10 Viridis256

import os
from bokeh.plotting import curdoc
from bokeh.layouts import column, row, Spacer
import threading
import logging 
from sys import argv

bokehlog = logging.getLogger("deepsong") 
bokehlog.setLevel(logging.INFO)
#bokehlog.info(...) 

_, configuration_file = argv

import model as M
import view as V
import controller as C

doc = curdoc()

M.init(configuration_file)
V.init(doc)
C.init(doc)

cluster_buttons = row(V.which_layer, V.which_species, V.which_word,
                      V.which_nohyphen, V.which_kind,
                      column(V.color_picker, width=75, align='end'),
                      width=M.gui_width_pix-75)
navigate_buttons = row(column(Spacer(height=41,width=40),
                              column(V.panleft, width=50),
                              Spacer(height=41,width=40),
                              column(V.allleft, width=50)),
                       column(V.zoomin,V.reset,V.zoomout,V.allout, width=50),
                       column(Spacer(height=41,width=40),
                              column(V.panright, width=50),
                              Spacer(height=41,width=40),
                              column(V.allright, width=50)))
label_widgets = row(column(V.label_count_widgets),
                    column(V.label_text_widgets, width=200))
main_content = row(column(cluster_buttons,
                          row(V.p_cluster, V.p_snippets),
                          row(column(V.dot_size, width=100),
                              column(V.dot_alpha, width=100),
                              column(V.circle_radius, width=100),
                              Spacer(width=(M.gui_width_pix-850)//3),
                              column(V.play, width=75, align='end'),
                              column(V.video_toggle, width=75, align='end'),
                              Spacer(width=(M.gui_width_pix-850)//3),
                              column(V.save_indicator, width=50, align='end'),
                              column(V.undo, width=75, align='end'),
                              column(V.redo, width=75, align='end'),
                              Spacer(width=(M.gui_width_pix-850)//3),
                              column(V.zoom_context, width=100),
                              column(V.zoom_offset, width=100)),
                          V.p_context,
                          Spacer(height=10),
                          row(navigate_buttons,
                              Spacer(width=20),
                              label_widgets,
                              Spacer(width=20),
                              V.video_div,
                              Spacer(width=20),
                              V.wordcounts, width=M.gui_width_pix)),
                   column(
                       row(row(V.labelsounds, V.makepredictions, V.fixfalsepositives,
                               V.fixfalsenegatives, V.generalize, V.tunehyperparameters,
                               V.examineerrors, V.testdensely, V.findnovellabels,
                               width=M.gui_width_pix-62),
                           row(V.doit,width=60)),
                       row(column(V.detect, width=M.gui_width_pix//16-11),
                           column(V.misses, width=M.gui_width_pix//16-6),
                           column(V.train, width=M.gui_width_pix//16-20),
                           column(V.leaveoneout, width=M.gui_width_pix//16+3),
                           column(V.leaveallout, width=M.gui_width_pix//16-5),
                           column(V.xvalidate, width=M.gui_width_pix//16+8),
                           column(V.mistakes, width=M.gui_width_pix//16+3),
                           column(V.activations, width=M.gui_width_pix//16+13),
                           column(V.cluster, width=M.gui_width_pix//16-8),
                           column(V.visualize, width=M.gui_width_pix//16+3),
                           column(V.accuracy, width=M.gui_width_pix//16+5),
                           column(V.freeze, width=M.gui_width_pix//16-10),
                           column(V.classify, width=M.gui_width_pix//16-5),
                           column(V.ethogram, width=M.gui_width_pix//16+7),
                           column(V.compare, width=M.gui_width_pix//16+4),
                           column(V.congruence, width=M.gui_width_pix//16-11)),
                       row(V.status_ticker, V.waitfor),
                       row(V.logs, column(V.logs_folder, width=M.gui_width_pix-120)),
                       row(V.model, column(V.model_file, width=M.gui_width_pix-120)),
                       row(V.wavtfcsvfiles,
                           column(V.wavtfcsvfiles_string, width=M.gui_width_pix-120)),
                       row(V.groundtruth,
                           column(V.groundtruth_folder, width=M.gui_width_pix-120)),
                       row(V.validationfiles,
                           column(V.validationfiles_string, width=M.gui_width_pix-120)),
                       row(V.testfiles,
                           column(V.testfiles_string, width=M.gui_width_pix-120)),
                       row(V.wantedwords,
                           column(V.wantedwords_string, width=M.gui_width_pix-120)),
                       row(V.labeltypes,
                           column(V.labeltypes_string, width=(M.gui_width_pix-70)//2-120),
                           V.prevalences,
                           column(V.prevalences_string, width=(M.gui_width_pix-70)//2-120),
                           column(V.copy, width=70)),
                       row(V.time_sigma_string,
                           V.time_smooth_ms_string,
                           V.nsteps_string,
                           V.restore_from_string,
                           V.representation,
                           V.window_ms_string,
                           V.stride_ms_string,
                           V.mel_dct_string,
                           V.connection_type,
                           V.activations_equalize_ratio_string,
                           V.activations_max_samples_string,
                           V.cluster_algorithm,
                           width=M.gui_width_pix),
                       row(
                           column(
                               row(V.frequency_n_ms_string,
                                   V.frequency_nw_string,
                                   V.save_and_validate_period_string,
                                   V.validate_percentage_string,
                                   V.kernel_sizes_string,
                                   V.last_conv_width_string,
                                   V.nfeatures_string,
                                   V.dilate_after_layer_string,
                                   V.stride_after_layer_string,
                                   V.umap_neighbors_string,
                                   V.pca_fraction_variance_to_retain_string,
                                   width=M.gui_width_pix-105),
                               row(V.frequency_p_string,
                                   V.frequency_smooth_ms_string,
                                   V.mini_batch_string,
                                   V.kfold_string,
                                   V.context_ms_string,
                                   V.shiftby_ms_string,
                                   V.optimizer,
                                   V.learning_rate_string,
                                   V.dropout_string,
                                   V.umap_distance_string,
                                   V.tsne_perplexity_string,
                                   width=M.gui_width_pix-105)),
                           column(V.cluster_these_layers, width=105)),
                       row(column(V.file_dialog_string,
                                  V.file_dialog_table, width=M.gui_width_pix//2),
                           column(row(column(Spacer(height=18),
                                             V.editconfiguration, width=50),
                                      Spacer(width=(M.gui_width_pix//2-55-415)),
                                      column(V.batch_seed_string, width=105),
                                      column(V.weights_seed_string, width=105),
                                      column(V.tsne_exaggeration_string, width=105),
                                      column(V.precision_recall_ratios_string, width=105)),
                                  V.configuration_contents, width=M.gui_width_pix//2))),
                   V.readme_contents)

doc.add_root(main_content)
doc.add_periodic_callback(M.save_annotations, 5000)
