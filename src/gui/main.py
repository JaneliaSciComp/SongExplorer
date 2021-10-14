# interactively browse clusters and annotate sounds
 
# visualize.py --args <audio-tic-rate> <audio-nchannels> <video-frame-rate> <snippets-ms> <nx-snippets> <ny-snippets> <gui-width-pix> <context-width-ms> <context-offset-ms> <cluster-background-color> <cluster-circle-color> <cluster-dot-colormap> <snippet-colormap>
# http://<hostname>:<port>/visualize

# e.g.
# bokeh serve --allow-websocket-origin=`hostname`:5006 --allow-websocket-origin=localhost:5006 --port 5006 --show /opt/songexplorer/src/visualize.py --args 5000 1 15 40 10 10 8 1200 400 0 #FFFFFF #FF00FF Category10 Viridis256

import os
from bokeh.plotting import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models.widgets import PreText
import threading
import logging 
from sys import argv

bokehlog = logging.getLogger("songexplorer") 
bokehlog.setLevel(logging.INFO)
#bokehlog.info(...) 

_, version, configuration_file = argv

import model as M
import view as V
import controller as C

doc = curdoc()

M.init(doc, configuration_file)
V.init(doc)
C.init(doc)

model_parameters = list(V.model_parameters.values())

main_content = row(column(row(V.which_layer, V.which_species, V.which_word,
                              V.which_nohyphen, V.which_kind,
                              column(V.color_picker, width=75),
                              width=M.gui_width_pix-75),
                          row(V.p_cluster, V.p_snippets),
                          row(column(V.dot_size, width=100),
                              column(V.dot_alpha, width=100),
                              column(V.circle_radius, width=100),
                              Spacer(width=(M.gui_width_pix-950)//3),
                              column(V.play, width=75, align='end'),
                              column(V.video_toggle, width=75, align='end'),
                              Spacer(width=(M.gui_width_pix-950)//3),
                              column(V.save_indicator, width=50, align='end'),
                              column(V.undo, width=75, align='end'),
                              column(V.redo, width=75, align='end'),
                              Spacer(width=(M.gui_width_pix-950)//3),
                              column(V.spectrogram_length, width=100) \
                                      if M.gui_snippets_spectrogram or \
                                         M.gui_context_spectrogram else Spacer(width=100),
                              column(V.zoom_context, width=100),
                              column(V.zoom_offset, width=100)),
                          V.recordings,
                          V.p_waveform if M.context_waveform==1 else Spacer(),
                          V.p_spectrogram if M.context_spectrogram==1 else Spacer(),
                          V.p_probability,
                          Spacer(height=10),
                          row(column(row(column(Spacer(height=42,width=40),
                                                column(V.panleft, width=50),
                                                Spacer(height=42,width=40),
                                                column(V.allleft, width=50)),
                                         column(V.zoomin,V.reset,V.zoomout,V.allout, width=50),
                                         column(Spacer(height=42,width=40),
                                                column(V.panright, width=50),
                                                Spacer(height=42,width=40),
                                                column(V.allright, width=50))),
                                     Spacer(height=60),
                                     PreText(text="version: \n"+version)),
                              Spacer(width=20),
                              row(column(V.nsounds_per_label_buttons),
                                  column(V.label_texts, width=200)),
                              Spacer(width=20),
                              V.video_div,
                              Spacer(width=20),
                              V.labelcounts, width=M.gui_width_pix)),
                   column(
                       row(row(V.labelsounds, V.makepredictions, V.fixfalsepositives,
                               V.fixfalsenegatives, V.generalize, V.tunehyperparameters,
                               V.examineerrors, V.testdensely, V.findnovellabels,
                               width=M.gui_width_pix-62),
                           row(V.doit,width=60)),
                       row(column(V.detect, width=M.gui_width_pix//17-11),
                           column(V.misses, width=M.gui_width_pix//17-6),
                           column(V.train, width=M.gui_width_pix//17-21),
                           column(V.leaveoneout, width=M.gui_width_pix//17+2),
                           column(V.leaveallout, width=M.gui_width_pix//17-5),
                           column(V.xvalidate, width=M.gui_width_pix//17+8),
                           column(V.mistakes, width=M.gui_width_pix//17+3),
                           column(V.activations, width=M.gui_width_pix//17+12),
                           column(V.cluster, width=M.gui_width_pix//17-9),
                           column(V.visualize, width=M.gui_width_pix//17+2),
                           column(V.accuracy, width=M.gui_width_pix//17+5),
                           column(V.freeze, width=M.gui_width_pix//17-10),
                           column(V.ensemble, width=M.gui_width_pix//17+8),
                           column(V.classify, width=M.gui_width_pix//17-5),
                           column(V.ethogram, width=M.gui_width_pix//17+7),
                           column(V.compare, width=M.gui_width_pix//17+3),
                           column(V.congruence, width=M.gui_width_pix//17-12)),
                       row(V.status_ticker, V.waitfor),
                       row(V.logs_folder_button, column(V.logs_folder, width=M.gui_width_pix-120)),
                       row(V.model_file_button, column(V.model_file, width=M.gui_width_pix-120)),
                       row(V.wavcsv_files_button,
                           column(V.wavcsv_files, width=M.gui_width_pix-120)),
                       row(V.groundtruth_folder_button,
                           column(V.groundtruth_folder, width=M.gui_width_pix-120)),
                       row(V.validation_files_button,
                           column(V.validation_files, width=M.gui_width_pix-120)),
                       row(V.test_files_button,
                           column(V.test_files, width=M.gui_width_pix-120)),
                       row(V.labels_touse_button,
                           column(V.labels_touse, width=M.gui_width_pix-120)),
                       row(V.kinds_touse_button,
                           column(V.kinds_touse, width=(M.gui_width_pix-70)//2-120),
                           V.prevalences_button,
                           column(V.prevalences, width=(M.gui_width_pix-70)//2-120),
                           column(V.copy, width=70)),
                       row(
                           column(
                               row(V.time_sigma,
                                   V.time_smooth_ms,
                                   V.nsteps,
                                   V.restore_from,
                                   V.weights_seed,
                                   V.optimizer,
                                   V.context_ms,
                                   V.activations_equalize_ratio,
                                   width=M.gui_width_pix-420),
                               row(V.frequency_n_ms,
                                   V.frequency_nw,
                                   V.save_and_validate_period,
                                   V.validate_percentage,
                                   V.batch_seed,
                                   V.learning_rate,
                                   V.shiftby_ms,
                                   V.activations_max_sounds,
                                   width=M.gui_width_pix-420),
                               row(V.frequency_p,
                                   V.frequency_smooth_ms,
                                   V.mini_batch,
                                   V.kfold,
                                   V.nreplicates,
                                   Spacer(width=340),
                                   width=M.gui_width_pix-420)),
                           column(V.cluster_these_layers,
                                  width=105),
                           column(
                               row(V.cluster_algorithm,
                                   V.pca_fraction_variance_to_retain,
                                   V.precision_recall_ratios,
                                   width=315),
                               row(V.tsne_perplexity,
                                   V.tsne_exaggeration,
                                   V.congruence_portion,
                                   width=315),
                               row(V.umap_neighbors,
                                   V.umap_distance,
                                   V.congruence_convolve,
                                   width=315))),
                       row(column(V.file_dialog_string,
                                  V.file_dialog_table),
                           column(*[row(model_parameters[i:min(len(model_parameters)+1,i+6)])
                                    for i in range(0,len(model_parameters),6)],
                                  V.configuration_contents,
                                  width=M.gui_width_pix//2))),
                   V.readme_contents)

doc.add_root(main_content)
doc.add_periodic_callback(M.save_annotations, 5000)
