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

_, version, configuration_file, use_aitch = argv

use_aitch = use_aitch == 'True'

import model as M
import view as V
import controller as C

doc = curdoc()

M.init(doc, configuration_file, use_aitch)
V.init(doc)
C.init(doc)

os.environ['PATH'] = os.pathsep.join([*M.bindirs, *os.environ['PATH'].split(os.pathsep)])

detect_parameters = list(V.detect_parameters.values())
doubleclick_parameters = list(V.doubleclick_parameters.values())
model_parameters = list(V.model_parameters.values())
cluster_parameters = list(V.cluster_parameters.values())

spacer_width = 790 + 100*(len(M.snippets_spectrogram)>0) + 103*len(doubleclick_parameters)
main_content = row(column(row(V.which_layer, V.which_species, V.which_word,
                              V.which_nohyphen, V.which_kind,
                              column(V.color_picker, width=75),
                              width=M.gui_width_pix-75),
                          row(V.p_cluster, V.p_snippets),
                          row(column(V.dot_size, width=100),
                              column(V.dot_alpha, width=100),
                              column(V.circle_radius, width=100),
                              Spacer(width=(M.gui_width_pix-spacer_width)//4),
                              column(V.play, width=60, align='end'),
                              column(V.video_toggle, width=60, align='end'),
                              Spacer(width=(M.gui_width_pix-spacer_width)//4),
                              *doubleclick_parameters,
                              Spacer(width=(M.gui_width_pix-spacer_width)//4),
                              column(V.save_indicator, width=50, align='end'),
                              column(V.undo, width=60, align='end'),
                              column(V.redo, width=60, align='end'),
                              Spacer(width=(M.gui_width_pix-spacer_width)//4),
                              column(V.spectrogram_length, width=100) \
                                      if M.snippets_spectrogram or \
                                         M.context_spectrogram else Spacer(),
                              column(V.zoom_width, width=100),
                              column(V.zoom_offset, width=100)),
                          V.recordings,
                          V.p_waveform if M.context_waveform else Spacer(),
                          V.p_spectrogram if M.context_spectrogram else Spacer(),
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
                                     row(Spacer(height=42,width=15),
                                         column(V.prevlabel, width=60),
                                         column(V.nextlabel, width=60),
                                         Spacer(height=42,width=15), width=150),
                                     row(Spacer(height=42,width=15),
                                         column(V.firstlabel, width=60),
                                         column(V.lastlabel, width=60),
                                         Spacer(height=42,width=15), width=150),
                                     Spacer(height=20),
                                     PreText(text="version: \n"+version),
                                     V.load_multimedia),
                              Spacer(width=20),
                              column(row(column(V.nsounds_per_label_buttons),
                                         column(V.label_texts, width=200)),
                                     row(V.remaining, width=100)),
                              Spacer(width=20),
                              column(V.video_slider,
                                     V.video_div),
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
                       row(V.status_ticker,
                           column(V.deletefailures, width=110),
                           column(V.waitfor, width=105)),
                       row(V.logs_folder_button, column(V.logs_folder, width=M.gui_width_pix-130)),
                       row(V.model_file_button, column(V.model_file, width=M.gui_width_pix-130)),
                       row(V.wavcsv_files_button,
                           column(V.wavcsv_files, width=M.gui_width_pix-130)),
                       row(V.groundtruth_folder_button,
                           column(V.groundtruth_folder, width=M.gui_width_pix-130)),
                       row(V.validation_files_button,
                           column(V.validation_files, width=M.gui_width_pix-130)),
                       row(V.test_files_button,
                           column(V.test_files, width=M.gui_width_pix-130)),
                       row(V.labels_touse_button,
                           column(V.labels_touse, width=M.gui_width_pix-130)),
                       row(V.kinds_touse_button,
                           column(V.kinds_touse, width=(M.gui_width_pix-161)//2-130),
                           V.prevalences_button,
                           column(V.prevalences, width=(M.gui_width_pix-161)//2-130),
                           column(V.delete_ckpts, width=100),
                           column(V.copy, width=60)),
                       row(
                           column(
                               row(V.nsteps,
                                   V.restore_from,
                                   V.weights_seed,
                                   V.optimizer,
                                   V.context,
                                   V.mini_batch,
                                   V.nreplicates,
                                   V.activations_equalize_ratio,
                                   width=M.gui_width_pix-420),
                               row(V.save_and_validate_period,
                                   V.validate_percentage,
                                   V.batch_seed,
                                   V.learning_rate,
                                   V.shiftby,
                                   V.loss,
                                   V.kfold,
                                   V.activations_max_sounds,
                                   width=M.gui_width_pix-420),
                               *[row([detect_parameters[x] for x in p],
                                     width=M.gui_width_pix-420)
                                 for p in V.detect_parameters_partitioned]),
                           column(V.precision_recall_ratios,
                                  V.cluster_these_layers,
                                  width=105),
                           column(
                               *[row([cluster_parameters[x] for x in p],
                                     width=210)
                                 for p in V.cluster_parameters_partitioned]),
                           column(V.congruence_portion,
                                  V.congruence_convolve,
                                  V.congruence_measure,
                                  width=105)),
                       row(column(V.file_dialog_string,
                                  V.file_dialog_table),
                           column(row(V.augment_volume,
                                      V.augment_noise,
                                      V.augment_dc,
                                      V.augment_reverse,
                                      V.augment_invert,
                                      width=M.gui_width_pix//2),
                                  *[row([model_parameters[x] for x in p])
                                    for p in V.model_parameters_partitioned],
                                  V.model_summary,
                                  width=M.gui_width_pix//2))),
                   column(
                       V.readme_contents,
                       V.configuration_contents))

doc.add_root(main_content)
doc.add_periodic_callback(M.save_annotations, 5000)
