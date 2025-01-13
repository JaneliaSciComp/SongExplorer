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
augmentation_parameters = list(V.augmentation_parameters.values())

main_content = row(
        column(
            row(V.which_layer, V.which_species, V.which_word,
                V.which_nohyphen, V.which_kind, V.color_picker),
            row(V.p_cluster, V.p_snippets),
            row(V.dot_size,
                V.dot_alpha,
                V.circle_radius,
                Spacer(width=10),
                V.play,
                V.video_toggle,
                Spacer(width=10),
                *doubleclick_parameters,
                Spacer(width=10 if doubleclick_parameters else 0),
                V.save_indicator,
                V.undo,
                V.redo,
                Spacer(width=10),
                V.spectrogram_length \
                        if M.snippets_spectrogram or M.context_spectrogram \
                        else Spacer(width=10),
                V.zoom_width,
                V.zoom_offset, align=("center","center")),
            V.recordings,
            V.p_waveform if M.context_waveform else Spacer(height=1),
            V.p_spectrogram if M.context_spectrogram else Spacer(height=1),
            V.p_probability,
            Spacer(height=10),
            row(column(
                    row(Spacer(),  V.zoomin,  Spacer(), align="center"),
                    row(V.panleft, V.reset,   V.panright, align="center"),
                    row(Spacer(),  V.zoomout, Spacer(), align="center"),
                    row(V.allleft,  V.allout,  V.allright, align="center"),
                    row(V.prevlabel, V.nextlabel, align="center"),
                    row(V.firstlabel, V.lastlabel, align="center"),
                    Spacer(height=20),
                    PreText(text="version: \n"+version),
                    V.load_multimedia),
                Spacer(width=20),
                column(
                    *[row(x,y) for (x,y) in zip(V.nsounds_per_label_buttons, V.label_texts)],
                    row(V.remaining)),
                Spacer(width=20),
                column(V.video_slider,
                       V.video_div),
                Spacer(width=20),
                V.labelcounts)),
        column(
            row(V.labelsounds, V.makepredictions, V.fixfalsepositives,
                V.fixfalsenegatives, V.generalize, V.tunehyperparameters,
                V.examineerrors, V.testdensely, V.findnovellabels,
                V.doit, width=M.gui_width_pix),
            row(V.detect, V.misses, V.train, V.leaveout, V.xvalidate,
                V.mistakes, V.activations, V.cluster, V.visualize, V.accuracy, V.freeze,
                V.ensemble, V.classify, V.ethogram, V.compare, V.congruence,
                width=M.gui_width_pix),
            row(V.status_ticker, V.deletefailures, V.waitfor,
                width=M.gui_width_pix, align="center"),
            row(V.logs_folder_button, V.logs_folder, width=M.gui_width_pix),
            row(V.model_file_button, V.model_file, width=M.gui_width_pix),
            row(V.wavcsv_files_button, V.wavcsv_files, width=M.gui_width_pix),
            row(V.groundtruth_folder_button, V.groundtruth_folder, width=M.gui_width_pix),
            row(V.validation_files_button, V.validation_files, width=M.gui_width_pix),
            row(V.test_files_button, V.test_files, width=M.gui_width_pix),
            row(V.labels_touse_button, V.labels_touse, width=M.gui_width_pix),
            row(V.kinds_touse_button, V.kinds_touse,
                V.prevalences_button, V.prevalences,
                V.delete_ckpts, V.copy, width=M.gui_width_pix),
            row(V.nsteps, V.restore_from, V.weights_seed, V.optimizer, V.context,
                V.parallelize, V.mini_batch, V.nreplicates, V.activations_equalize_ratio,
                V.precision_recall_ratios, V.congruence_portion,
                width=M.gui_width_pix),
            row(V.save_and_validate_period, V.validate_percentage, V.batch_seed,
                V.learning_rate, V.shiftby, V.loss, V.kfold, V.activations_max_sounds,
                V.congruence_convolve, V.congruence_measure,
                width=M.gui_width_pix),
            row(column(*[row(*[column(detect_parameters[c],
                                      width=round(M.gui_width_pix/11*V.detect_parameters_width[c]))
                               for c in r])
                         for r in V.detect_parameters_partitioned],
                       background="seashell"),
                column(V.cluster_these_layers,
                       height=235,
                       width=M.gui_width_pix//11, background="honeydew"),
                column(*[row(*[column(cluster_parameters[c],
                                      width=round(M.gui_width_pix/11*V.cluster_parameters_width[c]))
                               for c in r])
                         for r in V.cluster_parameters_partitioned],
                       background="honeydew"),
                column(*[row(*[column(augmentation_parameters[c],
                                      width=round(M.gui_width_pix/11*V.augmentation_parameters_width[c]))
                               for c in r])
                         for r in V.augmentation_parameters_partitioned],
                       background="azure"),
                column(*[row(*[column(model_parameters[c],
                                      width=round(M.gui_width_pix/11*V.model_parameters_width[c]))
                               for c in r])
                         for r in V.model_parameters_partitioned],
                       background="ivory")),
            row(column(V.file_dialog_string,
                       V.file_dialog_table,
                       width=M.gui_width_pix//2),
                V.model_summary,
                width=M.gui_width_pix)),
        column(
            V.readme_contents,
            V.configuration_contents,
            width=M.gui_width_pix//2))

doc.add_root(main_content)
doc.add_periodic_callback(M.save_annotations, 5000)
