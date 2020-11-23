import os
from bokeh.models.widgets import RadioButtonGroup, TextInput, Button, Div, DateFormatter, TextAreaInput, Select, NumberFormatter, Slider, Toggle, ColorPicker, MultiSelect
from bokeh.models import ColumnDataSource, TableColumn, DataTable, LayoutDOM
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.events import Tap, DoubleTap, PanStart, Pan, PanEnd, ButtonClick, MouseWheel
from bokeh.models.callbacks import CustomJS
from bokeh.models.markers import Circle
import numpy as np
import glob
from datetime import datetime
import markdown
import pandas as pd
import wave
import scipy.io.wavfile as spiowav
from scipy.signal import decimate, spectrogram
import logging 
import base64
import io
from natsort import natsorted
import pims
import av
from bokeh import palettes
from itertools import cycle, product
import ast
from bokeh.core.properties import Instance, String, List, Float
from bokeh.util.compiler import TypeScript
import asyncio

bokehlog = logging.getLogger("songexplorer") 

import model as M
import controller as C

bokeh_document, cluster_dot_palette, snippet_palette, p_cluster, cluster_dots, p_snippets, label_sources, label_sources_new, wav_sources, line_glyphs, quad_grey_snippets, dot_size_cluster, dot_alpha_cluster, circle_fuchsia_cluster, p_context, p_spectrogram, spectrogram_source, image_glyph, p_line_red_context, line_red_context, quad_grey_context_old, quad_grey_context_new, quad_grey_context_pan, quad_fuchsia_context, quad_fuchsia_snippets, wav_source, line_glyph, label_source, label_source_new, which_layer, which_species, which_word, which_nohyphen, which_kind, color_picker, circle_radius, dot_size, dot_alpha, zoom_context, zoom_offset, zoomin, zoomout, reset, panleft, panright, allleft, allout, allright, save_indicator, label_count_widgets, label_text_widgets, play, play_callback, video_toggle, video_div, undo, redo, detect, misses, configuration_file, train, leaveoneout, leaveallout, xvalidate, mistakes, activations, cluster, visualize, accuracy, freeze, classify, ethogram, compare, congruence, status_ticker, waitfor, file_dialog_source, file_dialog_source, configuration_contents, logs, logs_folder, model, model_file, wavtfcsvfiles, wavtfcsvfiles_string, groundtruth, groundtruth_folder, validationfiles, testfiles, validationfiles_string, testfiles_string, wantedwords, wantedwords_string, labeltypes, labeltypes_string, prevalences, prevalences_string, copy, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, generalize, tunehyperparameters, findnovellabels, examineerrors, testdensely, doit, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string, nsteps_string, restore_from_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string, kfold_string, activations_equalize_ratio_string, activations_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, umap_neighbors_string, umap_distance_string, cluster_algorithm, cluster_these_layers, connection_type, precision_recall_ratios_string, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, replicates_string, batch_seed_string, weights_seed_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, editconfiguration, file_dialog_string, file_dialog_table, readme_contents, wordcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs = [None]*160

class ScatterNd(LayoutDOM):

    __implementation__ = TypeScript("""
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"
import {ColumnDataSource} from "models/sources/column_data_source"
import {LayoutItem} from "core/layout"
import * as p from "core/properties"

declare namespace Plotly {
  class newPlot { constructor(el: HTMLElement, data: object, OPTIONS: object) }
}

let OPTIONS2 = {
  margin: { l: 0, r: 0, b: 0, t: 0 },
  showlegend: false,
  xaxis: { visible: false },
  yaxis: { visible: false },
  hovermode: 'closest',
  shapes: [ {
      type: 'circle',
      xref: 'x', yref: 'y',
      x0: 0, y0: 0,
      x1: 0, y1: 0,
      line: { color: 'fuchsia' } } ]
}
let OPTIONS3 = {
  margin: { l: 0, r: 0, b: 0, t: 0 },
  hovermode: 'closest',
  hoverlabel: { bgcolor: 'white' },
  showlegend: false,
  scene: {
    xaxis: { visible: false },
    yaxis: { visible: false },
    zaxis: { visible: false },
  },
  shapes: [],
}

// https://github.com/caosdoar/spheres
let icosphere12 = [[0.525731, 0.850651, 0]]
let icosphere42 = icosphere12.slice().concat([[0.809017, 0.5, 0.309017],
                                              [0, 0, 1]])
let icosphere162 = icosphere42.slice().concat([[0.69378, 0.702046, 0.160622],
                                               [0.587785, 0.688191, 0.425325],
                                               [0.433889, 0.862668, 0.259892],
                                               [0.273267, 0.961938, 0],
                                               [0.16246, 0.951057, 0.262866]])

// @ts-ignore
let xicosphere = []
// @ts-ignore
let yicosphere = []
// @ts-ignore
let zicosphere = []
icosphere162.forEach((x)=>{
  // @ts-ignore
  let V = []
  for (let i=1; i==1 || i==-1 && x[0]>0; i-=2) {
    for (let j=1; j==1 || j==-1 && x[1]>0; j-=2) {
      for (let k=1; k==1 || k==-1 && x[2]>0; k-=2) {
        // @ts-ignore
        V = V.concat([i*x[0], j*x[1], k*x[2]])
      }
    }
  }
  // @ts-ignore
  xicosphere = xicosphere.concat(V)
  V.push(V.shift())
  // @ts-ignore
  yicosphere = yicosphere.concat(V)
  V.push(V.shift())
  // @ts-ignore
  zicosphere = zicosphere.concat(V)
});

export class ScatterNdView extends LayoutDOMView {
  model: ScatterNd

  initialize(): void {
    super.initialize()

    const url = "https://cdn.plot.ly/plotly-latest.min.js"
    const script = document.createElement("script")
    script.onload = () => this._init()
    script.async = false
    script.src = url
    document.head.appendChild(script)
  }

  ndims() {
    if (this.model.dots_source.data[this.model.dz].length==0) {
      return 0 }
    else if (isNaN(this.model.dots_source.data[this.model.dz][0])) {
      return 2 }
    return 3
  }

  get_dots_data() {
    return {x: this.model.dots_source.data[this.model.dx],
            y: this.model.dots_source.data[this.model.dy],
            z: this.model.dots_source.data[this.model.dz],
            text: this.model.dots_source.data[this.model.dl],
            marker: {
              color: this.model.dots_source.data[this.model.dc],
              size: this.model.dot_size_source.data[this.model.ds][0],
              opacity: this.model.dot_alpha_source.data[this.model.da][0],
            }
           };
  }

  set_circle_fuchsia_data2() {
    if (this.model.circle_fuchsia_source.data[this.model.cx].length==0) {
      OPTIONS2.shapes[0].x0 = 0
      OPTIONS2.shapes[0].y0 = 0
      OPTIONS2.shapes[0].x1 = 0
      OPTIONS2.shapes[0].y1 = 0 }
    else {
      OPTIONS2.shapes[0].line.color = this.model.circle_fuchsia_source.data[this.model.cc][0]
      let x = this.model.circle_fuchsia_source.data[this.model.cx][0]
      let y = this.model.circle_fuchsia_source.data[this.model.cy][0]
      let r = this.model.circle_fuchsia_source.data[this.model.cr][0]
      OPTIONS2.shapes[0].x0 = x-r
      OPTIONS2.shapes[0].y0 = y-r
      OPTIONS2.shapes[0].x1 = x- -r
      OPTIONS2.shapes[0].y1 = y- -r }
  }

  get_circle_fuchsia_data3() {
    if (this.model.circle_fuchsia_source.data[this.model.cx].length==0) {
      return {type: 'mesh3d',
              x:[0], y:[0], z:[0],
             }; }
    else {
      let radius = this.model.circle_fuchsia_source.data[this.model.cr][0]
      return {type: 'mesh3d',
              // @ts-ignore
              x: xicosphere.map(x=>x*radius+this.model.circle_fuchsia_source.data[this.model.cx][0]),
              // @ts-ignore
              y: yicosphere.map(x=>x*radius+this.model.circle_fuchsia_source.data[this.model.cy][0]),
              // @ts-ignore
              z: zicosphere.map(x=>x*radius+this.model.circle_fuchsia_source.data[this.model.cz][0]),
             }; }
  }

  private _init(): void {
    new Plotly.newPlot(this.el,
                       [{alphahull: 1.0,
                         opacity: 0.2,
                        },
                        {hovertemplate: "%{text}<extra></extra>",
                         mode: 'markers',
                        }],
                       {xaxis: { visible: false },
                        yaxis: { visible: false } });

    this.connect(this.model.dots_source.change, () => {
      let new_data = this.get_dots_data()
      let N = this.ndims()
      if (N==2) {
        this.set_circle_fuchsia_data2()
        // @ts-ignore
        Plotly.update(this.el, {type: '', x:[[]], y:[[]], z:[[]]}, OPTIONS2, [0]);
        // @ts-ignore
        Plotly.update(this.el,
                      {type: 'scatter',
                       x: [new_data['x']], y: [new_data['y']],
                       text: [new_data['text']],
                       marker: new_data['marker'] },
                      OPTIONS2,
                      [1]);
      }
      else if (N==3) {
        // @ts-ignore
        Plotly.update(this.el, {type: 'mesh3d', x:[[]], y:[[]], z:[[]]}, OPTIONS3, [0]);
        // @ts-ignore
        Plotly.update(this.el,
                       {type: 'scatter3d',
                        x: [new_data['x']], y: [new_data['y']], z: [new_data['z']],
                        text: [new_data['text']],
                        marker: new_data['marker'] },
                       OPTIONS3,
                       [1]);
      }
    });

    this.connect(this.model.dot_size_source.change, () => {
      let new_data = this.get_dots_data()
      // @ts-ignore
      Plotly.restyle(this.el, { marker: new_data['marker'] }, [1]);
    });

    this.connect(this.model.dot_alpha_source.change, () => {
      let new_data = this.get_dots_data()
      // @ts-ignore
      Plotly.restyle(this.el, { marker: new_data['marker'] }, [1]);
    });

    // @ts-ignore
    (<HTMLDivElement>this.el).on('plotly_click', (data) => {
      let N = this.ndims()
      if (N==2) {
        // @ts-ignore
        this.model.click_position = [data.points[0].x,data.points[0].y] }
      else if (N==3) {
        // @ts-ignore
        this.model.click_position = [data.points[0].x,data.points[0].y,data.points[0].z] }
    });

    this.connect(this.model.circle_fuchsia_source.change, () => {
      let N = this.ndims()
      if (N==2) {
        this.set_circle_fuchsia_data2()
        // @ts-ignore
        Plotly.relayout(this.el, OPTIONS2); }
      else if (N==3) {
        let new_data = this.get_circle_fuchsia_data3()
        // @ts-ignore
        Plotly.restyle(this.el,
                       {x: [new_data['x']], y: [new_data['y']], z: [new_data['z']],
                        color: this.model.circle_fuchsia_source.data[this.model.cc][0]},
                       [0]); }
    });
  }

  get child_models(): LayoutDOM[] { return [] }

  _update_layout(): void {
    this.layout = new LayoutItem()
    this.layout.set_sizing(this.box_sizing())
  }
}

export namespace ScatterNd {
  export type Attrs = p.AttrsOf<Props>

  export type Props = LayoutDOM.Props & {
    cx: p.Property<string>
    cy: p.Property<string>
    cz: p.Property<string>
    cr: p.Property<string>
    cc: p.Property<string>
    dx: p.Property<string>
    dy: p.Property<string>
    dz: p.Property<string>
    dl: p.Property<string>
    dc: p.Property<string>
    ds: p.Property<string>
    da: p.Property<string>
    click_position: p.Property<number[]>
    circle_fuchsia_source: p.Property<ColumnDataSource>
    dots_source: p.Property<ColumnDataSource>
    dot_size_source: p.Property<ColumnDataSource>
    dot_alpha_source: p.Property<ColumnDataSource>
  }
}

export interface ScatterNd extends ScatterNd.Attrs {}

export class ScatterNd extends LayoutDOM {
  properties: ScatterNd.Props

  constructor(attrs?: Partial<ScatterNd.Attrs>) { super(attrs) }

  static __name__ = "ScatterNd"

  static init_ScatterNd() {
    this.prototype.default_view = ScatterNdView

    this.define<ScatterNd.Props>({
      cx: [ p.String   ],
      cy: [ p.String   ],
      cz: [ p.String   ],
      cr: [ p.String   ],
      cc: [ p.String   ],
      dx: [ p.String   ],
      dy: [ p.String   ],
      dz: [ p.String   ],
      dl: [ p.String   ],
      dc: [ p.String   ],
      ds: [ p.String   ],
      da: [ p.String   ],
      click_position:  [ p.Array   ],
      circle_fuchsia_source: [ p.Instance ],
      dots_source: [ p.Instance ],
      dot_size_source: [ p.Instance ],
      dot_alpha_source: [ p.Instance ],
    })
  }
}
"""
)

    cx = String
    cy = String
    cz = String
    cr = String
    cc = String

    dx = String
    dy = String
    dz = String
    dl = String
    dc = String
    ds = String
    da = String

    click_position = List(Float)

    circle_fuchsia_source = Instance(ColumnDataSource)
    dots_source = Instance(ColumnDataSource)
    dot_size_source = Instance(ColumnDataSource)
    dot_alpha_source = Instance(ColumnDataSource)

def cluster_initialize(newcolors=True):
    global precomputed_dots
    global p_cluster_xmax, p_cluster_ymax, p_cluster_zmax
    global p_cluster_xmin, p_cluster_ymin, p_cluster_zmin

    cluster_file = os.path.join(groundtruth_folder.value,'cluster.npz')
    if not os.path.isfile(cluster_file):
        bokehlog.info("ERROR: "+cluster_file+" not found")
        return False
    npzfile = np.load(cluster_file, allow_pickle=True)
    M.clustered_samples = npzfile['samples']
    M.clustered_activations = npzfile['activations_clustered']

    M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_samples]
    isort = np.argsort(M.clustered_starts_sorted)
    for i in range(len(M.clustered_activations)):
        if M.clustered_activations[i] is not None:
            layer0 = i
            M.clustered_activations[i] = M.clustered_activations[i][isort,:]
    M.clustered_samples = [M.clustered_samples[x] for x in isort]
    M.clustered_starts_sorted = [M.clustered_starts_sorted[x] for x in isort]

    M.clustered_stops = [x['ticks'][1] for x in M.clustered_samples]
    M.iclustered_stops_sorted = np.argsort(M.clustered_stops)

    cluster_isnotnan = [not np.isnan(x[0]) and not np.isnan(x[1]) \
                        for x in M.clustered_activations[layer0]]

    M.nlayers = len(M.clustered_activations)
    M.ndcluster = np.shape(M.clustered_activations[layer0])[1]
    cluster_dots.data.update(dx=[], dy=[], dz=[], dl=[], dc=[])
    circle_fuchsia_cluster.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])

    M.layers = ["input"]+["hidden #"+str(i) for i in range(1,M.nlayers-1)]+["output"]
    M.species = set([x['label'].split('-')[0]+'-' \
                     for x in M.clustered_samples if '-' in x['label']])
    M.species |= set([''])
    M.species = natsorted(list(M.species))
    M.words = set(['-'+x['label'].split('-')[1] \
                   for x in M.clustered_samples if '-' in x['label']])
    M.words |= set([''])
    M.words = natsorted(list(M.words))
    M.nohyphens = set([x['label'] for x in M.clustered_samples if '-' not in x['label']])
    M.nohyphens |= set([''])
    M.nohyphens = natsorted(list(M.nohyphens))
    M.kinds = natsorted(list(set([x['kind'] for x in M.clustered_samples])))

    if newcolors:
        allcombos = [x[0][:-1]+x[1] for x in product(M.species[1:], M.words[1:])]
        M.cluster_dot_colors = { l:c for l,c in zip(allcombos+ M.nohyphens[1:],
                                                    cycle(cluster_dot_palette)) }

    p_cluster_xmin, p_cluster_xmax = [0]*M.nlayers, [0]*M.nlayers
    p_cluster_ymin, p_cluster_ymax = [0]*M.nlayers, [0]*M.nlayers
    p_cluster_zmin, p_cluster_zmax = [0]*M.nlayers, [0]*M.nlayers
    precomputed_dots = [None]*M.nlayers
    for ilayer in range(M.nlayers):
        precomputed_dots[ilayer] = [None]*len(M.species)
        if M.clustered_activations[ilayer] is not None:
            p_cluster_xmin[ilayer] = np.min(M.clustered_activations[ilayer][:,0])
            p_cluster_xmax[ilayer] = np.max(M.clustered_activations[ilayer][:,0])
            p_cluster_ymin[ilayer] = np.min(M.clustered_activations[ilayer][:,1])
            p_cluster_ymax[ilayer] = np.max(M.clustered_activations[ilayer][:,1])
            if M.ndcluster==3:
                p_cluster_zmin[ilayer] = np.min(M.clustered_activations[ilayer][:,2])
                p_cluster_zmax[ilayer] = np.max(M.clustered_activations[ilayer][:,2])
        for (ispecies,specie) in enumerate(M.species):
            precomputed_dots[ilayer][ispecies] = [None]*len(M.words)
            for (iword,word) in enumerate(M.words):
                precomputed_dots[ilayer][ispecies][iword] = [None]*len(M.nohyphens)
                for (inohyphen,nohyphen) in enumerate(M.nohyphens):
                    precomputed_dots[ilayer][ispecies][iword][inohyphen] = \
                            [None]*len(M.kinds)
                    for (ikind,kind) in enumerate(M.kinds):
                        if inohyphen!=0 and (ispecies!=0 or iword!=0):
                            continue
                        if M.clustered_activations[ilayer] is None:
                            continue
                        bidx = np.logical_and([specie in x['label'] and \
                                               word in x['label'] and \
                                               (nohyphen=="" or nohyphen==x['label']) and \
                                               kind==x['kind'] \
                                               for x in M.clustered_samples], \
                                               cluster_isnotnan)
                        if not any(bidx):
                            continue
                        if inohyphen>0:
                            colors = [M.cluster_dot_colors[nohyphen] for b in bidx if b]
                        else:
                            colors = [M.cluster_dot_colors[x['label']] \
                                      if x['label'] in M.cluster_dot_colors else "black" \
                                      for x,b in zip(M.clustered_samples,bidx) if b]
                        data = {'x': M.clustered_activations[ilayer][bidx,0], \
                                'y': M.clustered_activations[ilayer][bidx,1], \
                                'l': [x['label'] for x,b in zip(M.clustered_samples,bidx) if b], \
                                'c': colors }
                        if M.ndcluster==2:
                            data['z'] = [np.nan]*len(M.clustered_activations[ilayer][bidx,1])
                        else:
                            data['z'] = M.clustered_activations[ilayer][bidx,2]
                        precomputed_dots[ilayer][ispecies][iword][inohyphen][ikind] = data

    which_layer.options = M.layers
    which_species.options = M.species
    which_word.options = M.words
    which_nohyphen.options = M.nohyphens
    which_kind.options = M.kinds

    circle_radius.disabled=False
    dot_size.disabled=False
    dot_alpha.disabled=False

    M.ilayer=0
    M.ispecies=0
    M.iword=0
    M.inohyphen=0
    M.ikind=0

    return True

def cluster_update():
    global cluster_dots
    global p_cluster_xmax, p_cluster_xmin, p_cluster_ymax, p_cluster_ymin
    dot_alpha.disabled=False
    if precomputed_dots == None:
        return
    selected_dots = precomputed_dots[M.ilayer][M.ispecies][M.iword][M.inohyphen][M.ikind]
    if selected_dots is None:
        kwargs = dict(dx=[0,0,0,0,0,0,0,0],
                      dy=[0,0,0,0,0,0,0,0],
                      dz=[0,0,0,0,0,0,0,0],
                      dl=['', '', '', '', '', '', '', ''],
                      dc=['#ffffff00', '#ffffff00', '#ffffff00', '#ffffff00',
                          '#ffffff00', '#ffffff00', '#ffffff00', '#ffffff00'])
    else:
        kwargs = dict(dx=[*selected_dots['x'],
                          p_cluster_xmin[M.ilayer], p_cluster_xmin[M.ilayer],
                          p_cluster_xmin[M.ilayer], p_cluster_xmin[M.ilayer],
                          p_cluster_xmax[M.ilayer], p_cluster_xmax[M.ilayer],
                          p_cluster_xmax[M.ilayer], p_cluster_xmax[M.ilayer]],
                      dy=[*selected_dots['y'],
                          p_cluster_ymin[M.ilayer], p_cluster_ymin[M.ilayer],
                          p_cluster_ymax[M.ilayer], p_cluster_ymax[M.ilayer],
                          p_cluster_ymin[M.ilayer], p_cluster_ymin[M.ilayer],
                          p_cluster_ymax[M.ilayer], p_cluster_ymax[M.ilayer]],
                      dz=[*selected_dots['z'],
                          p_cluster_zmin[M.ilayer], p_cluster_zmax[M.ilayer],
                          p_cluster_zmin[M.ilayer], p_cluster_zmax[M.ilayer],
                          p_cluster_zmin[M.ilayer], p_cluster_zmax[M.ilayer],
                          p_cluster_zmin[M.ilayer], p_cluster_zmax[M.ilayer]],
                      dl=[*selected_dots['l'], '', '', '', '', '', '', '', ''],
                      dc=[*selected_dots['c'],
                          '#ffffff00', '#ffffff00', '#ffffff00', '#ffffff00',
                          '#ffffff00', '#ffffff00', '#ffffff00', '#ffffff00'])
    cluster_dots.data.update(**kwargs)
    extent = min(p_cluster_xmax[M.ilayer] - p_cluster_xmin[M.ilayer],
                 p_cluster_ymax[M.ilayer] - p_cluster_ymin[M.ilayer])
    if M.ndcluster==3:
        extent = min(extent, p_cluster_zmax[M.ilayer] - p_cluster_zmin[M.ilayer])
    circle_radius.end = max(np.finfo(np.float32).eps, extent)
    circle_radius.step = extent/100
    #npoints = np.shape(M.clustered_activations[M.ilayer])[0]
    #dot_size.value = max(1, round(100 * extent / np.sqrt(npoints)))

def within_an_annotation(sample):
    if len(M.annotated_starts_sorted)>0:
        ifrom = np.searchsorted(M.annotated_starts_sorted, sample['ticks'][0],
                                side='right') - 1
        if 0 <= ifrom and ifrom < len(M.annotated_starts_sorted) and \
                    M.annotated_samples[ifrom]['ticks'][1] >= sample['ticks'][1]:
            return ifrom
    return -1

def snippets_update(redraw_wavs):
    if len(M.species)==0:
        return
    if M.isnippet>0 and not np.isnan(M.xcluster) and not np.isnan(M.ycluster) \
                and (M.ndcluster==2 or not np.isnan(M.zcluster)):
        quad_fuchsia_snippets.data.update(
                left=[M.xsnippet*(M.snippets_gap_pix+M.snippets_pix)],
                right=[(M.xsnippet+1)*(M.snippets_gap_pix+M.snippets_pix)-
                       M.snippets_gap_pix],
                top=[-M.ysnippet*2+1], bottom=[-M.ysnippet*2-1])
    else:
        quad_fuchsia_snippets.data.update(left=[], right=[], top=[], bottom=[])

    isubset = np.where([M.species[M.ispecies] in x['label'] and
                      M.words[M.iword] in x['label'] and
                      (M.nohyphens[M.inohyphen]=="" or \
                       M.nohyphens[M.inohyphen]==x['label']) and
                      M.kinds[M.ikind]==x['kind'] for x in M.clustered_samples])[0]
    origin = [M.xcluster,M.ycluster]
    if M.ndcluster==3:
        origin.append(M.zcluster)
    distance = [] if M.clustered_activations[M.ilayer] is None else \
               np.linalg.norm(M.clustered_activations[M.ilayer][isubset,:] - origin, \
                              axis=1)
    isort = np.argsort(distance)
    songs, labels, labels_new, scales = [], [], [], []
    for isnippet in range(M.nx*M.ny):
        if isnippet<len(distance) and \
                    distance[isort[isnippet]] < float(M.state["circle_radius"]):
            M.nearest_samples[isnippet] = isubset[isort[isnippet]]
            thissample = M.clustered_samples[M.nearest_samples[isnippet]]
            labels.append(thissample['label'])
            midpoint = np.mean(thissample['ticks'], dtype=int)
            if redraw_wavs:
                _, wavs = spiowav.read(thissample['file'], mmap=True)
                if np.ndim(wavs)==1:
                  wavs = np.expand_dims(wavs, axis=1)
                start_frame = max(0, midpoint-M.snippets_tic//2)
                nframes_to_get = min(np.shape(wavs)[0] - start_frame,
                                     M.snippets_tic+1,
                                     M.snippets_tic+1+(midpoint-M.snippets_tic//2))
                left_pad = max(0, M.snippets_pix-nframes_to_get if start_frame==0 else 0)
                right_pad = max(0, M.snippets_pix-nframes_to_get if start_frame>0 else 0)
                song = [[]]*M.audio_nchannels
                scale = [[]]*M.audio_nchannels
                for ichannel in range(M.audio_nchannels):
                    wavi = wavs[start_frame : start_frame+nframes_to_get, ichannel]
                    wavi = decimate(wavi, M.snippets_decimate_by,
                                    n=M.filter_order,
                                    ftype='iir', zero_phase=True)
                    np.pad(wavi, ((left_pad, right_pad),),
                           'constant', constant_values=(np.nan,))
                    wavi = wavi[:M.snippets_pix]
                    scale[ichannel]=np.minimum(np.iinfo(np.int16).max-1,
                                               np.max(np.abs(wavi)))
                    song[ichannel]=wavi/scale[ichannel]
                songs.append(song)
                scales.append(scale)
            else:
                songs.append([[]])
            iannotated = within_an_annotation(thissample)
            if iannotated == -1:
                labels_new.append('')
            else:
                labels_new.append(M.annotated_samples[iannotated]['label'])
        else:
            M.nearest_samples[isnippet] = -1
            labels.append('')
            labels_new.append('')
            scales.append([0]*M.audio_nchannels)
            songs.append([np.full(M.snippets_pix,np.nan)]*M.audio_nchannels)
    label_sources.data.update(text=labels)
    label_sources_new.data.update(text=labels_new)
    left, right, top, bottom = [], [], [], []
    for (isong,song) in enumerate(songs):
        ix, iy = isong%M.nx, isong//M.nx
        if redraw_wavs:
            xdata = range(ix*(M.snippets_gap_pix+M.snippets_pix),
                          (ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            for ichannel in range(M.audio_nchannels):
                ydata = -iy*2 + \
                        (M.audio_nchannels-1-2*ichannel)/M.audio_nchannels + \
                        song[ichannel]/M.audio_nchannels
                wav_sources[isong][ichannel].data.update(x=xdata, y=ydata)
                ipalette = int(np.floor(scales[isong][ichannel] /
                                        np.iinfo(np.int16).max *
                                        len(snippet_palette)))
                line_glyphs[isong][ichannel].glyph.line_color = snippet_palette[ipalette]
        if labels_new[isong]!='':
            left.append(ix*(M.snippets_gap_pix+M.snippets_pix))
            right.append((ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            top.append(-iy*2+1)
            bottom.append(-iy*2-1)
    quad_grey_snippets.data.update(left=left, right=right, top=top, bottom=bottom)

def nparray2base64wav(data, samplerate):
    fid=io.BytesIO()
    wav=wave.open(fid, "w")
    wav.setframerate(samplerate)
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.writeframes(data.tobytes())
    wav.close()
    fid.seek(0)
    ret_val = base64.b64encode(fid.read()).decode('utf-8')
    fid.close()
    return ret_val

def nparray2base64mp4(filename, start_sec, stop_sec):
    vid = pims.open(filename)

    start_frame = round(start_sec * vid.frame_rate).astype(np.int)
    stop_frame = round(stop_sec * vid.frame_rate).astype(np.int)

    fid=io.BytesIO()
    container = av.open(fid, mode='w', format='mp4')

    stream = container.add_stream('h264', rate=vid.frame_rate)
    stream.width = video_div.width = vid.frame_shape[0]
    stream.height = video_div.height = vid.frame_shape[1]
    stream.pix_fmt = 'yuv420p'

    for iframe in range(start_frame, stop_frame):
        frame = av.VideoFrame.from_ndarray(np.array(vid[iframe]), format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    fid.seek(0)
    ret_val = base64.b64encode(fid.read()).decode('utf-8')
    fid.close()
    return ret_val

# _context_update() might be able to be folded back in to context_update() with bokeh 2.0
# ditto for _doit_callback() and _groundtruth_update()
# see https://discourse.bokeh.org/t/bokeh-server-is-it-possible-to-push-updates-to-js-in-the-middle-of-a-python-callback/3455/4

def __context_update(wavi, tapped_sample, istart_bounded, ilength):
    if video_toggle.active:
        sample_basename=os.path.basename(tapped_sample)
        sample_dirname=os.path.dirname(tapped_sample)
        vids = list(filter(lambda x: x!=sample_basename and
                                     os.path.splitext(x)[0] == \
                                         os.path.splitext(sample_basename)[0] and
                                     os.path.splitext(x)[1].lower() in \
                                         ['.avi','.mp4','.mov'],
                           os.listdir(sample_dirname)))
        base64vid = nparray2base64mp4(os.path.join(sample_dirname,vids[0]),
                                      istart_bounded / M.audio_tic_rate,
                                      (istart_bounded+ilength) / M.audio_tic_rate) \
                    if len(vids)==1 else ""
        video_toggle.button_type="default"
    else:
        base64vid = ""

    play_callback.code = C.play_callback_code % \
                         (nparray2base64wav(wavi, M.audio_tic_rate), \
                          base64vid)

def _context_update(wavi, tapped_sample, istart_bounded, ilength):
    if video_toggle.active:
        video_toggle.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: \
            __context_update(wavi, tapped_sample, istart_bounded, ilength))

def context_update():
    p_context.title.text = ''
    tapped_ticks = [np.nan, np.nan]
    istart = np.nan
    scales = [0]*M.audio_nchannels
    ywavs = [np.full(1,np.nan)]*M.audio_nchannels
    xwavs = [np.full(1,np.nan)]*M.audio_nchannels
    M.spectrogram_freq = [np.full(1,np.nan)]*M.audio_nchannels
    M.spectrogram_time = [np.full(1,np.nan)]*M.audio_nchannels
    M.spectrogram_image = [np.full((1,1),np.nan)]*M.audio_nchannels
    xlabel, ylabel, tlabel = [], [], []
    xlabel_new, ylabel_new, tlabel_new = [], [], []
    left, right, top, bottom = [], [], [], []
    left_new, right_new, top_new, bottom_new = [], [], [], []

    if M.isnippet>=0:
        play.disabled=False
        video_toggle.disabled=False
        zoom_context.disabled=False
        zoom_offset.disabled=False
        zoomin.disabled=False
        zoomout.disabled=False
        reset.disabled=False
        panleft.disabled=False
        panright.disabled=False
        allleft.disabled=False
        allout.disabled=False
        allright.disabled=False
        tapped_sample = M.clustered_samples[M.isnippet]
        tapped_ticks = tapped_sample['ticks']
        M.context_midpoint_tic = np.mean(tapped_ticks, dtype=int)
        istart = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
        p_context.title.text = tapped_sample['file']
        _, wavs = spiowav.read(tapped_sample['file'], mmap=True)
        if np.ndim(wavs)==1:
            wavs = np.expand_dims(wavs, axis=1)
        M.file_nframes = np.shape(wavs)[0]
        if istart+M.context_width_tic>0 and istart<M.file_nframes:
            istart_bounded = np.maximum(0, istart)
            context_tic_adjusted = M.context_width_tic+1-(istart_bounded-istart)
            ilength = np.minimum(M.file_nframes-istart_bounded, context_tic_adjusted)

            tic2pix = M.context_width_tic / M.gui_width_pix
            context_decimate_by = round(tic2pix/M.filter_ratio_max) if \
                     tic2pix>M.filter_ratio_max else 1
            context_pix = round(M.context_width_tic / context_decimate_by)

            for ichannel in range(M.audio_nchannels):
                wavi = wavs[istart_bounded : istart_bounded+ilength, ichannel]
                if len(wavi)<M.context_width_tic+1:
                    npad = M.context_width_tic+1-len(wavi)
                    if istart<0:
                        wavi = np.concatenate((np.full((npad,),0), wavi))
                    if istart+M.context_width_tic>M.file_nframes:
                        wavi = np.concatenate((wavi, np.full((npad,),0)))

                if ichannel==0:
                    if bokeh_document: 
                        bokeh_document.add_next_tick_callback(lambda: \
                                _context_update(wavi,
                                                tapped_sample['file'],
                                                istart_bounded,
                                                ilength))

                wavi_downsampled = decimate(wavi, context_decimate_by, n=M.filter_order,
                                            ftype='iir', zero_phase=True)
                wavi_trimmed = wavi_downsampled[:context_pix]

                scales[ichannel]=np.minimum(np.iinfo(np.int16).max-1,
                                            np.max(np.abs(wavi_trimmed)))
                ywavs[ichannel]=wavi_trimmed/scales[ichannel]
                xwavs[ichannel]=[(istart+i*context_decimate_by)/M.audio_tic_rate \
                                 for i in range(len(wavi_trimmed))]
                if ichannel==0:
                    song_max = np.max(wavi_trimmed) / scales[ichannel]
                    song_max /= M.audio_nchannels
                    song_max += (M.audio_nchannels-1-2*ichannel)/M.audio_nchannels
                if ichannel==M.audio_nchannels-1:
                    song_min = np.min(wavi_trimmed) / scales[ichannel]
                    song_min /= M.audio_nchannels
                    song_min += (M.audio_nchannels-1-2*ichannel)/M.audio_nchannels

                window_length = round(M.spectrogram_length_ms/1000*M.audio_tic_rate)
                M.spectrogram_freq[ichannel], M.spectrogram_time[ichannel], M.spectrogram_image[ichannel] = \
                        spectrogram(wavi,
                                    fs=M.audio_tic_rate,
                                    window=M.spectrogram_window,
                                    nperseg=window_length,
                                    noverlap=round(window_length*M.spectrogram_overlap))

            line_red_context.data.update(x=[xwavs[0][0],xwavs[0][0]])
            p_line_red_context.visible=True

            ileft = np.searchsorted(M.clustered_starts_sorted, istart+M.context_width_tic)
            samples_to_plot = set(range(0,ileft))
            iright = np.searchsorted(M.clustered_stops, istart,
                                    sorter=M.iclustered_stops_sorted)
            samples_to_plot &= set([M.iclustered_stops_sorted[i] for i in \
                    range(iright, len(M.iclustered_stops_sorted))])

            tapped_wav_in_view = False
            for isample in samples_to_plot:
                if tapped_sample['file']!=M.clustered_samples[isample]['file']:
                    continue
                L = np.max([istart, M.clustered_samples[isample]['ticks'][0]])
                R = np.min([istart+M.context_width_tic,
                            M.clustered_samples[isample]['ticks'][1]])
                xlabel.append((L+R)/2/M.audio_tic_rate)
                tlabel.append(M.clustered_samples[isample]['kind']+'\n'+\
                              M.clustered_samples[isample]['label'])
                ylabel.append(song_max)
                left.append(L/M.audio_tic_rate)
                right.append(R/M.audio_tic_rate)
                top.append(song_max)
                bottom.append(0)
                if tapped_sample==M.clustered_samples[isample] and not np.isnan(M.xcluster):
                    quad_fuchsia_context.data.update(left=[L/M.audio_tic_rate],
                                                     right=[R/M.audio_tic_rate],
                                                     top=[song_max], bottom=[0])
                    tapped_wav_in_view = True

            if not tapped_wav_in_view:
                quad_fuchsia_context.data.update(left=[], right=[], top=[], bottom=[])

            if len(M.annotated_starts_sorted)>0:
                ileft = np.searchsorted(M.annotated_starts_sorted,
                                        istart+M.context_width_tic)
                samples_to_plot = set(range(0,ileft))
                iright = np.searchsorted(M.annotated_stops, istart,
                                         sorter=M.iannotated_stops_sorted)
                samples_to_plot &= set([M.iannotated_stops_sorted[i] for i in \
                        range(iright, len(M.iannotated_stops_sorted))])

                for isample in samples_to_plot:
                    if tapped_sample['file']!=M.annotated_samples[isample]['file']:
                        continue
                    L = np.max([istart, M.annotated_samples[isample]['ticks'][0]])
                    R = np.min([istart+M.context_width_tic,
                                M.annotated_samples[isample]['ticks'][1]])
                    xlabel_new.append((L+R)/2/M.audio_tic_rate)
                    tlabel_new.append(M.annotated_samples[isample]['label'])
                    ylabel_new.append(song_min)
                    left_new.append(L/M.audio_tic_rate)
                    right_new.append(R/M.audio_tic_rate)
                    top_new.append(0)
                    bottom_new.append(song_min)
    else:
        play.disabled=True
        video_toggle.disabled=True
        zoom_context.disabled=True
        zoom_offset.disabled=True
        zoomin.disabled=True
        zoomout.disabled=True
        reset.disabled=True
        panleft.disabled=True
        panright.disabled=True
        allleft.disabled=True
        allout.disabled=True
        allright.disabled=True
        quad_fuchsia_context.data.update(left=[], right=[], top=[], bottom=[])
        line_red_context.data.update(x=[0,0])
        p_line_red_context.visible=False
        play_callback.code = C.play_callback_code % ("", "")

    for ichannel in range(M.audio_nchannels):
        xdata = xwavs[ichannel]
        ydata = ywavs[ichannel]/M.audio_nchannels + \
                (M.audio_nchannels-1-2*ichannel) / M.audio_nchannels
        wav_source[ichannel].data.update(x=xdata, y=ydata)
        ipalette = int(np.floor(scales[ichannel] /
                                np.iinfo(np.int16).max *
                                len(snippet_palette)))
        line_glyph[ichannel].glyph.line_color = snippet_palette[ipalette]
        if not np.isnan(M.spectrogram_time[ichannel][0]):
            ilow = np.argmin(np.abs(M.spectrogram_freq[ichannel]-M.spectrogram_low_hz))
            ihigh = np.argmin(np.abs(M.spectrogram_freq[ichannel]-M.spectrogram_high_hz))
            image_glyph[ichannel].glyph.x = M.spectrogram_time[ichannel][0]
            image_glyph[ichannel].glyph.y = M.spectrogram_freq[ichannel][ilow] / M.spectrogram_freq_scale
            image_glyph[ichannel].glyph.dw = M.spectrogram_time[ichannel][-1]
            image_glyph[ichannel].glyph.dh = \
                    (M.spectrogram_freq[ichannel][ihigh] - M.spectrogram_freq[ichannel][ilow]) / \
                    M.spectrogram_freq_scale
            spectrogram_source[ichannel].data.update(image=[np.log10(M.spectrogram_image[ichannel][ilow:ihigh,:])])
        else:
            spectrogram_source[ichannel].data.update(image=[])
    quad_grey_context_old.data.update(left=left, right=right, top=top, bottom=bottom)
    quad_grey_context_new.data.update(left=left_new, right=right_new,
                                      top=top_new, bottom=bottom_new)
    label_source.data.update(x=xlabel, y=ylabel, text=tlabel)
    label_source_new.data.update(x=xlabel_new, y=ylabel_new, text=tlabel_new)

def save_update(n):
    save_indicator.label=str(n)
    if n==0:
        save_indicator.button_type="default"
    elif n<10:
        save_indicator.button_type="warning"
    else:
        save_indicator.button_type="danger"

def configuration_contents_update():
    if M.configuration_file:
        with open(M.configuration_file, 'r') as fid:
            configuration_contents.value = fid.read()

def waitfor_update():
    if len(M.waitfor_job)>0:
        waitfor.disabled=False

def model_file_update(attr, old, new):
    M.save_state_callback()
    buttons_update()

def cluster_these_layers_update():
    if os.path.isfile(os.path.join(groundtruth_folder.value,'activations.npz')):
        npzfile = np.load(os.path.join(groundtruth_folder.value,'activations.npz'),
                          allow_pickle=True)
        nlayers = len(list(filter(lambda x: x.startswith('arr_'), npzfile.files)))
        cluster_these_layers.options = [("0", "input"),
                                        *[(str(i), "hidden #"+str(i)) \
                                          for i in range(1,nlayers-1)],
                                        (str(nlayers-1), "output")]
    else:
        cluster_these_layers.options = []

def _groundtruth_update():
    wordcounts_update()
    cluster_these_layers_update()
    M.save_state_callback()
    groundtruth.button_type="default"
    groundtruth.disabled=True
    buttons_update()

def groundtruth_update():
    groundtruth.button_type="warning"
    groundtruth.disabled=True
    if bokeh_document: 
        bokeh_document.add_next_tick_callback(_groundtruth_update)

def wantedwords_update_other():
    wantedwords = [x.value for x in label_text_widgets if x.value!='']
    if 'other' not in wantedwords:
        wantedwords.append('other')
    wantedwords_string.value=str.join(',',wantedwords)

def buttons_update():
    for button in wizard_buttons:
        button.button_type="success" if button==M.wizard else "default"
    for button in action_buttons:
        button.button_type="primary" if button==M.action else "default"
        button.disabled=False if button in wizard2actions[M.wizard] else True
    if M.action in [detect,classify]:
        wavtfcsvfiles.label='wav files:'
    elif M.action==ethogram:
        wavtfcsvfiles.label='tf files:'
    elif M.action==misses:
        wavtfcsvfiles.label='csv files:'
    else:
        wavtfcsvfiles.label='wav,tf,csv files:'
    if M.action == classify:
        model.label='pb file:'
    elif M.action == ethogram:
        model.label='threshold file:'
    else:
        model.label='checkpoint file:'
    for button in parameter_buttons:
        button.disabled=False if button in action2parameterbuttons[M.action] else True
    okay=True if M.action else False
    for textinput in parameter_textinputs:
        if textinput in action2parametertextinputs[M.action]:
            if textinput==window_ms_string:
                window_ms_string.disabled=True \
                        if representation.value=='waveform' else False
            elif textinput==stride_ms_string:
                stride_ms_string.disabled=True \
                        if representation.value=='waveform' else False
            elif textinput==mel_dct_string:
                mel_dct_string.disabled=False \
                        if representation.value=='mel-cepstrum' else True
            elif textinput==pca_fraction_variance_to_retain_string:
                pca_fraction_variance_to_retain_string.disabled=False \
                        if cluster_algorithm.value[:4] in ['tSNE','UMAP'] else True
            elif textinput==tsne_perplexity_string:
                tsne_perplexity_string.disabled=False \
                        if cluster_algorithm.value.startswith('tSNE') else True
            elif textinput==tsne_exaggeration_string:
                tsne_exaggeration_string.disabled=False \
                        if cluster_algorithm.value.startswith('tSNE') else True
            elif textinput==umap_neighbors_string:
                umap_neighbors_string.disabled=False \
                        if cluster_algorithm.value.startswith('UMAP') else True
            elif textinput==umap_distance_string:
                umap_distance_string.disabled=False \
                        if cluster_algorithm.value.startswith('UMAP') else True
            else:
                textinput.disabled=False
            if textinput.disabled==False and textinput.value=='' and \
                    textinput not in [testfiles_string, restore_from_string] and \
                    (M.action!=classify or \
                     textinput not in [wantedwords_string, prevalences_string]):
                okay=False
        else:
            textinput.disabled=True
    if M.action==congruence and \
            (validationfiles_string.value!='' or testfiles_string.value!=''):
        okay=True
    doit.button_type="default"
    if okay:
        doit.disabled=False
        doit.button_type="danger"
    else:
        doit.disabled=True
        doit.button_type="default"
    cluster_these_layers.disabled = False  # https://github.com/bokeh/bokeh/issues/10507

def file_dialog_update():
    thispath = os.path.join(M.file_dialog_root,M.file_dialog_filter)
    files = glob.glob(thispath)
    uniqdirnames = set([os.path.dirname(x) for x in files])
    files = natsorted(['.', '..', *files])
    if len(uniqdirnames)==1:
        names=[os.path.basename(x) + ('/' if os.path.isdir(x) else '') for x in files]
    else:
        names=[x + ('/' if os.path.isdir(x) else '') for x in files]
    file_dialog = dict(
        names=names,
        sizes=[os.path.getsize(f) for f in files],
        dates=[datetime.fromtimestamp(os.path.getmtime(f)) for f in files],
    )
    file_dialog_source.data = file_dialog

def wordcounts_update():
    if not os.path.isdir(groundtruth_folder.value):
        return
    dfs = []
    for subdir in filter(lambda x: os.path.isdir(os.path.join(groundtruth_folder.value,x)), \
                         os.listdir(groundtruth_folder.value)):
        for csvfile in filter(lambda x: x.endswith('.csv'), \
                              os.listdir(os.path.join(groundtruth_folder.value, subdir))):
            filepath = os.path.join(groundtruth_folder.value, subdir, csvfile)
            if os.path.getsize(filepath) > 0:
                dfs.append(pd.read_csv(filepath, header=None, index_col=False))
    if dfs:
        df = pd.concat(dfs)
        M.kinds = sorted(set(df[3]))
        bkinds = {}
        table_str = '<table><tr><th></th><th>'+'</th><th>'.join(M.kinds)+'</th></tr>'
        for word in sorted(set(df[4])):
            bword = np.array(df[4]==word)
            table_str += '<tr><th>'+word+'</th>'
            for kind in M.kinds:
                if kind not in bkinds:
                    bkinds[kind] = np.array(df[3]==kind)
                table_str += '<td align="center">'+str(np.sum(np.logical_and(bkinds[kind], bword)))+'</td>'
            table_str += '</tr>'
        table_str += '<tr><th>TOTAL</th>'
        for kind in M.kinds:
            table_str += '<td align="center">'+str(np.sum(bkinds[kind]))+'</td>'
        table_str += '</tr>'
        table_str += '</table>'
        wordcounts.text = table_str
    else:
        wordcounts.text = ""

async def status_ticker_update():
    if len(M.status_ticker_queue)>0:
        newtext = []
        for k in M.status_ticker_queue.keys():
            if M.status_ticker_queue[k]=="pending":
                color = "gray"
            elif M.status_ticker_queue[k]=="running":
                color = "black"
            elif M.status_ticker_queue[k]=="succeeded":
                color = "blue"
            elif M.status_ticker_queue[k]=="failed":
                color = "red"
            newtext.append("<span style='color:"+color+"'>"+k+"</span>")
        newtext = (', ').join(newtext)
    else:
        newtext = ''
    status_ticker.text = status_ticker_pre+newtext+status_ticker_post

def init(_bokeh_document):
    global bokeh_document, cluster_dot_palette, snippet_palette, p_cluster, cluster_dots, p_cluster_dots, precomputed_dots, p_snippets, label_sources, label_sources_new, wav_sources, line_glyphs, quad_grey_snippets, dot_size_cluster, dot_alpha_cluster, circle_fuchsia_cluster, p_context, p_spectrogram, spectrogram_source, image_glyph, p_line_red_context, line_red_context, quad_grey_context_old, quad_grey_context_new, quad_grey_context_pan, quad_fuchsia_context, quad_fuchsia_snippets, wav_source, line_glyph, label_source, label_source_new, which_layer, which_species, which_word, which_nohyphen, which_kind, color_picker, circle_radius, dot_size, dot_alpha, zoom_context, zoom_offset, zoomin, zoomout, reset, panleft, panright, allleft, allout, allright, save_indicator, label_count_widgets, label_text_widgets, play, play_callback, video_toggle, video_div, undo, redo, detect, misses, configuration_file, train, leaveoneout, leaveallout, xvalidate, mistakes, activations, cluster, visualize, accuracy, freeze, classify, ethogram, compare, congruence, status_ticker, waitfor, file_dialog_source, file_dialog_source, configuration_contents, logs, logs_folder, model, model_file, wavtfcsvfiles, wavtfcsvfiles_string, groundtruth, groundtruth_folder, validationfiles, testfiles, validationfiles_string, testfiles_string, wantedwords, wantedwords_string, labeltypes, labeltypes_string, prevalences, prevalences_string, copy, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, generalize, tunehyperparameters, findnovellabels, examineerrors, testdensely, doit, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string, nsteps_string, restore_from_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string, kfold_string, activations_equalize_ratio_string, activations_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, umap_neighbors_string, umap_distance_string, cluster_algorithm, cluster_these_layers, connection_type, precision_recall_ratios_string, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, replicates_string, batch_seed_string, weights_seed_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, editconfiguration, file_dialog_string, file_dialog_table, readme_contents, wordcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs, status_ticker_update, status_ticker_pre, status_ticker_post

    bokeh_document = _bokeh_document

    M.cluster_circle_color = M.cluster_circle_color

    if '#' in M.cluster_dot_colormap:
      cluster_dot_palette = ast.literal_eval(M.cluster_dot_colormap)
    else:
      cluster_dot_palette = getattr(palettes, M.cluster_dot_colormap)

    snippet_palette = getattr(palettes, M.snippet_colormap)

    dot_size_cluster = ColumnDataSource(data=dict(ds=[M.state["dot_size"]]))
    dot_alpha_cluster = ColumnDataSource(data=dict(da=[M.state["dot_alpha"]]))

    cluster_dots = ColumnDataSource(data=dict(dx=[], dy=[], dz=[], dl=[], dc=[]))
    circle_fuchsia_cluster = ColumnDataSource(data=dict(cx=[], cy=[], cz=[], cr=[], cc=[]))
    p_cluster = ScatterNd(dx='dx', dy='dy', dz='dz', dl='dl', dc='dc',
                          dots_source=cluster_dots,
                          cx='cx', cy='cy', cz='cz', cr='cr', cc='cc',
                          circle_fuchsia_source=circle_fuchsia_cluster,
                          ds='ds',
                          dot_size_source=dot_size_cluster,
                          da='da',
                          dot_alpha_source=dot_alpha_cluster,
                          width=M.gui_width_pix//2)
    p_cluster.on_change("click_position", lambda a,o,n: C.cluster_tap_callback(n))

    precomputed_dots = None

    p_snippets = figure(plot_width=M.gui_width_pix//2, \
                        background_fill_color='#FFFFFF', toolbar_location=None)
    p_snippets.toolbar.active_drag = None
    p_snippets.grid.visible = False
    p_snippets.xaxis.visible = False
    p_snippets.yaxis.visible = False

    xdata = [(i%M.nx)*(M.snippets_gap_pix+M.snippets_pix) for i in range(M.nx*M.ny)]
    ydata = [-(i//M.nx*2-1) for i in range(M.nx*M.ny)]
    text = ['' for i in range(M.nx*M.ny)]
    label_sources = ColumnDataSource(data=dict(x=xdata, y=ydata, text=text))
    p_snippets.text('x', 'y', source=label_sources, text_font_size='6pt',
                    text_baseline='top')

    xdata = [(i%M.nx)*(M.snippets_gap_pix+M.snippets_pix) for i in range(M.nx*M.ny)]
    ydata = [-(i//M.nx*2+1) for i in range(M.nx*M.ny)]
    text_new = ['' for i in range(M.nx*M.ny)]
    label_sources_new = ColumnDataSource(data=dict(x=xdata, y=ydata, text=text_new))
    p_snippets.text('x', 'y', source=label_sources_new, text_font_size='6pt')

    wav_sources=[None]*(M.nx*M.ny)
    line_glyphs=[None]*(M.nx*M.ny)
    for ixy in range(M.nx*M.ny):
        wav_sources[ixy]=[None]*M.audio_nchannels
        line_glyphs[ixy]=[None]*M.audio_nchannels
        for ichannel in range(M.audio_nchannels):
            wav_sources[ixy][ichannel]=ColumnDataSource(data=dict(x=[], y=[]))
            line_glyphs[ixy][ichannel]=p_snippets.line('x', 'y',
                                                       source=wav_sources[ixy][ichannel])

    quad_grey_snippets = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=quad_grey_snippets,
                fill_color="lightgrey", line_color="lightgrey", level='underlay')

    p_context = figure(plot_width=M.gui_width_pix, plot_height=150,
                       background_fill_color='#FFFFFF', toolbar_location=None)
    p_context.toolbar.active_drag = None
    p_context.grid.visible = False
    p_context.xaxis.axis_label = 'time (sec)'
    p_context.yaxis.visible = False
    p_context.x_range.range_padding = 0.0
    p_context.title.text=' '

    line_red_context = ColumnDataSource(data=dict(x=[0,0], y=[-1,0]))
    p_line_red_context = p_context.line('x','y',source=line_red_context, color="red")
    p_line_red_context.visible=False

    quad_grey_context_old = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_grey_context_old,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    quad_grey_context_new = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_grey_context_new,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    quad_grey_context_pan = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_grey_context_pan,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    quad_fuchsia_context = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_fuchsia_context,
                fill_color=None, line_color="fuchsia", level='underlay')
    quad_fuchsia_snippets = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=quad_fuchsia_snippets,
                fill_color=None, line_color="fuchsia", level='underlay')

    wav_source=[None]*M.audio_nchannels
    line_glyph=[None]*M.audio_nchannels
    for ichannel in range(M.audio_nchannels):
        wav_source[ichannel] = ColumnDataSource(data=dict(x=[], y=[]))
        line_glyph[ichannel] = p_context.line('x', 'y', source=wav_source[ichannel])

    label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_context.text('x', 'y', source=label_source,
                   text_font_size='6pt', text_align='center', text_baseline='top',
                   text_line_height=0.8, level='underlay')
    label_source_new = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_context.text('x', 'y', source=label_source_new,
                   text_font_size='6pt', text_align='center', text_baseline='bottom',
                   text_line_height=0.8, level='underlay')

    p_snippets.on_event(Tap, C.snippets_tap_callback)

    p_context.on_event(DoubleTap, C.context_doubletap_callback)

    p_context.on_event(PanStart, C.context_pan_start_callback)
    p_context.on_event(Pan, C.context_pan_callback)
    p_context.on_event(PanEnd, C.context_pan_end_callback)

    p_snippets.on_event(DoubleTap, C.snippets_doubletap_callback)

    p_spectrogram = figure(plot_width=M.gui_width_pix, plot_height=150,
                        background_fill_color='#FFFFFF', toolbar_location=None)
    p_spectrogram.toolbar.active_drag = None
    p_spectrogram.x_range.range_padding = p_spectrogram.y_range.range_padding = 0
    p_spectrogram.grid.grid_line_width = 0.5
    p_spectrogram.xaxis.visible = False
    p_spectrogram.yaxis.axis_label = 'Frequency (' + M.spectrogram_units + ')'

    spectrogram_source = [None]*M.audio_nchannels
    image_glyph = [None]*M.audio_nchannels
    for ichannel in range(M.audio_nchannels):
        spectrogram_source[ichannel] = ColumnDataSource(data=dict(image=[]))
        image_glyph[ichannel] = p_spectrogram.image('image', source=spectrogram_source[ichannel],
                                                 palette=M.spectrogram_palette, level="image")

    p_spectrogram.on_event(MouseWheel, C.spectrogram_mousewheel_callback)
    p_spectrogram.on_event(PanStart, C.spectrogram_pan_start_callback)
    p_spectrogram.on_event(PanEnd, C.spectrogram_pan_end_callback)
    p_spectrogram.on_event(Tap, C.spectrogram_tap_callback)

    which_layer = Select(title="layer:")
    which_layer.on_change('value', lambda a,o,n: C.layer_callback(n))

    which_species = Select(title="species:")
    which_species.on_change('value', lambda a,o,n: C.species_callback(n))

    which_word = Select(title="word:")
    which_word.on_change('value', lambda a,o,n: C.word_callback(n))

    which_nohyphen = Select(title="no hyphen:")
    which_nohyphen.on_change('value', lambda a,o,n: C.nohyphen_callback(n))

    which_kind = Select(title="kind:")
    which_kind.on_change('value', lambda a,o,n: C.kind_callback(n))

    color_picker = ColorPicker(disabled=True)
    color_picker.on_change("color", lambda a,o,n: C.color_picker_callback(n))

    circle_radius = Slider(start=0, end=10, step=1, \
                           value=M.state["circle_radius"], \
                           title="circle radius", \
                           disabled=True)
    circle_radius.on_change("value_throttled", C.circle_radius_callback)

    dot_size = Slider(start=1, end=24, step=1, \
                      value=M.state["dot_size"], \
                      title="dot size", \
                      disabled=True)
    dot_size.on_change("value", C.dot_size_callback)

    dot_alpha = Slider(start=0.01, end=1.0, step=0.01, \
                       value=M.state["dot_alpha"], \
                       title="dot alpha", \
                       disabled=True)
    dot_alpha.on_change("value", C.dot_alpha_callback)

    cluster_update()

    zoom_context = TextInput(value=str(M.context_width_ms),
                             title="context (msec):",
                             disabled=True)
    zoom_context.on_change("value", C.zoom_context_callback)

    zoom_offset = TextInput(value=str(M.context_offset_ms),
                            title="offset (msec):",
                            disabled=True)
    zoom_offset.on_change("value", C.zoom_offset_callback)

    zoomin = Button(label='\u2191', disabled=True, width=40)
    zoomin.on_click(C.zoomin_callback)

    zoomout = Button(label='\u2193', disabled=True, width=40)
    zoomout.on_click(C.zoomout_callback)

    reset = Button(label='\u25ef', disabled=True, width=40)
    reset.on_click(C.zero_callback)

    panleft = Button(label='\u2190', disabled=True, width=40)
    panleft.on_click(C.panleft_callback)

    panright = Button(label='\u2192', disabled=True, width=40)
    panright.on_click(C.panright_callback)

    allleft = Button(label='\u21e4', disabled=True, width=40)
    allleft.on_click(C.allleft_callback)

    allout = Button(label='\u2913', disabled=True, width=40)
    allout.on_click(C.allout_callback)

    allright = Button(label='\u21e5', disabled=True, width=40)
    allright.on_click(C.allright_callback)

    save_indicator = Button(label='0', width=40)

    label_count_callbacks=[]
    label_count_widgets=[]
    label_text_callbacks=[]
    label_text_widgets=[]

    for i in range(M.nlabels):
        label_count_callbacks.append(lambda i=i: C.label_count_callback(i))
        label_count_widgets.append(Button(label='0', css_classes=['hide-label'], width=40))
        label_count_widgets[-1].on_click(label_count_callbacks[-1])

        label_text_callbacks.append(lambda a,o,n,i=i: C.label_text_callback(n,i))
        label_text_widgets.append(TextInput(value=M.state['labels'][i],
                                            css_classes=['hide-label']))
        label_text_widgets[-1].on_change("value", label_text_callbacks[-1])

    C.label_count_callback(M.ilabel)

    play = Button(label='play', disabled=True)
    play_callback = CustomJS(args=dict(line_red_context=line_red_context),
                                 code=C.play_callback_code % ("",""))
    play.js_on_event(ButtonClick, play_callback)

    video_toggle = Toggle(label='video', active=False, disabled=True)
    video_toggle.on_click(context_update)

    video_div = Div(text="""<video id="context_video"></video>""", width=0, height=0)

    undo = Button(label='undo', disabled=True)
    undo.on_click(C.undo_callback)

    redo = Button(label='redo', disabled=True)
    redo.on_click(C.redo_callback)

    detect = Button(label='detect')
    detect.on_click(lambda: C.action_callback(detect, C.detect_actuate))

    misses = Button(label='misses')
    misses.on_click(lambda: C.action_callback(misses, C.misses_actuate))

    train = Button(label='train')
    train.on_click(lambda: C.action_callback(train, C.train_actuate))

    leaveoneout = Button(label='omit one')
    leaveoneout.on_click(lambda: C.action_callback(leaveoneout,
                                                   lambda: C.leaveout_actuate(False)))

    leaveallout = Button(label='omit all')
    leaveallout.on_click(lambda: C.action_callback(leaveallout,
                                                   lambda: C.leaveout_actuate(True)))

    xvalidate = Button(label='x-validate')
    xvalidate.on_click(lambda: C.action_callback(xvalidate, C.xvalidate_actuate))

    mistakes = Button(label='mistakes')
    mistakes.on_click(lambda: C.action_callback(mistakes, C.mistakes_actuate))

    activations = Button(label='activations')
    activations.on_click(lambda: C.action_callback(activations, C.activations_actuate))

    cluster = Button(label='cluster')
    cluster.on_click(lambda: C.action_callback(cluster, C.cluster_actuate))

    visualize = Button(label='visualize')
    visualize.on_click(lambda: C.action_callback(visualize, C.visualize_actuate))

    accuracy = Button(label='accuracy')
    accuracy.on_click(lambda: C.action_callback(accuracy, C.accuracy_actuate))

    freeze = Button(label='freeze')
    freeze.on_click(lambda: C.action_callback(freeze, C.freeze_actuate))

    classify = Button(label='classify')
    classify.on_click(C.classify_callback)

    ethogram = Button(label='ethogram')
    ethogram.on_click(lambda: C.action_callback(ethogram, C.ethogram_actuate))

    compare = Button(label='compare')
    compare.on_click(lambda: C.action_callback(compare, C.compare_actuate))

    congruence = Button(label='congruence')
    congruence.on_click(lambda: C.action_callback(congruence, C.congruence_actuate))

    status_ticker_pre="<div style='overflow:auto; white-space:nowrap; width:"+str(M.gui_width_pix-126)+"px'>status: "
    status_ticker_post="</div>"
    status_ticker = Div(text=status_ticker_pre+status_ticker_post)

    file_dialog_source = ColumnDataSource(data=dict(names=[], sizes=[], dates=[]))
    file_dialog_source.selected.on_change('indices', C.file_dialog_callback)

    file_dialog_columns = [
        TableColumn(field="names", title="Name", width=M.gui_width_pix//2-50-115-10),
        TableColumn(field="sizes", title="Size", width=50, \
                    formatter=NumberFormatter(format="0 b")),
        TableColumn(field="dates", title="Date", width=115, \
                    formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
    ]
    file_dialog_table = DataTable(source=file_dialog_source, \
                                  columns=file_dialog_columns, \
                                  height=727, width=M.gui_width_pix//2-10, \
                                  index_position=None,
                                  fit_columns=False)

    waitfor = Toggle(label='wait for last job', active=False, disabled=True, width=100)
    waitfor.on_click(C.waitfor_callback)

    configuration_contents = TextAreaInput(rows=46, max_length=50000, \
                                        disabled=True, css_classes=['fixedwidth'])
    configuration_contents_update()
    configuration_contents.on_change('value', C.configuration_textarea_callback)

    logs = Button(label='logs folder:', width=110)
    logs.on_click(C.logs_callback)
    logs_folder = TextInput(value=M.state['logs'], title="", disabled=False)
    logs_folder.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    model = Button(label='checkpoint file:', width=110)
    model.on_click(C.model_callback)
    model_file = TextInput(value=M.state['model'], title="", disabled=False)
    model_file.on_change('value', model_file_update)

    wavtfcsvfiles = Button(label='wav,tf,csv files:', width=110)
    wavtfcsvfiles.on_click(C.wavtfcsvfiles_callback)
    wavtfcsvfiles_string = TextInput(value=M.state['wavtfcsvfiles'], title="", disabled=False)
    wavtfcsvfiles_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    groundtruth = Button(label='ground truth:', width=110)
    groundtruth.on_click(C.groundtruth_callback)
    groundtruth_folder = TextInput(value=M.state['groundtruth'], title="", disabled=False)
    groundtruth_folder.on_change('value', lambda a,o,n: groundtruth_update())

    validationfiles = Button(label='validation files:', width=110)
    validationfiles.on_click(C.validationfiles_callback)
    validationfiles_string = TextInput(value=M.state['validationfiles'], title="", disabled=False)
    validationfiles_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    testfiles = Button(label='test files:', width=110)
    testfiles.on_click(C.testfiles_callback)
    testfiles_string = TextInput(value=M.state['testfiles'], title="", disabled=False)
    testfiles_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    wantedwords = Button(label='wanted words:', width=110)
    wantedwords.on_click(C.wantedwords_callback)
    wantedwords_string = TextInput(value=M.state['wantedwords'], title="", disabled=False)
    wantedwords_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    labeltypes = Button(label='label types:', width=110)
    labeltypes_string = TextInput(value=M.state['labeltypes'], title="", disabled=False)
    labeltypes_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    prevalences = Button(label='prevalences:', width=110)
    prevalences.on_click(C.prevalences_callback)
    prevalences_string = TextInput(value=M.state['prevalences'], title="", disabled=False)
    prevalences_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    copy = Button(label='copy')
    copy.on_click(C.copy_callback)

    labelsounds = Button(label='label sounds')
    labelsounds.on_click(lambda: C.wizard_callback(labelsounds))

    makepredictions = Button(label='make predictions')
    makepredictions.on_click(lambda: C.wizard_callback(makepredictions))

    fixfalsepositives = Button(label='fix false positives')
    fixfalsepositives.on_click(lambda: C.wizard_callback(fixfalsepositives))

    fixfalsenegatives = Button(label='fix false negatives')
    fixfalsenegatives.on_click(lambda: C.wizard_callback(fixfalsenegatives))

    generalize = Button(label='test generalization')
    generalize.on_click(lambda: C.wizard_callback(generalize))

    tunehyperparameters = Button(label='tune h-parameters')
    tunehyperparameters.on_click(lambda: C.wizard_callback(tunehyperparameters))

    findnovellabels = Button(label='find novel labels')
    findnovellabels.on_click(lambda: C.wizard_callback(findnovellabels))

    examineerrors = Button(label='examine errors')
    examineerrors.on_click(lambda: C.wizard_callback(examineerrors))

    testdensely = Button(label='test densely')
    testdensely .on_click(lambda: C.wizard_callback(testdensely))

    doit = Button(label='do it!', disabled=True)
    doit.on_click(C.doit_callback)

    time_sigma_string = TextInput(value=M.state['time_sigma'], \
                                  title="time σ", \
                                  disabled=False)
    time_sigma_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    time_smooth_ms_string = TextInput(value=M.state['time_smooth_ms'], \
                                      title="time smooth", \
                                      disabled=False)
    time_smooth_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    frequency_n_ms_string = TextInput(value=M.state['frequency_n_ms'], \
                                      title="freq N (msec)", \
                                      disabled=False)
    frequency_n_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    frequency_nw_string = TextInput(value=M.state['frequency_nw'], \
                                    title="freq NW", \
                                    disabled=False)
    frequency_nw_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    frequency_p_string = TextInput(value=M.state['frequency_p'], \
                                   title="freq ρ", \
                                   disabled=False)
    frequency_p_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    frequency_smooth_ms_string = TextInput(value=M.state['frequency_smooth_ms'], \
                                           title="freq smooth", \
                                           disabled=False)
    frequency_smooth_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    nsteps_string = TextInput(value=M.state['nsteps'], title="# steps", disabled=False)
    nsteps_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    restore_from_string = TextInput(value=M.state['restore_from'], title="restore from", disabled=False)
    restore_from_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    save_and_validate_period_string = TextInput(value=M.state['save_and_validate_interval'], \
                                                title="validate period", \
                                                disabled=False)
    save_and_validate_period_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    validate_percentage_string = TextInput(value=M.state['validate_percentage'], \
                                           title="validate %", \
                                           disabled=False)
    validate_percentage_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    mini_batch_string = TextInput(value=M.state['mini_batch'], \
                                  title="mini-batch", \
                                  disabled=False)
    mini_batch_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    kfold_string = TextInput(value=M.state['kfold'], title="k-fold",  disabled=False)
    kfold_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    activations_equalize_ratio_string = TextInput(value=M.state['activations_equalize_ratio'], \
                                             title="equalize ratio", \
                                             disabled=False)
    activations_equalize_ratio_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    activations_max_samples_string = TextInput(value=M.state['activations_max_samples'], \
                                          title="max samples", \
                                          disabled=False)
    activations_max_samples_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    pca_fraction_variance_to_retain_string = TextInput(value=M.state['pca_fraction_variance_to_retain'], \
                                                       title="PCA fraction", \
                                                       disabled=False)
    pca_fraction_variance_to_retain_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    tsne_perplexity_string = TextInput(value=M.state['tsne_perplexity'], \
                                       title="perplexity", \
                                       disabled=False)
    tsne_perplexity_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    tsne_exaggeration_string = TextInput(value=M.state['tsne_exaggeration'], \
                                        title="exaggeration", \
                                        disabled=False)
    tsne_exaggeration_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    umap_neighbors_string = TextInput(value=M.state['umap_neighbors'], \
                                      title="neighbors", \
                                      disabled=False)
    umap_neighbors_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    umap_distance_string = TextInput(value=M.state['umap_distance'], \
                                     title="distance", \
                                     disabled=False)
    umap_distance_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    precision_recall_ratios_string = TextInput(value=M.state['precision_recall_ratios'], \
                                               title="P/Rs", \
                                               disabled=False)
    precision_recall_ratios_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    context_ms_string = TextInput(value=M.state['context_ms'], \
                                  title="context (msec)", \
                                  disabled=False)
    context_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    shiftby_ms_string = TextInput(value=M.state['shiftby_ms'], \
                                  title="shift by (msec)", \
                                  disabled=False)
    shiftby_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    representation = Select(title="representation", height=50, \
                            value=M.state['representation'], \
                            options=["waveform", "spectrogram", "mel-cepstrum"])
    representation.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    cluster_algorithm = Select(title="cluster", height=50, \
                               value=M.state['cluster_algorithm'], \
                               options=["PCA 2D", "PCA 3D", \
                                        "tSNE 2D", "tSNE 3D", \
                                        "UMAP 2D", "UMAP 3D"])
    cluster_algorithm.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    cluster_these_layers = MultiSelect(title='layers', height=108, \
                                       value=M.state['cluster_these_layers'], \
                                       options=[])
    cluster_these_layers.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))
    cluster_these_layers_update()

    connection_type = Select(title="connection", height=50, \
                             value=M.state['connection_type'], \
                             options=["plain", "residual"])
    connection_type.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    window_ms_string = TextInput(value=M.state['window_ms'], \
                                 title="window (msec)", \
                                 disabled=False)
    window_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    stride_ms_string = TextInput(value=M.state['stride_ms'], \
                                 title="stride (msec)", \
                                 disabled=False)
    stride_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    mel_dct_string = TextInput(value=M.state['mel&dct'], \
                               title="Mel & DCT", \
                               disabled=False)
    mel_dct_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    dropout_string = TextInput(value=M.state['dropout'], \
                               title="dropout", \
                               disabled=False)
    dropout_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    replicates_string = TextInput(value=M.state['replicates'], \
                                  title="replicates", \
                                  disabled=False)
    replicates_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    batch_seed_string = TextInput(value=M.state['batch_seed'], \
                                  title="batch seed", \
                                  disabled=False)
    batch_seed_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    weights_seed_string = TextInput(value=M.state['weights_seed'], \
                                    title="weights seed", \
                                    disabled=False)
    weights_seed_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    optimizer = Select(title="optimizer", height=50, \
                       value=M.state['optimizer'], \
                       options=[("sgd","SGD"), ("adam","Adam"), ("adagrad","AdaGrad"), \
                                ("rmsprop","RMSProp")])
    optimizer.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    learning_rate_string = TextInput(value=M.state['learning_rate'], \
                                     title="learning rate", \
                                     disabled=False)
    learning_rate_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    kernel_sizes_string = TextInput(value=M.state['kernel_sizes'], \
                                    title="kernels", \
                                    disabled=False)
    kernel_sizes_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    last_conv_width_string = TextInput(value=M.state['last_conv_width'], \
                                       title="last conv width", \
                                       disabled=False)
    last_conv_width_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    nfeatures_string = TextInput(value=M.state['nfeatures'], \
                                 title="# features", \
                                 disabled=False)
    nfeatures_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    dilate_after_layer_string = TextInput(value=M.state['dilate_after_layer'], \
                                          title="dilate after", \
                                          disabled=False)
    dilate_after_layer_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    stride_after_layer_string = TextInput(value=M.state['stride_after_layer'], \
                                          title="stride after", \
                                          disabled=False)
    stride_after_layer_string.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    editconfiguration = Button(label='edit', button_type="default")
    editconfiguration.on_click(C.editconfiguration_callback)

    file_dialog_string = TextInput(disabled=False)
    file_dialog_string.on_change("value", C.file_dialog_path_callback)
    file_dialog_string.value = M.state['file_dialog_string']
     
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','README.md'), 'r', encoding='utf-8') as fid:
        contents = fid.read()
    html = markdown.markdown(contents, extensions=['tables','toc'])
    readme_contents = Div(text=html, style={'overflow':'scroll','width':'600px','height':'1397px'})

    wordcounts = Div(text="")
    wordcounts_update()

    wizard_buttons = set([
        labelsounds,
        makepredictions,
        fixfalsepositives,
        fixfalsenegatives,
        generalize,
        tunehyperparameters,
        findnovellabels,
        examineerrors,
        testdensely])

    action_buttons = set([
        detect,
        train,
        leaveoneout,
        leaveallout,
        xvalidate,
        mistakes,
        activations,
        cluster,
        visualize,
        accuracy,
        freeze,
        classify,
        ethogram,
        misses,
        compare,
        congruence])

    parameter_buttons = set([
        logs,
        model,
        wavtfcsvfiles,
        groundtruth,
        validationfiles,
        testfiles,
        wantedwords,
        labeltypes,
        prevalences])

    parameter_textinputs = set([
        logs_folder,
        model_file,
        wavtfcsvfiles_string,
        groundtruth_folder,
        validationfiles_string,
        testfiles_string,
        wantedwords_string,
        labeltypes_string,
        prevalences_string,

        time_sigma_string,
        time_smooth_ms_string,
        frequency_n_ms_string,
        frequency_nw_string,
        frequency_p_string,
        frequency_smooth_ms_string,
        nsteps_string,
        restore_from_string,
        save_and_validate_period_string,
        validate_percentage_string,
        mini_batch_string,
        kfold_string,
        activations_equalize_ratio_string,
        activations_max_samples_string,
        pca_fraction_variance_to_retain_string,
        tsne_perplexity_string,
        tsne_exaggeration_string,
        umap_neighbors_string,
        umap_distance_string,
        cluster_algorithm,
        cluster_these_layers,
        connection_type,
        precision_recall_ratios_string,
        context_ms_string,
        shiftby_ms_string,
        representation,
        window_ms_string,
        stride_ms_string,
        mel_dct_string,
        dropout_string,
        replicates_string,
        batch_seed_string,
        weights_seed_string,
        optimizer,
        learning_rate_string,
        kernel_sizes_string,
        last_conv_width_string,
        nfeatures_string,
        dilate_after_layer_string,
        stride_after_layer_string])

    wizard2actions = {
            labelsounds: [detect,train,activations,cluster,visualize],
            makepredictions: [train, accuracy, freeze, classify, ethogram],
            fixfalsepositives: [activations, cluster, visualize],
            fixfalsenegatives: [detect, misses, activations, cluster, visualize],
            generalize: [leaveoneout, leaveallout, accuracy],
            tunehyperparameters: [xvalidate, accuracy, compare],
            findnovellabels: [detect, train, activations, cluster, visualize],
            examineerrors: [detect, mistakes, activations, cluster, visualize],
            testdensely: [detect, activations, cluster, visualize, classify, ethogram, congruence],
            None: action_buttons }

    action2parameterbuttons = {
            detect: [wavtfcsvfiles],
            train: [logs, groundtruth, wantedwords, testfiles, labeltypes],
            leaveoneout: [logs, groundtruth, validationfiles, testfiles, wantedwords, labeltypes],
            leaveallout: [logs, groundtruth, validationfiles, testfiles, wantedwords, labeltypes],
            xvalidate: [logs, groundtruth, testfiles, wantedwords, labeltypes],
            mistakes: [groundtruth],
            activations: [logs, model, groundtruth, wantedwords, labeltypes],
            cluster: [groundtruth],
            visualize: [groundtruth],
            accuracy: [logs],
            freeze: [logs, model],
            classify: [logs, model, wavtfcsvfiles, wantedwords, prevalences],
            ethogram: [model, wavtfcsvfiles],
            misses: [wavtfcsvfiles],
            compare: [logs],
            congruence: [groundtruth, validationfiles, testfiles],
            None: parameter_buttons }

    action2parametertextinputs = {
            detect: [wavtfcsvfiles_string, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string],
            train: [context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, replicates_string, batch_seed_string, weights_seed_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string],
            leaveoneout: [context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, batch_seed_string, weights_seed_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, validationfiles_string, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, mini_batch_string],
            leaveallout: [context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, batch_seed_string, weights_seed_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, validationfiles_string, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, mini_batch_string],
            xvalidate: [context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, batch_seed_string, weights_seed_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, mini_batch_string, kfold_string],
            mistakes: [groundtruth_folder],
            activations: [context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, model_file, groundtruth_folder, wantedwords_string, labeltypes_string, activations_equalize_ratio_string, activations_max_samples_string, mini_batch_string],
            cluster: [groundtruth_folder, cluster_algorithm, cluster_these_layers, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, umap_neighbors_string, umap_distance_string],
            visualize: [groundtruth_folder],
            accuracy: [logs_folder, precision_recall_ratios_string],
            freeze: [context_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, model_file],
            classify: [context_ms_string, shiftby_ms_string, representation, stride_ms_string, logs_folder, model_file, wavtfcsvfiles_string, wantedwords_string, prevalences_string],
            ethogram: [model_file, wavtfcsvfiles_string],
            misses: [wavtfcsvfiles_string],
            compare: [logs_folder],
            congruence: [groundtruth_folder, validationfiles_string, testfiles_string],
            None: parameter_textinputs }
