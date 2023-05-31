import os
import sys
from bokeh.models.widgets import RadioButtonGroup, TextInput, Button, Div, DateFormatter, TextAreaInput, Select, NumberFormatter, Slider, Toggle, ColorPicker, MultiSelect, Paragraph
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models import ColumnDataSource, TableColumn, DataTable, LayoutDOM, Span
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

import av
from bokeh import palettes
from itertools import cycle, product
import ast
from bokeh.core.properties import Instance, String, List, Float
from bokeh.util.compiler import TypeScript
import asyncio
from collections import OrderedDict

bokehlog = logging.getLogger("songexplorer") 

import model as M
import controller as C

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

    cx = String()
    cy = String()
    cz = String()
    cr = String()
    cc = String()

    dx = String()
    dy = String()
    dz = String()
    dl = String()
    dc = String()
    ds = String()
    da = String()

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
    M.clustered_sounds = npzfile['sounds']
    M.clustered_activations = npzfile['activations_clustered']

    M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_sounds]
    isort = np.argsort(M.clustered_starts_sorted)
    for i in range(len(M.clustered_activations)):
        if M.clustered_activations[i] is not None:
            layer0 = i
            M.clustered_activations[i] = M.clustered_activations[i][isort,:]
    M.clustered_sounds = [M.clustered_sounds[x] for x in isort]
    M.clustered_starts_sorted = [M.clustered_starts_sorted[x] for x in isort]

    M.clustered_stops = [x['ticks'][1] for x in M.clustered_sounds]
    M.iclustered_stops_sorted = np.argsort(M.clustered_stops)

    recordings.options = sorted(list(set([x['file'] for x in M.clustered_sounds])))
    for recording in recordings.options:
        M.clustered_recording2firstsound[recording] = \
              next(filter(lambda x: x[1]['file']==recording, enumerate(M.clustered_sounds)))[0]
    recordings.options = [""] + recordings.options

    cluster_isnotnan = [not np.isnan(x[0]) and not np.isnan(x[1]) \
                        for x in M.clustered_activations[layer0]]

    M.nlayers = len(M.clustered_activations)
    M.ndcluster = np.shape(M.clustered_activations[layer0])[1]
    cluster_dots.data.update(dx=[], dy=[], dz=[], dl=[], dc=[])
    cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])

    M.layers = ["input"]+["hidden #"+str(i) for i in range(1,M.nlayers-1)]+["output"]
    M.species = set([x['label'].split('-')[0]+'-' \
                     for x in M.clustered_sounds if '-' in x['label']])
    M.species |= set([''])
    M.species = natsorted(list(M.species))
    M.words = set(['-'+x['label'].split('-')[1] \
                   for x in M.clustered_sounds if '-' in x['label']])
    M.words |= set([''])
    M.words = natsorted(list(M.words))
    M.nohyphens = set([x['label'] for x in M.clustered_sounds if '-' not in x['label']])
    M.nohyphens |= set([''])
    M.nohyphens = natsorted(list(M.nohyphens))
    M.kinds = set([x['kind'] for x in M.clustered_sounds])
    M.kinds |= set([''])
    M.kinds = natsorted(list(M.kinds))

    if newcolors:
        allcombos = [x[0][:-1]+x[1] for x in product(M.species[1:], M.words[1:])]
        M.cluster_dot_colors = { l:c for l,c in zip(allcombos+ M.nohyphens[1:],
                                                    cycle(cluster_dot_palette)) }
    M.clustered_labels = set([x['label'] for x in M.clustered_sounds])

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
                        M.ilayer=ilayer
                        bidx = np.logical_and([specie in x['label'] and \
                                               word in x['label'] and \
                                               (nohyphen=="" or nohyphen==x['label']) and \
                                               (kind=="" or kind==x['kind']) \
                                               for x in M.clustered_sounds], \
                                               cluster_isnotnan)
                        if not any(bidx):
                            continue
                        if inohyphen>0:
                            colors = [M.cluster_dot_colors[nohyphen] for b in bidx if b]
                        else:
                            colors = [M.cluster_dot_colors[x['label']] \
                                      if x['label'] in M.cluster_dot_colors else "black" \
                                      for x,b in zip(M.clustered_sounds,bidx) if b]
                        data = {'x': M.clustered_activations[ilayer][bidx,0], \
                                'y': M.clustered_activations[ilayer][bidx,1], \
                                'l': [x['label'] for x,b in zip(M.clustered_sounds,bidx) if b], \
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

    M.ispecies = M.iword = M.inohyphen = M.ikind = 0

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

def within_an_annotation(sound):
    if len(M.annotated_starts_sorted)>0:
        ifrom = np.searchsorted(M.annotated_starts_sorted, sound['ticks'][0],
                                side='right') - 1
        if 0 <= ifrom and ifrom < len(M.annotated_starts_sorted) and \
                    M.annotated_sounds[ifrom]['ticks'][1] >= sound['ticks'][1]:
            return ifrom
    return -1

def snippets_update(redraw_wavs):
    if len(M.species)==0:
        return
    if M.isnippet>0 and not np.isnan(M.xcluster) and not np.isnan(M.ycluster) \
                and (M.ndcluster==2 or not np.isnan(M.zcluster)):
        snippets_quad_fuchsia.data.update(
                left=[M.xsnippet*(M.snippets_gap_pix+M.snippets_pix)],
                right=[(M.xsnippet+1)*(M.snippets_gap_pix+M.snippets_pix)-
                       M.snippets_gap_pix],
                top=[-M.ysnippet*snippets_dy+1],
                bottom=[-M.ysnippet*snippets_dy - 1 - snippets_both])
    else:
        snippets_quad_fuchsia.data.update(left=[], right=[], top=[], bottom=[])

    isubset = np.where([M.species[M.ispecies] in x['label'] and
                        M.words[M.iword] in x['label'] and
                        (M.nohyphens[M.inohyphen]=="" or \
                         M.nohyphens[M.inohyphen]==x['label']) and
                        (M.kinds[M.ikind]=="" or \
                         M.kinds[M.ikind]==x['kind']) for x in M.clustered_sounds])[0]
    origin = [M.xcluster,M.ycluster]
    if M.ndcluster==3:
        origin.append(M.zcluster)
    distance = [] if M.clustered_activations is None or M.clustered_activations[M.ilayer] is None else \
               np.linalg.norm(M.clustered_activations[M.ilayer][isubset,:] - origin, \
                              axis=1)
    isort = np.argsort(distance)
    ywavs, scales = [], []
    gram_freqs, gram_times, gram_images, ilows, ihighs  = [], [], [], [], []
    labels_clustered, labels_annotated = [], []
    warned_already=False
    for isnippet in range(M.snippets_nx*M.snippets_ny):
        if isnippet<len(distance) and \
                    distance[isort[isnippet]] < float(M.state["circle_radius"]):
            M.nearest_sounds[isnippet] = isubset[isort[isnippet]]
            thissound = M.clustered_sounds[M.nearest_sounds[isnippet]]
            labels_clustered.append(thissound['label'])
            iannotated = within_an_annotation(thissound)
            if iannotated == -1:
                labels_annotated.append('')
            else:
                labels_annotated.append(M.annotated_sounds[iannotated]['label'])
            midpoint = np.mean(thissound['ticks'], dtype=int)
            if redraw_wavs:
                start_tic = max(0, midpoint-M.snippets_tic//2)
                _, wavs = M.audio_read(os.path.join(groundtruth_folder.value, thissound['file']),
                                       start_tic, start_tic+M.snippets_tic)
                ntics_gotten = np.shape(wavs)[0]
                left_pad = max(0, M.snippets_pix-ntics_gotten if start_tic==0 else 0)
                right_pad = max(0, M.snippets_pix-ntics_gotten if start_tic>0 else 0)
                ywav = [[]]*len(M.snippets_waveform)
                scale = [[]]*len(M.snippets_waveform)
                gram_freq = [[]]*len(M.snippets_spectrogram)
                gram_time = [[]]*len(M.snippets_spectrogram)
                gram_image = [[]]*len(M.snippets_spectrogram)
                ilow = [[]]*len(M.snippets_spectrogram)
                ihigh = [[]]*len(M.snippets_spectrogram)
                for ichannel in range(M.audio_nchannels):
                    wavi = wavs[:, ichannel]
                    if ichannel+1 in M.snippets_waveform:
                        idx = M.snippets_waveform.index(ichannel+1)
                        wavi_downsampled = wavi[0::M.snippets_decimate_by].astype(float)
                        np.pad(wavi_downsampled, ((left_pad, right_pad),),
                               'constant', constant_values=(np.nan,))
                        wavi_trimmed = wavi_downsampled[:M.snippets_pix]
                        scale[idx]=np.minimum(np.iinfo(np.int16).max-1,
                                                   np.max(np.abs(wavi_trimmed)))
                        ywav[idx]=wavi_trimmed/scale[idx]
                    if ichannel+1 in M.snippets_spectrogram:
                        idx = M.snippets_spectrogram.index(ichannel+1)
                        window_length = int(round(M.spectrogram_length_ms[ichannel]/1000*M.audio_tic_rate))
                        if window_length > len(wavi):
                            window_length = len(wavi)
                            if not warned_already:
                                bokehlog.info("WARNING: spectrogram window length is greater than snippet duration")
                                warned_already=True
                        gram_freq[idx], gram_time[idx], gram_image[idx] = \
                                spectrogram(wavi,
                                            fs=M.audio_tic_rate,
                                            window=M.spectrogram_window,
                                            nperseg=window_length,
                                            noverlap=round(window_length*M.spectrogram_overlap))
                        ilow[idx] = np.argmin(np.abs(gram_freq[idx] - \
                                                     M.spectrogram_low_hz[ichannel]))
                        ihigh[idx] = np.argmin(np.abs(gram_freq[idx] - \
                                                      M.spectrogram_high_hz[ichannel]))
                ywavs.append(ywav)
                scales.append(scale)
                gram_freqs.append(gram_freq)
                gram_times.append(gram_time)
                gram_images.append(gram_image)
                ilows.append(ilow)
                ihighs.append(ihigh)
        else:
            M.nearest_sounds[isnippet] = -1
            labels_clustered.append('')
            labels_annotated.append('')
            scales.append([0]*len(M.snippets_waveform))
            ywavs.append([np.full(M.snippets_pix,np.nan)]*len(M.snippets_waveform))
            gram_images.append([])
    snippets_label_sources_clustered.data.update(text=labels_clustered)
    snippets_label_sources_annotated.data.update(text=labels_annotated)
    left_clustered, right_clustered, top_clustered, bottom_clustered = [], [], [], []
    for isnippet in range(M.snippets_nx*M.snippets_ny):
        ix, iy = isnippet%M.snippets_nx, isnippet//M.snippets_nx
        if redraw_wavs:
            xdata = range(ix*(M.snippets_gap_pix+M.snippets_pix),
                          (ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            for ichannel in range(M.audio_nchannels):
                if ichannel+1 in M.snippets_waveform:
                    idx = M.snippets_waveform.index(ichannel+1)
                    ydata = -iy*snippets_dy + \
                            (len(M.snippets_waveform)-1-2*idx)/len(M.snippets_waveform) + \
                            ywavs[isnippet][idx]/len(M.snippets_waveform)
                    snippets_wave_sources[isnippet][idx].data.update(x=xdata, y=ydata)
                    ipalette = int(np.floor(scales[isnippet][idx] /
                                            np.iinfo(np.int16).max *
                                            len(snippet_palette)))
                    snippets_wave_glyphs[isnippet][idx].glyph.line_color = snippet_palette[ipalette]
                if ichannel+1 in M.snippets_spectrogram:
                    idx = M.snippets_spectrogram.index(ichannel+1)
                    if gram_images[isnippet]:
                        snippets_gram_glyphs[isnippet][idx].glyph.x = xdata[0]
                        snippets_gram_glyphs[isnippet][idx].glyph.y = \
                                -iy*snippets_dy - 1 - snippets_both \
                                +len(M.snippets_spectrogram) - 1 - idx
                        snippets_gram_glyphs[isnippet][idx].glyph.dw = xdata[-1] - xdata[0] + \
                                                                       xdata[1] - xdata[0]
                        snippets_gram_glyphs[isnippet][idx].glyph.dh = 2/len(M.snippets_spectrogram)
                        snippets_gram_sources[isnippet][idx].data.update(image=[np.log10(1e-15+ \
                                gram_images[isnippet][idx][ilows[isnippet][idx]:1+ihighs[isnippet][idx],:])])
                    else:
                        snippets_gram_sources[isnippet][idx].data.update(image=[])
        if labels_annotated[isnippet]!='':
            left_clustered.append(ix*(M.snippets_gap_pix+M.snippets_pix))
            right_clustered.append((ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            top_clustered.append(-iy*snippets_dy+1)
            bottom_clustered.append(-iy*snippets_dy - 1 - snippets_both)
    snippets_quad_grey.data.update(left=left_clustered, right=right_clustered,
                                   top=top_clustered, bottom=bottom_clustered)

def nparray2base64wav(data, tic_rate):
    fid=io.BytesIO()
    wav=wave.open(fid, "w")
    wav.setframerate(tic_rate)
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.writeframes(data.tobytes())
    wav.close()
    fid.seek(0)
    ret_val = base64.b64encode(fid.read()).decode('utf-8')
    fid.close()
    return ret_val

def nparray2base64mp4(filename, start_sec, stop_sec):
    frame_rate, video_data = M.video_read(filename)

    start_frame = np.ceil(start_sec * frame_rate).astype(int)
    stop_frame = np.floor(stop_sec * frame_rate).astype(int)

    fid=io.BytesIO()
    container = av.open(fid, mode='w', format='mp4')

    stream = container.add_stream('h264', rate=frame_rate)
    stream.width = video_data.shape[1]
    stream.height = video_data.shape[2]
    stream.pix_fmt = 'yuv420p'

    for iframe in range(start_frame, stop_frame):
        frame = av.VideoFrame.from_ndarray(np.array(video_data[iframe]), format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    fid.seek(0)
    ret_val = base64.b64encode(fid.read()).decode('utf-8')
    fid.close()
    return ret_val, stream.height, stream.width, frame_rate

# _context_update() might be able to be folded back in to context_update() with bokeh 2.0
# ditto for _doit_callback() and _groundtruth_update()
# see https://discourse.bokeh.org/t/bokeh-server-is-it-possible-to-push-updates-to-js-in-the-middle-of-a-python-callback/3455/4

def reset_video():
    load_multimedia_callback.code = C.load_multimedia_callback_code % ("", "")
    load_multimedia.text = ""

def ___context_update(start_sec, stop_sec, frame_rate, tapped_sound):
    if video_toggle.active:
        video_slider.visible = True
        video_slider.start = np.ceil(start_sec * frame_rate) / frame_rate
        video_slider.end = np.floor(stop_sec * frame_rate) / frame_rate
        video_slider.step = 1/frame_rate
        midpoint_tics = (tapped_sound['ticks'][0] + tapped_sound['ticks'][1]) / 2
        midpoint_frames = np.round(midpoint_tics / M.audio_tic_rate * frame_rate) / frame_rate
        video_slider.value = np.clip(midpoint_frames, video_slider.start, video_slider.end)
        waveform_span_red.location = video_slider.value
        spectrogram_span_red.location = video_slider.value
        probability_span_red.location = video_slider.value
    else:
        video_slider.value = 0
        video_slider.visible = False
        waveform_span_red.location = start_sec
        spectrogram_span_red.location = start_sec
        probability_span_red.location = start_sec

def __context_update(wavi, tapped_sound, istart_bounded, ilength):
    start_sec = istart_bounded / M.audio_tic_rate
    stop_sec = (istart_bounded+ilength) / M.audio_tic_rate

    if video_toggle.active:
        video_toggle.button_type="default"
        sound_basename=os.path.basename(tapped_sound['file'])
        sound_dirname=os.path.join(groundtruth_folder.value, os.path.dirname(tapped_sound['file']))
        vidfile = M.video_findfile(sound_dirname, sound_basename)
        if not vidfile:
            bokehlog.info("ERROR: video file corresponding to "+tapped_sound['file']+" not found")
            return
        base64vid, height, width, frame_rate = nparray2base64mp4(os.path.join(sound_dirname,
                                                                              vidfile),
                                                                 start_sec, stop_sec)
        labelcounts.style = {'overflow-y':'hidden', 'overflow-x':'scroll',
                             'width':str(max(100,M.gui_width_pix-450-width))+'px'}
        video_div.style = {'width':str(width)+'px', 'height':str(height)+'px'}
    else:
        frame_rate = 0
        base64vid = ""
        labelcounts.style = {'overflow-y':'hidden', 'overflow-x':'scroll',
                            'width':str(M.gui_width_pix-450-1)+'px'}
        video_div.style = {'width':'1px', 'height':'1px'}

    base64wav = nparray2base64wav(wavi, M.audio_tic_rate)
    load_multimedia_callback.code = C.load_multimedia_callback_code % (base64wav, base64vid)
    load_multimedia.text = str(np.random.random())

    bokeh_document.add_next_tick_callback(lambda: \
            ___context_update(start_sec, stop_sec, frame_rate, tapped_sound))

def _context_update(wavi, tapped_sound, istart_bounded, ilength):
    if video_toggle.active:
        video_toggle.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: \
            __context_update(wavi, tapped_sound, istart_bounded, ilength))

def context_update():
    tapped_ticks = [np.nan, np.nan]
    istart = np.nan
    scales = [0]*len(M.context_waveform)
    ywav = [np.full(1,np.nan)]*len(M.context_waveform)
    xwav = [np.full(1,np.nan)]*len(M.context_waveform)
    gram_freq = [np.full(1,np.nan)]*len(M.context_spectrogram)
    gram_time = [np.full(1,np.nan)]*len(M.context_spectrogram)
    gram_image = [np.full((1,1),np.nan)]*len(M.context_spectrogram)
    yprob = [np.full(1,np.nan)]*len(M.clustered_labels)
    xprob = [np.full(1,np.nan)]*len(M.clustered_labels)
    ilow = [0]*len(M.context_spectrogram)
    ihigh = [1]*len(M.context_spectrogram)
    xlabel_clustered, tlabel_clustered = [], []
    xlabel_annotated, tlabel_annotated = [], []
    left_clustered, right_clustered = [], []
    left_annotated, right_annotated = [], []

    if M.isnippet>=0:
        play.disabled=False
        video_toggle.disabled=False
        zoom_width.disabled=False
        zoom_offset.disabled=False
        zoomin.disabled=False
        zoomout.disabled=False
        reset.disabled=False
        panleft.disabled=False
        panright.disabled=False
        allleft.disabled=False
        allout.disabled=False
        allright.disabled=False
        firstlabel.disabled=False
        nextlabel.disabled=False
        prevlabel.disabled=False
        lastlabel.disabled=False
        tapped_sound = M.clustered_sounds[M.isnippet]
        tapped_ticks = tapped_sound['ticks']
        M.context_midpoint_tic = np.mean(tapped_ticks, dtype=int)
        istart = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
        if recordings.value != tapped_sound['file']:
            M.user_changed_recording=False
        recordings.value = tapped_sound['file']
        _, wavs = M.audio_read(os.path.join(groundtruth_folder.value, tapped_sound['file']))
        M.file_nframes = np.shape(wavs)[0]
        probs = [None]*len(M.clustered_labels)
        for ilabel,label in enumerate(M.clustered_labels):
            prob_wavfile = os.path.join(groundtruth_folder.value,
                                        tapped_sound['file'][:-4]+'-'+label+'.wav')
            if os.path.isfile(prob_wavfile):
                prob_tic_rate, probs[ilabel] = spiowav.read(prob_wavfile, mmap=True)
        if istart+M.context_width_tic>0 and istart<M.file_nframes:
            istart_bounded = np.maximum(0, istart)
            context_tic_adjusted = M.context_width_tic+1-(istart_bounded-istart)
            ilength = np.minimum(M.file_nframes-istart_bounded, context_tic_adjusted)

            tic2pix = M.context_width_tic / M.gui_width_pix
            context_decimate_by = round(tic2pix/M.tic2pix_max) if tic2pix>M.tic2pix_max else 1
            context_pix = round(M.context_width_tic / context_decimate_by)

            if any([isinstance(x, np.ndarray) for x in probs]):
                tic_rate_ratio = prob_tic_rate / M.audio_tic_rate
                tic2pix = round(M.context_width_tic*tic_rate_ratio / M.gui_width_pix)
                prob_decimate_by = round(tic2pix/M.tic2pix_max) if tic2pix>M.tic2pix_max else 1
                prob_pix = round(M.context_width_tic*tic_rate_ratio / prob_decimate_by)

            for ichannel in range(M.audio_nchannels):
                wavi = wavs[istart_bounded : istart_bounded+ilength, ichannel]
                if len(wavi)<M.context_width_tic+1:
                    npad = M.context_width_tic+1-len(wavi)
                    if istart<0:
                        wavi = np.concatenate((np.full((npad,),0), wavi))
                    if istart+M.context_width_tic>M.file_nframes:
                        wavi = np.concatenate((wavi, np.full((npad,),0)))
                M.context_data[ichannel] = wavi
                M.context_data_istart = istart_bounded

                if ichannel==0:
                    if bokeh_document: 
                        bokeh_document.add_next_tick_callback(lambda: \
                                _context_update(wavi,
                                                tapped_sound,
                                                istart_bounded,
                                                ilength))

                xwav0 = istart_bounded/M.audio_tic_rate
                if ichannel+1 in M.context_waveform:
                    idx = M.context_waveform.index(ichannel+1)
                    wavi_downsampled = wavi[0::context_decimate_by]
                    wavi_trimmed = wavi_downsampled[:context_pix]

                    scales[idx]=np.minimum(np.iinfo(np.int16).max-1,
                                                np.max(np.abs(wavi_trimmed)))
                    wavi_scaled = wavi_trimmed/scales[idx]
                    icliplow = np.where((wavi_scaled < M.context_waveform_low[ichannel]))[0]
                    icliphigh = np.where((wavi_scaled > M.context_waveform_high[ichannel]))[0]
                    wavi_zoomed = np.copy(wavi_scaled)
                    wavi_zoomed[icliplow] = M.context_waveform_low[ichannel]
                    wavi_zoomed[icliphigh] = M.context_waveform_high[ichannel]
                    ywav[idx] = (wavi_zoomed - M.context_waveform_low[ichannel]) / \
                                (M.context_waveform_high[ichannel] - M.context_waveform_low[ichannel]) \
                                * 2 - 1
                    xwav[idx]=[(istart+i*context_decimate_by)/M.audio_tic_rate \
                               for i in range(len(wavi_trimmed))]

                if ichannel+1 in M.context_spectrogram:
                    idx = M.context_spectrogram.index(ichannel+1)
                    window_length = int(round(M.spectrogram_length_ms[ichannel]/1000*M.audio_tic_rate))
                    gram_freq[idx], gram_time[idx], gram_image[idx] = \
                            spectrogram(wavi,
                                        fs=M.audio_tic_rate,
                                        window=M.spectrogram_window,
                                        nperseg=window_length,
                                        noverlap=round(window_length*M.spectrogram_overlap))
                    ilow[idx] = np.argmin(np.abs(gram_freq[idx] - \
                                                 M.spectrogram_low_hz[ichannel]))
                    ihigh[idx] = np.argmin(np.abs(gram_freq[idx] - \
                                                  M.spectrogram_high_hz[ichannel]))

            for ilabel in range(len(M.clustered_labels)):
                if not isinstance(probs[ilabel], np.ndarray):  continue
                prob_istart = int(np.rint(istart_bounded*tic_rate_ratio))
                prob_istop = int(np.rint((istart_bounded+ilength)*tic_rate_ratio))
                probi = probs[ilabel][prob_istart : prob_istop : prob_decimate_by]
                if len(probi)<round(M.context_width_tic*tic_rate_ratio)+1:
                    npad = round(M.context_width_tic*tic_rate_ratio)+1-len(probi)
                    if istart<0:
                        probi = np.concatenate((np.full((npad,),0), probi))
                    if istart+round(M.context_width_tic*tic_rate_ratio)>M.file_nframes:
                        probi = np.concatenate((probi, np.full((npad,),0)))
                probi_trimmed = probi[:prob_pix]
                yprob[ilabel] = probi_trimmed / np.iinfo(np.int16).max
                xprob[ilabel]=[(prob_istart+i*prob_decimate_by)/prob_tic_rate \
                                 for i in range(len(probi_trimmed))]

            if M.context_spectrogram:
                p_spectrogram.yaxis.formatter = FuncTickFormatter(
                    args=dict(low_hz=[gram_freq[i][x] / M.context_spectrogram_freq_scale \
                                      for i,x in enumerate(ilow)],
                              high_hz=[gram_freq[i][x] / M.context_spectrogram_freq_scale \
                                       for i,x in enumerate(ihigh)]),
                    code="""
                         if (tick==0) {
                             return low_hz[low_hz.length-1] }
                         else if (tick == high_hz.length) {
                             return high_hz[0] }
                         else {
                             return low_hz[low_hz.length-tick-1] + "," + high_hz[high_hz.length-tick] }
                         """)

            ileft = np.searchsorted(M.clustered_starts_sorted, istart+M.context_width_tic)
            sounds_to_plot = set(range(0,ileft))
            iright = np.searchsorted(M.clustered_stops, istart,
                                    sorter=M.iclustered_stops_sorted)
            sounds_to_plot &= set([M.iclustered_stops_sorted[i] for i in \
                    range(iright, len(M.iclustered_stops_sorted))])

            tapped_wav_in_view = False
            M.remaining_isounds = []
            for isound in sounds_to_plot:
                if tapped_sound['file']!=M.clustered_sounds[isound]['file']:
                    continue
                L = np.max([istart, M.clustered_sounds[isound]['ticks'][0]])
                R = np.min([istart+M.context_width_tic,
                            M.clustered_sounds[isound]['ticks'][1]])
                if L>istart and R<istart+M.context_width_tic and \
                        M.clustered_sounds[isound]['label'] in M.state['labels']:
                    M.remaining_isounds.append(isound)
                xlabel_clustered.append((L+R)/2/M.audio_tic_rate)
                tlabel_clustered.append(M.clustered_sounds[isound]['kind']+'\n'+\
                              M.clustered_sounds[isound]['label'])
                left_clustered.append(L/M.audio_tic_rate)
                right_clustered.append(R/M.audio_tic_rate)
                if tapped_sound==M.clustered_sounds[isound] and not np.isnan(M.xcluster):
                    if M.context_waveform:
                        waveform_quad_fuchsia.data.update(left=[L/M.audio_tic_rate],
                                                          right=[R/M.audio_tic_rate],
                                                          top=[1],
                                                          bottom=[0])
                    if M.context_spectrogram:
                        spectrogram_quad_fuchsia.data.update(left=[L/M.audio_tic_rate],
                                                             right=[R/M.audio_tic_rate],
                                                             top=[len(M.context_spectrogram)],
                                                             bottom=[len(M.context_spectrogram)/2])
                    tapped_wav_in_view = True

            M.remaining_isounds = [i for i in M.remaining_isounds \
                                   if all([i==j or \
                                           M.clustered_sounds[i]['ticks'][0] > M.clustered_sounds[j]['ticks'][1] or \
                                           M.clustered_sounds[i]['ticks'][1] < M.clustered_sounds[j]['ticks'][0] \
                                           for j in M.remaining_isounds])]

            if M.context_waveform:
                waveform_span_red.visible=True
            if M.context_spectrogram:
                spectrogram_span_red.visible=True
            probability_span_red.visible=True

            if not tapped_wav_in_view:
                if M.context_waveform:
                    waveform_quad_fuchsia.data.update(left=[], right=[], top=[], bottom=[])
                if M.context_spectrogram:
                    spectrogram_quad_fuchsia.data.update(left=[], right=[], top=[], bottom=[])

            if len(M.annotated_starts_sorted)>0:
                ileft = np.searchsorted(M.annotated_starts_sorted,
                                        istart+M.context_width_tic)
                sounds_to_plot = set(range(0,ileft))
                iright = np.searchsorted(M.annotated_stops, istart,
                                         sorter=M.iannotated_stops_sorted)
                sounds_to_plot &= set([M.iannotated_stops_sorted[i] for i in \
                        range(iright, len(M.iannotated_stops_sorted))])

                for isound in sounds_to_plot:
                    if tapped_sound['file']!=M.annotated_sounds[isound]['file']:
                        continue

                    M.remaining_isounds = [i for i in M.remaining_isounds \
                                           if M.annotated_sounds[isound]['ticks'][0] > M.clustered_sounds[i]['ticks'][1] or \
                                              M.annotated_sounds[isound]['ticks'][1] < M.clustered_sounds[i]['ticks'][0]]
                        
                    L = np.max([istart, M.annotated_sounds[isound]['ticks'][0]])
                    R = np.min([istart+M.context_width_tic,
                                M.annotated_sounds[isound]['ticks'][1]])
                    xlabel_annotated.append((L+R)/2/M.audio_tic_rate)
                    tlabel_annotated.append(M.annotated_sounds[isound]['label'])
                    left_annotated.append(L/M.audio_tic_rate)
                    right_annotated.append(R/M.audio_tic_rate)
    else:
        play.disabled=True
        video_toggle.disabled=True
        zoom_width.disabled=True
        zoom_offset.disabled=True
        zoomin.disabled=True
        zoomout.disabled=True
        reset.disabled=True
        panleft.disabled=True
        panright.disabled=True
        allleft.disabled=True
        allout.disabled=True
        allright.disabled=True
        firstlabel.disabled=True
        nextlabel.disabled=True
        prevlabel.disabled=True
        lastlabel.disabled=True
        if M.context_waveform:
            waveform_quad_fuchsia.data.update(left=[], right=[], top=[], bottom=[])
            waveform_span_red.location=0
            waveform_span_red.visible=False
        if M.context_spectrogram:
            spectrogram_quad_fuchsia.data.update(left=[], right=[], top=[], bottom=[])
            spectrogram_span_red.location=0
            spectrogram_span_red.visible=False
        probability_span_red.visible=False
        reset_video()

    for ichannel in range(M.audio_nchannels):
        if ichannel+1 in M.context_waveform:
            idx = M.context_waveform.index(ichannel+1)
            xdata = xwav[ichannel]
            ydata = (ywav[idx] + len(M.context_waveform)-1-2*idx) / len(M.context_waveform)
            waveform_source[idx].data.update(x=xdata, y=ydata)
            ipalette = int(np.floor(scales[idx] /
                                    np.iinfo(np.int16).max *
                                    len(snippet_palette)))
            waveform_glyph[idx].glyph.line_color = snippet_palette[ipalette]
        if ichannel+1 in M.context_spectrogram:
            idx = M.context_spectrogram.index(ichannel+1)
            if not np.isnan(gram_time[idx][0]):
                spectrogram_glyph[idx].glyph.x = xwav0 + gram_time[idx][0]
                spectrogram_glyph[idx].glyph.y = len(M.context_spectrogram) - 1 - idx
                spectrogram_glyph[idx].glyph.dw = gram_time[idx][-1] - gram_time[idx][0]
                spectrogram_glyph[idx].glyph.dh = 1
                spectrogram_source[idx].data.update(image=[np.log10(1e-15+ \
                        gram_image[idx][ilow[idx]:1+ihigh[idx],:])])
            else:
                spectrogram_source[idx].data.update(image=[])

    probability_source.data.update(xs=xprob, ys=yprob,
                                   colors=[M.cluster_dot_colors[x] for x in M.clustered_labels],
                                   labels=list(M.clustered_labels))

    if M.context_waveform:
        waveform_quad_grey_clustered.data.update(left=left_clustered,
                                                 right=right_clustered,
                                                 top=[1]*len(left_clustered),
                                                 bottom=[0]*len(left_clustered))
        waveform_quad_grey_annotated.data.update(left=left_annotated,
                                                 right=right_annotated,
                                                 top=[0]*len(left_annotated),
                                                 bottom=[-1]*len(left_annotated))
        waveform_label_source_clustered.data.update(x=xlabel_clustered,
                                                    y=[1]*len(xlabel_clustered),
                                                    text=tlabel_clustered)
        waveform_label_source_annotated.data.update(x=xlabel_annotated,
                                                    y=[-1]*len(xlabel_annotated),
                                                    text=tlabel_annotated)
    if M.context_spectrogram:
        spectrogram_quad_grey_clustered.data.update(left=left_clustered,
                                                    right=right_clustered,
                                                    top=[len(M.context_spectrogram)]*len(left_clustered),
                                                    bottom=[len(M.context_spectrogram)/2]*len(left_clustered))
        spectrogram_quad_grey_annotated.data.update(left=left_annotated,
                                                    right=right_annotated,
                                                    top=[len(M.context_spectrogram)/2]*len(left_annotated),
                                                    bottom=[0]*len(left_annotated))
        spectrogram_label_source_clustered.data.update(x=xlabel_clustered,
                                                       y=[len(M.context_spectrogram)]*len(xlabel_clustered),
                                                       text=tlabel_clustered)
        spectrogram_label_source_annotated.data.update(x=xlabel_annotated,
                                                       y=[0]*len(xlabel_annotated),
                                                       text=tlabel_annotated)

    if M.remaining_isounds:
        remaining.disabled=False

def save_update(n):
    save_indicator.label=str(n)
    if n==0:
        save_indicator.button_type="default"
    elif n<10:
        save_indicator.button_type="warning"
    else:
        save_indicator.button_type="danger"

def waitfor_update():
    if len(M.waitfor_job)>0:
        waitfor.disabled=False

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

def recordings_update():
    M.clustered_activations = None
    if M.dfs:
        kinds = kinds_touse.value.split(',')
        labels = labels_touse.value.split(',')
        wavfiles = set()
        M.clustered_sounds = []
        M.clustered_recording2firstsound = {}
        for df,subdir in zip(M.dfs,M.subdirs):
            bidx = np.logical_and(np.array(df[3].apply(lambda x: x in kinds)),
                                  np.array(df[4].apply(lambda x: x in labels)))
            if any(bidx):
                M.clustered_sounds.extend(list(df[bidx].apply(lambda x:
                                                              {"file": os.path.join(subdir,x[0]),
                                                               "ticks": [x[1],x[2]],
                                                               "kind": x[3],
                                                               "label": x[4]},
                                                              1)))
                wavfiles |= set(df[bidx].apply(lambda x: os.path.join(subdir,x[0]), 1))

        M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_sounds]
        isort = np.argsort(M.clustered_starts_sorted)
        M.clustered_sounds = [M.clustered_sounds[x] for x in isort]
        M.clustered_starts_sorted = [M.clustered_starts_sorted[x] for x in isort]
        M.clustered_stops = [x['ticks'][1] for x in M.clustered_sounds]
        M.iclustered_stops_sorted = np.argsort(M.clustered_stops)
        M.clustered_labels = set([x['label'] for x in M.clustered_sounds])
        M.cluster_dot_colors = { l:c for l,c in zip(M.clustered_labels,
                                                    cycle(cluster_dot_palette)) }

        M.species = M.words = M.nohyphens = M.kinds = [""]
        M.ispecies = M.iwords = M.inohyphens = M.ikinds = 0

        recordings.options = sorted(list(wavfiles))
        for recording in recordings.options:
            M.clustered_recording2firstsound[recording] = \
                  next(filter(lambda x: x[1]['file']==recording, enumerate(M.clustered_sounds)))[0]
        recordings.options = [""] + recordings.options
        if recordings.value != "":
            M.user_changed_recording=False
        recordings.value = ""
    else:
        M.clustered_sounds = None
        M.clustered_starts_sorted = M.clustered_stops = M.iclustered_stops_sorted = None
        M.clustered_recording2firstsound = {}
        recordings.options = []

    M.isnippet = -1
    M.xcluster = M.ycluster = M.zcluster = np.nan
    M.xsnippet=M.ysnippet=0
    snippets_update(True)

    M.layers, M.species, M.words, M.nohyphens, M.kinds = [], [], [], [], []
    M.ilayer = M.ispecies = M.iword = M.inohyphen = M.ikind = M.nlayers = 0

    which_layer.options = M.layers
    which_species.options = M.species
    which_word.options = M.words
    which_nohyphen.options = M.nohyphens
    which_kind.options = M.kinds

    circle_radius.disabled=False
    dot_size.disabled=False
    dot_alpha.disabled=False

    kwargs = dict(dx=[0,0,0,0,0,0,0,0],
                  dy=[0,0,0,0,0,0,0,0],
                  dz=[0,0,0,0,0,0,0,0],
                  dl=['', '', '', '', '', '', '', ''],
                  dc=['#ffffff00', '#ffffff00', '#ffffff00', '#ffffff00',
                      '#ffffff00', '#ffffff00', '#ffffff00', '#ffffff00'])
    cluster_dots.data.update(**kwargs)

    context_update()

def _groundtruth_update():
    M.dfs, M.subdirs = labelcounts_update()
    cluster_these_layers_update()
    recordings_update()
    M.save_state_callback()
    recordings.disabled=False
    recordings.css_classes = []
    groundtruth_folder_button.disabled=True
    buttons_update()
    M.user_copied_parameters=0

def groundtruth_update():
    if groundtruth_folder.value!="" and not os.path.isdir(groundtruth_folder.value):
        bokehlog.info("ERROR: "+groundtruth_folder.value+" does not exist")
        return
    if M.user_copied_parameters<2:
        M.user_copied_parameters += 1
        recordings.disabled=True
        recordings.css_classes = ['changed']
        groundtruth_folder_button.disabled=True
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(_groundtruth_update)
        else:
            _groundtruth_update()

def labels_touse_update_other():
    theselabels_touse = [x.value for x in label_texts if x.value!='']
    if 'other' not in theselabels_touse:
        theselabels_touse.append('other')
    labels_touse.value=str.join(',',theselabels_touse)

def buttons_update():
    for button in wizard_buttons:
        button.button_type="success" if button==M.wizard else "default"
    for button in action_buttons:
        button.button_type="primary" if button==M.action else "default"
        button.disabled = button not in wizard2actions[M.wizard]
    if M.action in [detect,classify,ethogram]:
        wavcsv_files_button.label='wav files:'
    elif M.action==misses:
        wavcsv_files_button.label='csv files:'
    else:
        wavcsv_files_button.label='wav,csv files:'
    if M.action == classify:
        model_file_button.label='pb folder:'
    elif M.action == ethogram:
        model_file_button.label='threshold file:'
    elif M.action == ensemble or M.action == freeze:
        model_file_button.label='checkpoint file(s):'
    else:
        model_file_button.label='checkpoint file:'
    for button in parameter_buttons:
        button.disabled = button not in action2parameterbuttons[M.action]
    okay=True if M.action else False
    for textinput in parameter_textinputs:
        if textinput in action2parametertextinputs[M.action]:
            if textinput==pca_fraction_variance_to_retain:
                pca_fraction_variance_to_retain.disabled=False \
                        if cluster_algorithm.value[:4] in ['tSNE','UMAP'] else True
            elif textinput==tsne_perplexity:
                tsne_perplexity.disabled=False \
                        if cluster_algorithm.value.startswith('tSNE') else True
            elif textinput==tsne_exaggeration:
                tsne_exaggeration.disabled=False \
                        if cluster_algorithm.value.startswith('tSNE') else True
            elif textinput==umap_neighbors:
                umap_neighbors.disabled=False \
                        if cluster_algorithm.value.startswith('UMAP') else True
            elif textinput==umap_distance:
                umap_distance.disabled=False \
                        if cluster_algorithm.value.startswith('UMAP') else True
            elif textinput in detect_parameters.values():
                thislogic = detect_parameters_enable_logic[textinput]
                if thislogic:
                    textinput.disabled = detect_parameters[thislogic[0]].value not in thislogic[1]
                else:
                    textinput.disabled = False
            elif textinput in model_parameters.values():
                thislogic = model_parameters_enable_logic[textinput]
                if thislogic:
                    textinput.disabled = model_parameters[thislogic[0]].value not in thislogic[1]
                else:
                    textinput.disabled = False
            else:
                textinput.disabled=False
            if textinput.disabled==False and textinput.value=='':
                if M.action==classify:
                    if textinput not in [labels_touse, prevalences]:
                        okay=False
                elif M.action==congruence:
                    if textinput not in [validation_files, test_files]:
                        okay=False
                elif textinput in detect_parameters.values():
                    if detect_parameters_required[textinput]:
                        okay=False
                elif textinput in model_parameters.values():
                    if model_parameters_required[textinput]:
                        okay=False
                else:
                    if textinput not in [test_files, restore_from]:
                        okay=False
        else:
            textinput.disabled=True
    if M.action==classify and \
            prevalences.value!='' and labels_touse.value=='':
        okay=False
    if M.action==congruence and \
            validation_files.value=='' and test_files.value=='':
        okay=False
    if M.action==cluster and len(cluster_these_layers.value)==0:
        okay=False
    doit.button_type="default"
    if okay:
        doit.disabled=False
        doit.button_type="danger"
    else:
        doit.disabled=True
        doit.button_type="default"

def file_dialog_update():
    thispath = os.path.join(M.file_dialog_root,M.file_dialog_filter)
    globfiles = glob.glob(thispath)
    uniqdirnames = set([os.path.dirname(x) for x in globfiles])
    files = ['.', '..']
    for thisfile in natsorted(globfiles):
        try:
            os.stat(thisfile)
            files.append(thisfile)
        except:
            pass
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

def labelcounts_update():
    dfs, subdirs = [], []
    if not os.path.isdir(groundtruth_folder.value):
        labelcounts.text = ""
        return dfs, subdirs
    for subdir in filter(lambda x: os.path.isdir(os.path.join(groundtruth_folder.value,x)), \
                         os.listdir(groundtruth_folder.value)):
        for csvfile in filter(lambda x: x.endswith('.csv'), \
                              os.listdir(os.path.join(groundtruth_folder.value, subdir))):
            filepath = os.path.join(groundtruth_folder.value, subdir, csvfile)
            if os.path.getsize(filepath) > 0:
                try:
                    df = pd.read_csv(filepath, header=None, index_col=False)
                except:
                    bokehlog.info("WARNING: "+csvfile+" is not in the correct format")
                if 5<=len(df.columns)<=6:
                    dfs.append(df)
                    subdirs.append(subdir)
                else:
                    bokehlog.info("WARNING: "+csvfile+" is not in the correct format")
    if dfs:
        df = pd.concat(dfs)
        M.kinds = sorted(set(df[3]))
        words = sorted(set(df[4]))
        bkinds = {}
        table = np.empty((1+len(words),len(M.kinds)), dtype=int)
        for iword,word in enumerate(words):
            bword = np.array(df[4]==word)
            for ikind,kind in enumerate(M.kinds):
                if kind not in bkinds:
                    bkinds[kind] = np.array(df[3]==kind)
                table[iword,ikind] = np.sum(np.logical_and(bkinds[kind], bword))
        for ikind,kind in enumerate(M.kinds):
            table[len(words),ikind] = np.sum(bkinds[kind])
        words += ['TOTAL']

        if len(words)>len(M.kinds):
            rows = words
            cols = M.kinds
        else:
            rows = M.kinds
            cols = words
            table = np.transpose(table)
        table_str = '<table><tr><th></th><th nowrap>'+'</th><th nowrap>'.join(cols)+'</th></tr>'
        for irow,row in enumerate(rows):
            table_str += '<tr><th nowrap>'+row+'</th>'
            for icol,col in enumerate(cols):
                table_str += '<td align="center">'+str(table[irow,icol])+'</td>'
            table_str += '</tr>'
        table_str += '</table>'
        labelcounts.text = table_str
    else:
        labelcounts.text = ""
    return dfs, subdirs

async def status_ticker_update():
    deletefailures.disabled=True
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
                deletefailures.disabled=False
            newtext.append("<span style='color:"+color+"'>"+k+"</span>")
        newtext = (', ').join(newtext)
    else:
        newtext = ''
    status_ticker.text = status_ticker_pre+newtext+status_ticker_post

def init(_bokeh_document):
    global bokeh_document, configuration_file
    global p_cluster, cluster_dots, precomputed_dots, dot_size_cluster, dot_alpha_cluster, cluster_circle_fuchsia, cluster_dot_palette, circle_radius, dot_size, dot_alpha
    global p_snippets, snippet_palette, snippets_dy, snippets_both, snippets_label_sources_clustered, snippets_label_sources_annotated, snippets_wave_sources, snippets_wave_glyphs, snippets_gram_sources, snippets_gram_glyphs, snippets_quad_grey, snippets_quad_fuchsia
    global p_waveform, waveform_span_red, waveform_quad_grey_clustered, waveform_quad_grey_annotated, waveform_quad_grey_pan, waveform_quad_fuchsia, waveform_source, waveform_glyph, waveform_label_source_clustered, waveform_label_source_annotated
    global p_spectrogram, spectrogram_span_red, spectrogram_quad_grey_clustered, spectrogram_quad_grey_annotated, spectrogram_quad_grey_pan, spectrogram_quad_fuchsia, spectrogram_source, spectrogram_glyph, spectrogram_label_source_clustered, spectrogram_label_source_annotated, spectrogram_length
    global p_probability, probability_span_red, probability_source, probability_glyph
    global which_layer, which_species, which_word, which_nohyphen, which_kind
    global color_picker
    global zoom_width, zoom_offset, zoomin, zoomout, reset, panleft, panright, allleft, allout, allright, firstlabel, nextlabel, prevlabel, lastlabel
    global save_indicator, nsounds_per_label_buttons, label_texts
    global load_multimedia, play, video_slider, load_multimedia_callback, play_callback, video_slider_callback, video_toggle, video_div
    global undo, redo, remaining
    global recordings
    global detect, misses, train, leaveoneout, leaveallout, xvalidate, mistakes, activations, cluster, visualize, accuracy, freeze, ensemble, classify, ethogram, compare, congruence
    global status_ticker, waitfor, deletefailures
    global file_dialog_source, configuration_contents
    global logs_folder_button, logs_folder, model_file_button, model_file, wavcsv_files_button, wavcsv_files, groundtruth_folder_button, groundtruth_folder, validation_files_button, test_files_button, validation_files, test_files, labels_touse_button, labels_touse, kinds_touse_button, kinds_touse, prevalences_button, prevalences, copy, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, generalize, tunehyperparameters, findnovellabels, examineerrors, testdensely, doit, nsteps, restore_from, save_and_validate_period, validate_percentage, mini_batch, kfold, activations_equalize_ratio, activations_max_sounds, pca_fraction_variance_to_retain, tsne_perplexity, tsne_exaggeration, umap_neighbors, umap_distance, cluster_algorithm, cluster_these_layers, precision_recall_ratios, congruence_portion, congruence_convolve, congruence_measure, context_ms, shiftby_ms, optimizer, learning_rate, nreplicates, batch_seed, weights_seed, file_dialog_string, file_dialog_table, readme_contents, labelcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs, status_ticker_update, status_ticker_pre, status_ticker_post
    global detect_parameters, detect_parameters_enable_logic, detect_parameters_required, detect_parameters_partitioned
    global doubleclick_parameters, doubleclick_parameters_enable_logic, doubleclick_parameters_required
    global model_parameters, model_parameters_enable_logic, model_parameters_required, model_parameters_partitioned

    bokeh_document = _bokeh_document

    M.cluster_circle_color = M.cluster_circle_color

    if '#' in M.cluster_dot_palette:
      cluster_dot_palette = ast.literal_eval(M.cluster_dot_palette)
    else:
      cluster_dot_palette = getattr(palettes, M.cluster_dot_palette)

    snippet_palette = getattr(palettes, M.snippets_colormap)

    dot_size_cluster = ColumnDataSource(data=dict(ds=[M.state["dot_size"]]))
    dot_alpha_cluster = ColumnDataSource(data=dict(da=[M.state["dot_alpha"]]))

    cluster_dots = ColumnDataSource(data=dict(dx=[], dy=[], dz=[], dl=[], dc=[]))
    cluster_circle_fuchsia = ColumnDataSource(data=dict(cx=[], cy=[], cz=[], cr=[], cc=[]))
    p_cluster = ScatterNd(dx='dx', dy='dy', dz='dz', dl='dl', dc='dc',
                          dots_source=cluster_dots,
                          cx='cx', cy='cy', cz='cz', cr='cr', cc='cc',
                          circle_fuchsia_source=cluster_circle_fuchsia,
                          ds='ds',
                          dot_size_source=dot_size_cluster,
                          da='da',
                          dot_alpha_source=dot_alpha_cluster,
                          width=M.gui_width_pix//2)
    p_cluster.on_change("click_position", lambda a,o,n: C.cluster_tap_callback(n))

    precomputed_dots = None

    snippets_dy = 2*((len(M.snippets_waveform)>0) + (len(M.snippets_spectrogram)>0))
    snippets_both = 2*((len(M.snippets_waveform)>0) and (len(M.snippets_spectrogram)>0))

    p_snippets = figure(plot_width=M.gui_width_pix//2, \
                        background_fill_color='#FFFFFF', toolbar_location=None)
    p_snippets.toolbar.active_drag = None
    p_snippets.grid.visible = False
    p_snippets.xaxis.visible = False
    p_snippets.yaxis.visible = False

    snippets_gram_sources=[None]*(M.snippets_nx*M.snippets_ny)
    snippets_gram_glyphs=[None]*(M.snippets_nx*M.snippets_ny)
    for ixy in range(M.snippets_nx*M.snippets_ny):
        snippets_gram_sources[ixy]=[None]*len(M.snippets_spectrogram)
        snippets_gram_glyphs[ixy]=[None]*len(M.snippets_spectrogram)
        for idx in range(len(M.snippets_spectrogram)):
            snippets_gram_sources[ixy][idx] = ColumnDataSource(data=dict(image=[]))
            snippets_gram_glyphs[ixy][idx] = p_snippets.image('image',
                    source=snippets_gram_sources[ixy][idx],
                    palette=M.spectrogram_colormap)

    snippets_quad_grey = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=snippets_quad_grey,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey")

    snippets_wave_sources=[None]*(M.snippets_nx*M.snippets_ny)
    snippets_wave_glyphs=[None]*(M.snippets_nx*M.snippets_ny)
    for ixy in range(M.snippets_nx*M.snippets_ny):
        snippets_wave_sources[ixy]=[None]*len(M.snippets_waveform)
        snippets_wave_glyphs[ixy]=[None]*len(M.snippets_waveform)
        for idx in range(len(M.snippets_waveform)):
            snippets_wave_sources[ixy][idx]=ColumnDataSource(data=dict(x=[], y=[]))
            snippets_wave_glyphs[ixy][idx]=p_snippets.line(
                    'x', 'y', source=snippets_wave_sources[ixy][idx])

    xdata = [(i%M.snippets_nx)*(M.snippets_gap_pix+M.snippets_pix)
             for i in range(M.snippets_nx*M.snippets_ny)]
    ydata = [-(i//M.snippets_nx*snippets_dy-1)
             for i in range(M.snippets_nx*M.snippets_ny)]
    text = ['' for i in range(M.snippets_nx*M.snippets_ny)]
    snippets_label_sources_clustered = ColumnDataSource(data=dict(x=xdata, y=ydata, text=text))
    p_snippets.text('x', 'y', source=snippets_label_sources_clustered, text_font_size='6pt',
                    text_baseline='top',
                    text_color='black' if M.snippets_waveform else 'white')

    xdata = [(i%M.snippets_nx)*(M.snippets_gap_pix+M.snippets_pix)
             for i in range(M.snippets_nx*M.snippets_ny)]
    ydata = [-(i//M.snippets_nx*snippets_dy+1+snippets_both)
             for i in range(M.snippets_nx*M.snippets_ny)]
    text_annotated = ['' for i in range(M.snippets_nx*M.snippets_ny)]
    snippets_label_sources_annotated = ColumnDataSource(data=dict(x=xdata, y=ydata,
                                                                  text=text_annotated))
    p_snippets.text('x', 'y', source=snippets_label_sources_annotated,
                    text_font_size='6pt',
                    text_color='white' if M.snippets_spectrogram else 'black')

    snippets_quad_fuchsia = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=snippets_quad_fuchsia,
                fill_color=None, line_color="fuchsia")

    p_snippets.on_event(Tap, C.snippets_tap_callback)
    p_snippets.on_event(DoubleTap, C.snippets_doubletap_callback)

    p_waveform = figure(plot_width=M.gui_width_pix,
                        plot_height=M.context_waveform_height_pix,
                        background_fill_color='#FFFFFF', toolbar_location=None)
    p_waveform.toolbar.active_drag = None
    p_waveform.grid.visible = False
    if M.context_spectrogram:
        p_waveform.xaxis.visible = False
    else:
        p_waveform.xaxis.axis_label = 'Time (sec)'
    p_waveform.yaxis.axis_label = ""
    p_waveform.yaxis.ticker = []
    p_waveform.x_range.range_padding = p_waveform.y_range.range_padding = 0.0
    p_waveform.y_range.start = -1
    p_waveform.y_range.end = 1

    waveform_span_red = Span(location=0, dimension='height', line_color='red')
    p_waveform.add_layout(waveform_span_red)
    waveform_span_red.visible=False

    waveform_quad_grey_clustered = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_waveform.quad('left','right','top','bottom',source=waveform_quad_grey_clustered,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    waveform_quad_grey_annotated = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_waveform.quad('left','right','top','bottom',source=waveform_quad_grey_annotated,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    waveform_quad_grey_pan = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_waveform.quad('left','right','top','bottom',source=waveform_quad_grey_pan,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    waveform_quad_fuchsia = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_waveform.quad('left','right','top','bottom',source=waveform_quad_fuchsia,
                fill_color=None, line_color="fuchsia", level='underlay')

    waveform_source=[None]*len(M.context_waveform)
    waveform_glyph=[None]*len(M.context_waveform)
    for idx in range(len(M.context_waveform)):
        waveform_source[idx] = ColumnDataSource(data=dict(x=[], y=[]))
        waveform_glyph[idx] = p_waveform.line('x', 'y', source=waveform_source[idx])

    waveform_label_source_clustered = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_waveform.text('x', 'y', source=waveform_label_source_clustered,
                   text_font_size='6pt', text_align='center', text_baseline='top',
                   text_line_height=0.8, level='underlay')
    waveform_label_source_annotated = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_waveform.text('x', 'y', source=waveform_label_source_annotated,
                   text_font_size='6pt', text_align='center', text_baseline='bottom',
                   text_line_height=0.8, level='underlay')

    p_waveform.on_event(DoubleTap, lambda e: C.context_doubletap_callback(e, 0))

    p_waveform.on_event(PanStart, C.waveform_pan_start_callback)
    p_waveform.on_event(Pan, C.waveform_pan_callback)
    p_waveform.on_event(PanEnd, C.waveform_pan_end_callback)
    p_waveform.on_event(Tap, C.waveform_tap_callback)

    p_spectrogram = figure(plot_width=M.gui_width_pix,
                           plot_height=M.context_spectrogram_height_pix,
                           background_fill_color='#FFFFFF', toolbar_location=None)
    p_spectrogram.toolbar.active_drag = None
    p_spectrogram.x_range.range_padding = p_spectrogram.y_range.range_padding = 0
    p_spectrogram.xgrid.visible = False
    p_spectrogram.ygrid.visible = True
    p_spectrogram.xaxis.axis_label = 'Time (sec)'
    p_spectrogram.yaxis.axis_label = 'Frequency (' + M.context_spectrogram_units + ')'
    p_spectrogram.yaxis.ticker = list(range(1+len(M.context_spectrogram)))

    spectrogram_source = [None]*len(M.context_spectrogram)
    spectrogram_glyph = [None]*len(M.context_spectrogram)
    for idx in range(len(M.context_spectrogram)):
        spectrogram_source[idx] = ColumnDataSource(data=dict(image=[]))
        spectrogram_glyph[idx] = p_spectrogram.image('image',
                                                     source=spectrogram_source[idx],
                                                     palette=M.spectrogram_colormap,
                                                     level="image")

    p_spectrogram.on_event(MouseWheel, C.spectrogram_mousewheel_callback)

    p_spectrogram.on_event(DoubleTap,
                           lambda e: C.context_doubletap_callback(e, len(M.context_spectrogram)/2))

    p_spectrogram.on_event(PanStart, C.spectrogram_pan_start_callback)
    p_spectrogram.on_event(Pan, C.spectrogram_pan_callback)
    p_spectrogram.on_event(PanEnd, C.spectrogram_pan_end_callback)
    p_spectrogram.on_event(Tap, C.spectrogram_tap_callback)

    spectrogram_span_red = Span(location=0, dimension='height', line_color='red')
    p_spectrogram.add_layout(spectrogram_span_red)
    spectrogram_span_red.visible=False

    spectrogram_quad_grey_clustered = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_spectrogram.quad('left','right','top','bottom',source=spectrogram_quad_grey_clustered,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    spectrogram_quad_grey_annotated = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_spectrogram.quad('left','right','top','bottom',source=spectrogram_quad_grey_annotated,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    spectrogram_quad_grey_pan = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_spectrogram.quad('left','right','top','bottom',source=spectrogram_quad_grey_pan,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    spectrogram_quad_fuchsia = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_spectrogram.quad('left','right','top','bottom',source=spectrogram_quad_fuchsia,
                fill_color=None, line_color="fuchsia", level='underlay')

    spectrogram_label_source_clustered = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_spectrogram.text('x', 'y', source=spectrogram_label_source_clustered,
                   text_font_size='6pt', text_align='center', text_baseline='top',
                   text_line_height=0.8, level='underlay', text_color='white')
    spectrogram_label_source_annotated = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_spectrogram.text('x', 'y', source=spectrogram_label_source_annotated,
                   text_font_size='6pt', text_align='center', text_baseline='bottom',
                   text_line_height=0.8, level='underlay', text_color='white')

    TOOLTIPS = """
        <div><div><span style="color:@colors;">@labels</span></div></div>
    """

    p_probability = figure(plot_width=M.gui_width_pix, tooltips=TOOLTIPS,
                           plot_height=M.context_probability_height_pix,
                           background_fill_color='#FFFFFF', toolbar_location=None)
    p_probability.toolbar.active_drag = None
    p_probability.grid.visible = False
    p_probability.yaxis.axis_label = "Probability"
    p_probability.x_range.range_padding = p_probability.y_range.range_padding = 0.0
    p_probability.y_range.start = 0
    p_probability.y_range.end = 1
    p_probability.xaxis.visible = False

    probability_source = ColumnDataSource(data=dict(xs=[], ys=[], colors=[], labels=[]))
    probability_glyph = p_probability.multi_line(xs='xs', ys='ys',
                                                 source=probability_source, color='colors')

    probability_span_red = Span(location=0, dimension='height', line_color='red')
    p_probability.add_layout(probability_span_red)
    probability_span_red.visible=False

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

    color_picker = ColorPicker(title="color:", disabled=True)
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

    if M.gui_snippets_spectrogram or M.gui_context_spectrogram:
        spectrogram_length = TextInput(value=','.join([str(x) for x in M.spectrogram_length_ms]), \
                                       title="length (msec)", \
                                       disabled=False)
        spectrogram_length.on_change('value', C.spectrogram_window_callback)

    zoom_width = TextInput(value=str(M.context_width_ms),
                             title="width (msec):",
                             disabled=True)
    zoom_width.on_change("value", C.zoom_width_callback)

    zoom_offset = TextInput(value=str(M.context_offset_ms),
                            title="offset (msec):",
                            disabled=True)
    zoom_offset.on_change("value", C.zoom_offset_callback)

    zoomin = Button(label='\u2191', disabled=True)
    zoomin.on_click(C.zoomin_callback)

    zoomout = Button(label='\u2193', disabled=True)
    zoomout.on_click(C.zoomout_callback)

    reset = Button(label='\u25ef', disabled=True)
    reset.on_click(C.zero_callback)

    panleft = Button(label='\u2190', disabled=True)
    panleft.on_click(C.panleft_callback)

    panright = Button(label='\u2192', disabled=True)
    panright.on_click(C.panright_callback)

    allleft = Button(label='\u21e4', disabled=True)
    allleft.on_click(C.allleft_callback)

    allout = Button(label='\u2913', disabled=True)
    allout.on_click(C.allout_callback)

    allright = Button(label='\u21e5', disabled=True)
    allright.on_click(C.allright_callback)

    firstlabel = Button(label='\u21e4 L', disabled=True)
    firstlabel.on_click(C.firstlabel_callback)

    prevlabel = Button(label='\u2190 L', disabled=True)
    prevlabel.on_click(C.prevlabel_callback)

    nextlabel = Button(label='L \u2192', disabled=True)
    nextlabel.on_click(C.nextlabel_callback)

    lastlabel = Button(label='L \u21e5', disabled=True)
    lastlabel.on_click(C.lastlabel_callback)

    save_indicator = Button(label='0')

    nsounds_per_label_callbacks=[]
    nsounds_per_label_buttons=[]
    label_callbacks=[]
    label_texts=[]

    for i in range(M.nlabels):
        nsounds_per_label_callbacks.append(lambda i=i: C.nsounds_per_label_callback(i))
        nsounds_per_label_buttons.append(Button(label='0', css_classes=['hide-label'], width=40))
        nsounds_per_label_buttons[-1].on_click(nsounds_per_label_callbacks[-1])

        label_callbacks.append(lambda a,o,n,i=i: C.label_callback(n,i))
        label_texts.append(TextInput(value=M.state['labels'][i],
                                     css_classes=['hide-label'], height=32))
        label_texts[-1].on_change("value", label_callbacks[-1])

    C.nsounds_per_label_callback(M.ilabel)

    load_multimedia = Paragraph(visible=False)
    load_multimedia_callback = CustomJS(code=C.load_multimedia_callback_code % ("",""))
    load_multimedia.js_on_change('text', load_multimedia_callback)

    play = Button(label='play', disabled=True)
    play_callback = CustomJS(args=dict(waveform_span_red=waveform_span_red,
                                       spectrogram_span_red=spectrogram_span_red,
                                       probability_span_red=probability_span_red,
                                       p=p_waveform if M.context_waveform else p_spectrogram),
                             code=C.play_callback_code)
    play.js_on_event(ButtonClick, play_callback)
    play.on_change('disabled', lambda a,o,n: reset_video())

    video_toggle = Toggle(label='video', active=False, disabled=True)
    video_toggle.on_click(lambda x: context_update())

    video_div = Div(text="""<video id="context_video"></video>""",
                    style={'width':'1px'})

    video_slider = Slider(title="", show_value=False, visible=False,
                          start=0, end=1, value=0, step=1)
    video_slider_callback = CustomJS(args=dict(waveform_span_red=waveform_span_red,
                                               spectrogram_span_red=spectrogram_span_red,
                                               probability_span_red=probability_span_red),
                                     code=C.video_slider_callback_code)
    video_slider.js_on_change("value", video_slider_callback)

    undo = Button(label='undo', disabled=True)
    undo.on_click(C.undo_callback)

    redo = Button(label='redo', disabled=True)
    redo.on_click(C.redo_callback)

    remaining = Button(label='add remaining', disabled=True)
    remaining.on_click(C.remaining_callback)

    recordings = Select(title="recording:", height=50)
    recordings.on_change('value', C.recordings_callback)

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

    ensemble = Button(label='ensemble')
    ensemble.on_click(lambda: C.action_callback(ensemble, C.ensemble_actuate))

    classify = Button(label='classify')
    classify.on_click(C.classify_callback)

    ethogram = Button(label='ethogram')
    ethogram.on_click(lambda: C.action_callback(ethogram, C.ethogram_actuate))

    compare = Button(label='compare')
    compare.on_click(lambda: C.action_callback(compare, C.compare_actuate))

    congruence = Button(label='congruence')
    congruence.on_click(lambda: C.action_callback(congruence, C.congruence_actuate))

    status_ticker_pre="<div style='overflow:auto; white-space:nowrap; width:"+str(M.gui_width_pix-236)+"px'>status: "
    status_ticker_post="</div>"
    status_ticker = Div(text=status_ticker_pre+status_ticker_post)

    file_dialog_source = ColumnDataSource(data=dict(names=[], sizes=[], dates=[]))
    file_dialog_source.selected.on_change('indices', C.file_dialog_callback)

    file_dialog_columns = [
        TableColumn(field="names", title="Name", width=M.gui_width_pix//2-50-115-30),
        TableColumn(field="sizes", title="Size", width=50, \
                    formatter=NumberFormatter(format="0 b")),
        TableColumn(field="dates", title="Date", width=115, \
                    formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
    ]
    file_dialog_table = DataTable(source=file_dialog_source, \
                                  columns=file_dialog_columns, \
                                  height=727, width=M.gui_width_pix//2-11, \
                                  index_position=None,
                                  fit_columns=False)

    deletefailures = Toggle(label='delete failures', active=False, disabled=True)
    deletefailures.on_click(C.deletefailures_callback)

    waitfor = Toggle(label='wait for last job', active=False, disabled=True)
    waitfor.on_click(C.waitfor_callback)

    logs_folder_button = Button(label='logs folder:', width=120)
    logs_folder_button.on_click(C.logs_callback)
    logs_folder = TextInput(value=M.state['logs_folder'], title="", disabled=False)
    logs_folder.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    model_file_button = Button(label='checkpoint file:', width=120)
    model_file_button.on_click(C.model_callback)
    model_file = TextInput(value=M.state['model_file'], title="", disabled=False)
    model_file.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    wavcsv_files_button = Button(label='wav,csv files:', width=120)
    wavcsv_files_button.on_click(C.wavcsv_files_callback)
    wavcsv_files = TextInput(value=M.state['wavcsv_files'], title="", disabled=False)
    wavcsv_files.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    groundtruth_folder_button = Button(label='ground truth:', width=120)
    groundtruth_folder_button.on_click(C.groundtruth_callback)
    groundtruth_folder = TextInput(value=M.state['groundtruth_folder'], title="", disabled=False)
    groundtruth_folder.on_change('value', lambda a,o,n: groundtruth_update())

    validation_files_button = Button(label='validation files:', width=120)
    validation_files_button.on_click(C.validationfiles_callback)
    validation_files = TextInput(value=M.state['validation_files'], title="", disabled=False)
    validation_files.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    test_files_button = Button(label='test files:', width=120)
    test_files_button.on_click(C.test_files_callback)
    test_files = TextInput(value=M.state['test_files'], title="", disabled=False)
    test_files.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    labels_touse_button = Button(label='labels to use:', width=120)
    labels_touse_button.on_click(C.labels_touse_callback)
    labels_touse = TextInput(value=M.state['labels_touse'], title="", disabled=False)
    labels_touse.on_change('value', lambda a,o,n: C.touse_callback(n,labels_touse_button))

    kinds_touse_button = Button(label='kinds to use:', width=120)
    kinds_touse = TextInput(value=M.state['kinds_touse'], title="", disabled=False)
    kinds_touse.on_change('value', lambda a,o,n: C.touse_callback(n,kinds_touse_button))

    prevalences_button = Button(label='prevalences:', width=120)
    prevalences_button.on_click(C.prevalences_callback)
    prevalences = TextInput(value=M.state['prevalences'], title="", disabled=False)
    prevalences.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

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

    nsteps = TextInput(value=M.state['nsteps'], title="# steps", disabled=False)
    nsteps.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    restore_from = TextInput(value=M.state['restore_from'], title="restore from", disabled=False)
    restore_from.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    save_and_validate_period = TextInput(value=M.state['save_and_validate_period'], \
                                                title="validate period", \
                                                disabled=False)
    save_and_validate_period.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    validate_percentage = TextInput(value=M.state['validate_percentage'], \
                                           title="validate %", \
                                           disabled=False)
    validate_percentage.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    mini_batch = TextInput(value=M.state['mini_batch'], \
                                  title="mini-batch", \
                                  disabled=False)
    mini_batch.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    kfold = TextInput(value=M.state['kfold'], title="k-fold",  disabled=False)
    kfold.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    activations_equalize_ratio = TextInput(value=M.state['activations_equalize_ratio'], \
                                             title="equalize ratio", \
                                             disabled=False)
    activations_equalize_ratio.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    activations_max_sounds = TextInput(value=M.state['activations_max_sounds'], \
                                          title="max sounds", \
                                          disabled=False)
    activations_max_sounds.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    pca_fraction_variance_to_retain = TextInput(value=M.state['pca_fraction_variance_to_retain'], \
                                                       title="PCA fraction", \
                                                       disabled=False)
    pca_fraction_variance_to_retain.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    tsne_perplexity = TextInput(value=M.state['tsne_perplexity'], \
                                       title="perplexity", \
                                       disabled=False)
    tsne_perplexity.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    tsne_exaggeration = TextInput(value=M.state['tsne_exaggeration'], \
                                        title="exaggeration", \
                                        disabled=False)
    tsne_exaggeration.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    umap_neighbors = TextInput(value=M.state['umap_neighbors'], \
                                      title="neighbors", \
                                      disabled=False)
    umap_neighbors.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    umap_distance = TextInput(value=M.state['umap_distance'], \
                                     title="distance", \
                                     disabled=False)
    umap_distance.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    precision_recall_ratios = TextInput(value=M.state['precision_recall_ratios'], \
                                               title="P/Rs", \
                                               disabled=False)
    precision_recall_ratios.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))
    
    congruence_portion = Select(title="portion", height=50, \
                                value=M.state['congruence_portion'], \
                                options=["union", "intersection"])
    congruence_portion.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    congruence_convolve = TextInput(value=M.state['congruence_convolve'], \
                                                   title="convolve (msec)", \
                                                   disabled=False)
    congruence_convolve.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))
    
    congruence_measure = Select(title="measure", height=50, \
                                value=M.state['congruence_measure'], \
                                options=["label", "tic", "both"])
    congruence_measure.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    context_ms = TextInput(value=M.state['context_ms'], \
                                  title="context (msec)", \
                                  disabled=False)
    context_ms.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    shiftby_ms = TextInput(value=M.state['shiftby_ms'], \
                                  title="shift by (msec)", \
                                  disabled=False)
    shiftby_ms.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    optimizer = Select(title="optimizer", height=50, \
                       value=M.state['optimizer'], \
                       options=["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSProp", "SGD"])
    optimizer.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    learning_rate = TextInput(value=M.state['learning_rate'], \
                                     title="learning rate", \
                                     disabled=False)
    learning_rate.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    V = sys.modules[__name__]

    def get_callback(f):
        def callback(a,o,n):
            f(n,M,V,C) if f else C.generic_parameters_callback(n)
        return callback

    def parse_plugin_parameters(Mparameters, width):
        parameters = OrderedDict()
        parameters_enable_logic = {}
        parameters_required = {}
        for parameter in Mparameters:
            if parameter[2]=='':
                thisparameter = TextInput(value=M.state[parameter[0]], \
                                          title=parameter[1], \
                                          disabled=False, width=parameter[4]*104-10)
            else:
                thisparameter = Select(value=M.state[parameter[0]], \
                                       title=parameter[1], \
                                       options=parameter[2], \
                                       height=50, width=parameter[4]*104-10)
            thisparameter.on_change('value', get_callback(parameter[6]))
            parameters[parameter[0]] = thisparameter
            parameters_enable_logic[thisparameter] = parameter[5]
            parameters_required[thisparameter] = parameter[7]

        parameters_width = [x[4] for x in Mparameters]
        parameters_partitioned = []
        i0=0
        for i in range(len(parameters_width)):
            if sum(parameters_width[i0:i+1]) > width:
                parameters_partitioned.append(range(i0, i))
                i0 = i
            if i==len(parameters_width)-1:
                parameters_partitioned.append(range(i0, i+1))

        return parameters, parameters_enable_logic, parameters_required, parameters_partitioned

    detect_parameters, detect_parameters_enable_logic, detect_parameters_required, detect_parameters_partitioned = parse_plugin_parameters(M.detect_parameters, 8)
    doubleclick_parameters, doubleclick_parameters_enable_logic, doubleclick_parameters_required, _ = parse_plugin_parameters(M.doubleclick_parameters, 1)
    model_parameters, model_parameters_enable_logic, model_parameters_required, model_parameters_partitioned = parse_plugin_parameters(M.model_parameters, 6)

    configuration_contents = TextAreaInput(rows=49-3*len(model_parameters_partitioned),
                                           max_length=50000, \
                                           disabled=True, css_classes=['fixedwidth'])
    if M.configuration_file:
        with open(M.configuration_file, 'r') as fid:
            configuration_contents.value = fid.read()


    cluster_algorithm = Select(title="cluster", height=50, \
                               value=M.state['cluster_algorithm'], \
                               options=["PCA 2D", "PCA 3D", \
                                        "tSNE 2D", "tSNE 3D", \
                                        "UMAP 2D", "UMAP 3D"])
    cluster_algorithm.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    cluster_these_layers = MultiSelect(title='layers', \
                                       value=M.state['cluster_these_layers'], \
                                       options=[],
                                       height=169)
    cluster_these_layers.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))
    cluster_these_layers_update()

    nreplicates = TextInput(value=M.state['nreplicates'], \
                                  title="# replicates", \
                                  disabled=False)
    nreplicates.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    batch_seed = TextInput(value=M.state['batch_seed'], \
                                  title="batch seed", \
                                  disabled=False)
    batch_seed.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    weights_seed = TextInput(value=M.state['weights_seed'], \
                                    title="weights seed", \
                                    disabled=False)
    weights_seed.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    file_dialog_string = TextInput(disabled=False)
    file_dialog_string.on_change("value", C.file_dialog_path_callback)
    file_dialog_string.value = M.state['file_dialog_string']
     
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','README.md'), 'r', encoding='utf-8') as fid:
        contents = fid.read()
    html = markdown.markdown(contents, extensions=['tables','toc'])
    readme_contents = Div(text=html, style={'overflow':'scroll','width':'600px','height':'1440px'})

    labelcounts = Div(text="",
                      style={'overflow-y':'hidden', 'overflow-x':'scroll',
                             'width':str(M.gui_width_pix-450-1)+'px'})
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
        ensemble,
        classify,
        ethogram,
        misses,
        compare,
        congruence])

    parameter_buttons = set([
        logs_folder_button,
        model_file_button,
        wavcsv_files_button,
        groundtruth_folder_button,
        validation_files_button,
        test_files_button,
        labels_touse_button,
        kinds_touse_button,
        prevalences_button])

    parameter_textinputs = set([
        logs_folder,
        model_file,
        wavcsv_files,
        groundtruth_folder,
        validation_files,
        test_files,
        labels_touse,
        kinds_touse,
        prevalences,

        nsteps,
        restore_from,
        save_and_validate_period,
        validate_percentage,
        mini_batch,
        kfold,
        activations_equalize_ratio,
        activations_max_sounds,
        pca_fraction_variance_to_retain,
        tsne_perplexity,
        tsne_exaggeration,
        umap_neighbors,
        umap_distance,
        cluster_algorithm,
        cluster_these_layers,
        precision_recall_ratios,
        congruence_portion,
        congruence_convolve,
        congruence_measure,
        nreplicates,
        batch_seed,
        weights_seed,

        context_ms,
        shiftby_ms,
        optimizer,
        learning_rate] +

        list(detect_parameters.values()) +
        list(model_parameters.values()))

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
            detect: [wavcsv_files_button],
            train: [logs_folder_button, groundtruth_folder_button, labels_touse_button, test_files_button, kinds_touse_button],
            leaveoneout: [logs_folder_button, groundtruth_folder_button, validation_files_button, test_files_button, labels_touse_button, kinds_touse_button],
            leaveallout: [logs_folder_button, groundtruth_folder_button, validation_files_button, test_files_button, labels_touse_button, kinds_touse_button],
            xvalidate: [logs_folder_button, groundtruth_folder_button, test_files_button, labels_touse_button, kinds_touse_button],
            mistakes: [groundtruth_folder_button],
            activations: [logs_folder_button, model_file_button, groundtruth_folder_button, labels_touse_button, kinds_touse_button],
            cluster: [groundtruth_folder_button],
            visualize: [groundtruth_folder_button],
            accuracy: [logs_folder_button],
            freeze: [logs_folder_button, model_file_button],
            ensemble: [logs_folder_button, model_file_button],
            classify: [logs_folder_button, model_file_button, wavcsv_files_button, labels_touse_button, prevalences_button],
            ethogram: [model_file_button, wavcsv_files_button],
            misses: [wavcsv_files_button],
            compare: [logs_folder_button],
            congruence: [groundtruth_folder_button, validation_files_button, test_files_button],
            None: parameter_buttons }

    action2parametertextinputs = {
            detect: [wavcsv_files] + list(detect_parameters.values()),
            train: [context_ms, shiftby_ms, optimizer, learning_rate, nreplicates, batch_seed, weights_seed, logs_folder, groundtruth_folder, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, validate_percentage, mini_batch] + list(model_parameters.values()),
            leaveoneout: [context_ms, shiftby_ms, optimizer, learning_rate, batch_seed, weights_seed, logs_folder, groundtruth_folder, validation_files, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch] + list(model_parameters.values()),
            leaveallout: [context_ms, shiftby_ms, optimizer, learning_rate, batch_seed, weights_seed, logs_folder, groundtruth_folder, validation_files, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch] + list(model_parameters.values()),
            xvalidate: [context_ms, shiftby_ms, optimizer, learning_rate, batch_seed, weights_seed, logs_folder, groundtruth_folder, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch, kfold] + list(model_parameters.values()),
            mistakes: [groundtruth_folder],
            activations: [context_ms, shiftby_ms, logs_folder, model_file, groundtruth_folder, labels_touse, kinds_touse, activations_equalize_ratio, activations_max_sounds, mini_batch, batch_seed] + list(model_parameters.values()),
            cluster: [groundtruth_folder, cluster_algorithm, cluster_these_layers, pca_fraction_variance_to_retain, tsne_perplexity, tsne_exaggeration, umap_neighbors, umap_distance],
            visualize: [groundtruth_folder],
            accuracy: [logs_folder, precision_recall_ratios],
            freeze: [context_ms, logs_folder, model_file] + list(model_parameters.values()),
            ensemble: [context_ms, logs_folder, model_file] + list(model_parameters.values()),
            classify: [context_ms, shiftby_ms, logs_folder, model_file, wavcsv_files, labels_touse, prevalences],
            ethogram: [model_file, wavcsv_files],
            misses: [wavcsv_files],
            compare: [logs_folder],
            congruence: [groundtruth_folder, validation_files, test_files, congruence_portion, congruence_convolve, congruence_measure],
            None: parameter_textinputs }

    groundtruth_update()
