import os
import sys
from bokeh.models.widgets import RadioButtonGroup, TextInput, Button, Div, DateFormatter, TextAreaInput, Select, NumberFormatter, Slider, Toggle, ColorPicker, MultiSelect, Paragraph
from bokeh.models.formatters import CustomJSTickFormatter
from bokeh.models import ColumnDataSource, TableColumn, DataTable, LayoutDOM, Span, HoverTool
from bokeh.plotting import figure
from bokeh.transform import linear_cmap, stack
from bokeh.events import Tap, DoubleTap, PanStart, Pan, PanEnd, ButtonClick, MouseWheel
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Circle
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
import re

try:
    import av
except:
    pass

from bokeh import palettes
from itertools import cycle, product
import ast
from bokeh.core.properties import Instance, String, List, Float
from bokeh.util.compiler import TypeScript
import asyncio
from collections import OrderedDict

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

bokehlog = logging.getLogger("songexplorer") 

import model as M
import controller as C

class ScatterNd(LayoutDOM):

    __implementation__ = TypeScript("""
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"
import {div} from "core/dom"
import type {StyleSheetLike} from "core/dom"

declare namespace Plotly {
  class newPlot {
    constructor(el: HTMLElement | DocumentFragment, data: object, OPTIONS: object)
    update(data: object): void
  }
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
  declare model: ScatterNd

  initialize(): void {
    super.initialize()

    const url = "https://cdn.plot.ly/plotly-latest.min.js"
    const script = document.createElement("script")
    script.onload = () => this._init()
    script.async = false
    script.src = url
    document.head.appendChild(script)
  }

  // https://discourse.bokeh.org/t/migrating-wrapper-around-plotly-js-to-bokeh-v3/12159
  override stylesheets(): StyleSheetLike[] {
    return [
      ...super.stylesheets(),
      `
      .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        direction: ltr;
        font-family: "Open Sans", verdana, arial, sans-serif;
        margin: 0;
        padding: 0;
      }
      .js-plotly-plot .plotly input, .js-plotly-plot .plotly button {
        font-family: "Open Sans", verdana, arial, sans-serif;
      }
      .js-plotly-plot .plotly input:focus, .js-plotly-plot .plotly button:focus {
        outline: none;
      }
      .js-plotly-plot .plotly a {
        text-decoration: none;
      }
      .js-plotly-plot .plotly a:hover {
        ext-decoration: none;
      }
      .js-plotly-plot .plotly .crisp {
        shape-rendering: crispEdges;
      }
      .js-plotly-plot .plotly .user-select-none {
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        -o-user-select: none;
        user-select: none;
      }
      .js-plotly-plot .plotly svg a {
        fill: #447adb;
      }
      .js-plotly-plot .plotly svg a:hover {
        fill: #3c6dc5;
      }
      .js-plotly-plot .plotly .main-svg {
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
      }
      .js-plotly-plot .plotly .main-svg .draglayer {
        pointer-events: all;
      }
      .js-plotly-plot .plotly .cursor-default {
        cursor: default;
      }
      .js-plotly-plot .plotly .cursor-pointer {
        cursor: pointer;
      }
      .js-plotly-plot .plotly .cursor-crosshair {
        cursor: crosshair;
      }
      .js-plotly-plot .plotly .cursor-move {
        cursor: move;
      }
      .js-plotly-plot .plotly .cursor-col-resize {
        cursor: col-resize;
      }
      .js-plotly-plot .plotly .cursor-row-resize {
        cursor: row-resize;
      }
      .js-plotly-plot .plotly .cursor-ns-resize {
        cursor: ns-resize;
      }
      .js-plotly-plot .plotly .cursor-ew-resize {
        cursor: ew-resize;
      }
      .js-plotly-plot .plotly .cursor-sw-resize {
        cursor: sw-resize;
      }
      .js-plotly-plot .plotly .cursor-s-resize {
        cursor: s-resize;
      }
      .js-plotly-plot .plotly .cursor-se-resize {
        cursor: se-resize;
      }
      .js-plotly-plot .plotly .cursor-w-resize {
        cursor: w-resize;
      }
      .js-plotly-plot .plotly .cursor-e-resize {
        cursor: e-resize;
      }
      .js-plotly-plot .plotly .cursor-nw-resize {
        cursor: nw-resize;
      }
      .js-plotly-plot .plotly .cursor-n-resize {
        cursor: n-resize;
      }
      .js-plotly-plot .plotly .cursor-ne-resize {
        cursor: ne-resize;
      }
      .js-plotly-plot .plotly .cursor-grab {
        cursor: -webkit-grab;
        cursor: grab;
      }
      .js-plotly-plot .plotly .modebar {
        position: absolute;
        top: 2px;
        right: 2px;
      }
      .js-plotly-plot .plotly .ease-bg {
        -webkit-transition: background-color 0.3s ease 0s;
        -moz-transition: background-color 0.3s ease 0s;
        -ms-transition: background-color 0.3s ease 0s;
        -o-transition: background-color 0.3s ease 0s;
        transition: background-color 0.3s ease 0s;
      }
      .js-plotly-plot .plotly .modebar--hover > :not(.watermark) {
        opacity: 0;
        -webkit-transition: opacity 0.3s ease 0s;
        -moz-transition: opacity 0.3s ease 0s;
        -ms-transition: opacity 0.3s ease 0s;
        -o-transition: opacity 0.3s ease 0s;
        transition: opacity 0.3s ease 0s;
      }
      .js-plotly-plot .plotly:hover .modebar--hover .modebar-group {
        opacity: 1;
      }
      .js-plotly-plot .plotly .modebar-group {
        float: left;
        display: inline-block;
        box-sizing: border-box;
        padding-left: 8px;
        position: relative;
        vertical-align: middle;
        white-space: nowrap;
      }
      .js-plotly-plot .plotly .modebar-btn {
        position: relative;
        font-size: 16px;
        padding: 3px 4px;
        height: 22px;
        /* display: inline-block; including this breaks 3d interaction in .embed mode. Chrome bug? */
        cursor: pointer;
        line-height: normal;
        box-sizing: border-box;
      }
      .js-plotly-plot .plotly .modebar-btn svg {
        position: relative;
        top: 2px;
      }
      .js-plotly-plot .plotly .modebar.vertical {
        display: flex;
        flex-direction: column;
        flex-wrap: wrap;
        align-content: flex-end;
        max-height: 100%;
      }
      .js-plotly-plot .plotly .modebar.vertical svg {
        top: -1px;
      }
      .js-plotly-plot .plotly .modebar.vertical .modebar-group {
        display: block;
        float: none;
        padding-left: 0px;
        padding-bottom: 8px;
      }
      .js-plotly-plot .plotly .modebar.vertical .modebar-group .modebar-btn {
        display: block;
        text-align: center;
      }
      .js-plotly-plot .plotly [data-title] {
        /**
         * tooltip body
         */
      }
      .js-plotly-plot .plotly [data-title]:before, .js-plotly-plot .plotly [data-title]:after {
        position: absolute;
        -webkit-transform: translate3d(0, 0, 0);
        -moz-transform: translate3d(0, 0, 0);
        -ms-transform: translate3d(0, 0, 0);
        -o-transform: translate3d(0, 0, 0);
        transform: translate3d(0, 0, 0);
        display: none;
        opacity: 0;
        z-index: 1001;
        pointer-events: none;
        top: 110%;
        right: 50%;
      }
      .js-plotly-plot .plotly [data-title]:hover:before, .js-plotly-plot .plotly [data-title]:hover:after {
        display: block;
        opacity: 1;
      }
      .js-plotly-plot .plotly [data-title]:before {
        content: "";
        position: absolute;
        background: transparent;
        border: 6px solid transparent;
        z-index: 1002;
        margin-top: -12px;
        border-bottom-color: #69738a;
        margin-right: -6px;
      }
      .js-plotly-plot .plotly [data-title]:after {
        content: attr(data-title);
        background: #69738a;
        color: white;
        padding: 8px 10px;
        font-size: 12px;
        line-height: 12px;
        white-space: nowrap;
        margin-right: -18px;
        border-radius: 2px;
      }
      .js-plotly-plot .plotly .vertical [data-title]:before, .js-plotly-plot .plotly .vertical [data-title]:after {
        top: 0%;
        right: 200%;
      }
      .js-plotly-plot .plotly .vertical [data-title]:before {
        border: 6px solid transparent;
        border-left-color: #69738a;
        margin-top: 8px;
        margin-right: -30px;
      }

      .plotly-notifier {
        font-family: "Open Sans", verdana, arial, sans-serif;
        position: fixed;
        top: 50px;
        right: 20px;
        z-index: 10000;
        font-size: 10pt;
        max-width: 180px;
      }
      .plotly-notifier p {
        margin: 0;
      }
      .plotly-notifier .notifier-note {
        min-width: 180px;
        max-width: 250px;
        border: 1px solid #fff;
        z-index: 3000;
        margin: 0;
        background-color: #8c97af;
        background-color: rgba(140, 151, 175, 0.9);
        color: #fff;
        padding: 10px;
        overflow-wrap: break-word;
        word-wrap: break-word;
        -ms-hyphens: auto;
        -webkit-hyphens: auto;
        hyphens: auto;
      }
      .plotly-notifier .notifier-close {
        color: #fff;
        opacity: 0.8;
        float: right;
        padding: 0 5px;
        background: none;
        border: none;
        font-size: 20px;
        font-weight: bold;
        line-height: 20px;
      }
      .plotly-notifier .notifier-close:hover {
        color: #444;
        text-decoration: none;
        cursor: pointer;
      }
      `,
    ]
  }

  ndims() {
    if (this.model.dots_source.get(this.model.dz).length==0) {
      return 0 }
    else if (isNaN(this.model.dots_source.get(this.model.dz)[0] as number)) {
      return 2 }
    return 3
  }

  get_dots_data() {
    return {x: this.model.dots_source.get(this.model.dx),
            y: this.model.dots_source.get(this.model.dy),
            z: this.model.dots_source.get(this.model.dz),
            text: this.model.dots_source.get(this.model.dl),
            marker: {
              color: this.model.dots_source.get(this.model.dc),
              size: this.model.dot_size_source.get(this.model.ds)[0],
              opacity: this.model.dot_alpha_source.get(this.model.da)[0],
            }
           };
  }

  set_circle_fuchsia_data2() {
    if (this.model.circle_fuchsia_source.get(this.model.cx).length==0) {
      OPTIONS2.shapes[0].x0 = 0
      OPTIONS2.shapes[0].y0 = 0
      OPTIONS2.shapes[0].x1 = 0
      OPTIONS2.shapes[0].y1 = 0
    } else {
      OPTIONS2.shapes[0].line.color = this.model.circle_fuchsia_source.get(this.model.cc)[0] as string
      let x = this.model.circle_fuchsia_source.get(this.model.cx)[0] as number
      let y = this.model.circle_fuchsia_source.get(this.model.cy)[0] as number
      let r = this.model.circle_fuchsia_source.get(this.model.cr)[0] as number
      OPTIONS2.shapes[0].x0 = x-r
      OPTIONS2.shapes[0].y0 = y-r
      OPTIONS2.shapes[0].x1 = x- -r
      OPTIONS2.shapes[0].y1 = y- -r
    }
  }

  get_circle_fuchsia_data3() {
    if (this.model.circle_fuchsia_source.get(this.model.cx).length==0) {
      return {type: 'mesh3d',
              x:[0], y:[0], z:[0],
             };
    } else {
      let radius = this.model.circle_fuchsia_source.get(this.model.cr)[0]
      return {type: 'mesh3d',
              // @ts-ignore
              x: xicosphere.map(x=>x*radius+this.model.circle_fuchsia_source.get(this.model.cx)[0]),
              // @ts-ignore
              y: yicosphere.map(x=>x*radius+this.model.circle_fuchsia_source.get(this.model.cy)[0]),
              // @ts-ignore
              z: zicosphere.map(x=>x*radius+this.model.circle_fuchsia_source.get(this.model.cz)[0]),
             }; }
  }

  private _init(): void {
    const wrapper_el = div({style: {width: "100%", height: "100%"}})
    this.shadow_el.append(wrapper_el)
    new Plotly.newPlot(wrapper_el,
                       [{alphahull: 1.0,
                         opacity: 0.2,
                        },
                        {hovertemplate: "%{text}<extra></extra>",
                         mode: 'markers',
                        }],
                       [{xaxis: { visible: false }, yaxis: { visible: false } }])

    this.connect(this.model.dots_source.change, () => {
      let new_data = this.get_dots_data()
      let N = this.ndims()
      if (N==2) {
        this.set_circle_fuchsia_data2()
        // @ts-ignore
        Plotly.update(wrapper_el, {type: '', x:[[]], y:[[]], z:[[]]}, OPTIONS2, [0]);
        // @ts-ignore
        Plotly.update(wrapper_el,
                      {type: 'scatter',
                       x: [new_data['x']], y: [new_data['y']],
                       text: [new_data['text']],
                       marker: new_data['marker'] },
                      OPTIONS2,
                      [1]);
      }
      else if (N==3) {
        // @ts-ignore
        Plotly.update(wrapper_el, {type: 'mesh3d', x:[[]], y:[[]], z:[[]]}, OPTIONS3, [0]);
        // @ts-ignore
        Plotly.update(wrapper_el,
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
      Plotly.restyle(wrapper_el, { marker: new_data['marker'] }, [1]);
    });

    this.connect(this.model.dot_alpha_source.change, () => {
      let new_data = this.get_dots_data()
      // @ts-ignore
      Plotly.restyle(wrapper_el, { marker: new_data['marker'] }, [1]);
    });

    // @ts-ignore
    wrapper_el.on('plotly_click', (data) => {
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
        Plotly.relayout(wrapper_el, OPTIONS2); }
      else if (N==3) {
        let new_data = this.get_circle_fuchsia_data3()
        // @ts-ignore
        Plotly.restyle(wrapper_el,
                       {x: [new_data['x']], y: [new_data['y']], z: [new_data['z']],
                        color: this.model.circle_fuchsia_source.get(this.model.cc)[0]},
                       [0]); }
    });
  }

  get child_models(): LayoutDOM[] { return [] }
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
  declare properties: ScatterNd.Props
  declare __view_type__: ScatterNdView

  constructor(attrs?: Partial<ScatterNd.Attrs>) { super(attrs) }

  static __name__ = "ScatterNd"

  static {
    this.prototype.default_view = ScatterNdView

    this.define<ScatterNd.Props>(({Str, Any, Ref}) => ({
      cx: [ Str ],
      cy: [ Str ],
      cz: [ Str ],
      cr: [ Str ],
      cc: [ Str ],
      dx: [ Str ],
      dy: [ Str ],
      dz: [ Str ],
      dl: [ Str ],
      dc: [ Str ],
      ds: [ Str ],
      da: [ Str ],
      click_position: [ Any ],
      circle_fuchsia_source: [ Ref(ColumnDataSource) ],
      dots_source: [ Ref(ColumnDataSource) ],
      dot_size_source: [ Ref(ColumnDataSource) ],
      dot_alpha_source: [ Ref(ColumnDataSource) ],
    }))
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

def cluster_initialize():
    cluster_file = os.path.join(groundtruth_folder.value,'cluster.npz')
    if not os.path.isfile(cluster_file):
        bokehlog.info("ERROR: "+cluster_file+" not found")
        return False
    npzfile = np.load(cluster_file, allow_pickle=True)

    labels_touse.value = ','.join(npzfile['labels_touse'])
    kinds_touse.value = ','.join(npzfile['kinds_touse'])
    if bokeh_document:
        bokeh_document.add_next_tick_callback(lambda: _cluster_initialize())
    else:
        _cluster_initialize()
    
    return True

def _cluster_initialize():
    global precomputed_dots
    global p_cluster_xmax, p_cluster_ymax, p_cluster_zmax
    global p_cluster_xmin, p_cluster_ymin, p_cluster_zmin

    cluster_file = os.path.join(groundtruth_folder.value,'cluster.npz')
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

    cluster_isnotnan = [not np.isnan(x[0]) and not np.isnan(x[1]) \
                        for x in M.clustered_activations[layer0]]

    M.nlayers = len(M.clustered_activations)
    M.ndcluster = np.shape(M.clustered_activations[layer0])[1]
    cluster_dots.data.update(dx=[], dy=[], dz=[], dl=[], dc=[])
    cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])

    M.layers = ["input"]+["hidden #"+str(i) for i in range(1,M.nlayers-1)]+["output"]
    M.species = set([x.split('-')[0]+'-' for x in npzfile['labels_touse'] if '-' in x])
    M.species |= set([''])
    M.species = natsorted(list(M.species))
    M.words = set(['-'+x.split('-')[1] for x in npzfile['labels_touse'] if '-' in x])
    M.words |= set([''])
    M.words = natsorted(list(M.words))
    M.nohyphens = set([x for x in npzfile['labels_touse'] if '-' not in x])
    M.nohyphens |= set([''])
    M.nohyphens = natsorted(list(M.nohyphens))
    M.kinds = set([x['kind'] for x in M.clustered_sounds])
    M.kinds |= set([''])
    M.kinds = natsorted(list(M.kinds))

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
                            colors = [M.label_colors[nohyphen] for b in bidx if b]
                        else:
                            colors = [M.label_colors[x['label']] \
                                      if x['label'] in M.label_colors else "black" \
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

def cluster_update():
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

def cluster_reset():
    global precomputed_dots
    precomputed_dots = None
    M.clustered_activations = None

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
    cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])

def within_an_annotation(sound):
    if len(M.annotated_starts_sorted)>0:
        ifrom = np.searchsorted(M.annotated_starts_sorted, sound['ticks'][0],
                                side='right') - 1
        if 0 <= ifrom and ifrom < len(M.annotated_starts_sorted) and \
                    M.annotated_sounds[ifrom]['ticks'][1] >= sound['ticks'][1]:
            return ifrom
    return -1

def snippets_update(redraw_wavs):
    if M.isnippet>=0 and not np.isnan(M.xcluster) and not np.isnan(M.ycluster) \
                and (M.ndcluster==2 or not np.isnan(M.zcluster)):
        snippets_quad_fuchsia.data.update(
                left=[M.xsnippet*(M.snippets_gap_pix+M.snippets_pix)],
                right=[(M.xsnippet+1)*(M.snippets_gap_pix+M.snippets_pix)-
                       M.snippets_gap_pix],
                top=[-M.ysnippet*snippets_dy+1],
                bottom=[-M.ysnippet*snippets_dy - 1 - snippets_both])
    else:
        snippets_quad_fuchsia.data.update(left=[], right=[], top=[], bottom=[])

    if len(M.species)>0:
        isubset = np.where([M.species[M.ispecies] in x['label'] and
                            M.words[M.iword] in x['label'] and
                            (M.nohyphens[M.inohyphen]=="" or \
                             M.nohyphens[M.inohyphen]==x['label']) and
                            (M.kinds[M.ikind]=="" or \
                             M.kinds[M.ikind]==x['kind']) for x in M.clustered_sounds])[0]
    else:
        isubset = []
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
                _, _, wavs = M.audio_read(os.path.join(groundtruth_folder.value, *thissound['file']),
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
                        window_length = int(round(M.spectrogram_length_sec[ichannel]*M.audio_tic_rate))
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
            xdata = list(range(ix*(M.snippets_gap_pix+M.snippets_pix),
                              (ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix))
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
                        log_img = np.log10(1e-15 + gram_images[isnippet][idx][ilows[isnippet][idx] : 1+ihighs[isnippet][idx], :])
                        clip_vals = np.percentile(log_img, M.spectrogram_clip)
                        np.where(log_img, log_img<clip_vals[0], clip_vals[0])
                        np.where(log_img, log_img>clip_vals[1], clip_vals[1])
                        snippets_gram_sources[isnippet][idx].data.update(
                                im = [log_img],
                                x = [xdata[0]],
                                y = [-iy*snippets_dy - 1 - snippets_both + len(M.snippets_spectrogram) - 1 - idx],
                                dw = [xdata[-1] - xdata[0] + xdata[1] - xdata[0]],
                                dh = [2/len(M.snippets_spectrogram)])
                    else:
                        snippets_gram_sources[isnippet][idx].data.update(im=[], x=[], y=[], dw=[], dh=[])
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

def nparray2base64mp4(filename, start_sec, stop_sec, npad_sec):
    frame_rate, video_shape, video_dtype, video_data = M.video_read(filename)

    start_frame = np.ceil(start_sec * frame_rate).astype(int)
    stop_frame = np.floor(stop_sec * frame_rate).astype(int)
    npad_frame = np.round(npad_sec * frame_rate).astype(int)

    fid=io.BytesIO()
    container = av.open(fid, mode='w', format='mp4')

    stream = container.add_stream('h264', rate=frame_rate)
    stream.width = video_shape[1]
    stream.height = video_shape[2]
    stream.pix_fmt = 'yuv420p'

    black_frame = np.zeros(video_shape[1:], dtype=video_dtype)
    for iframe in range(npad_frame, 0):
        frame = av.VideoFrame.from_ndarray(black_frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for iframe in range(start_frame, stop_frame):
        frame = av.VideoFrame.from_ndarray(np.array(video_data[iframe]), format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for iframe in range(npad_frame):
        frame = av.VideoFrame.from_ndarray(black_frame, format='rgb24')
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

def ___context_update(start_sec, stop_sec, frame_rate, context_sound):
    if video_toggle.active:
        video_slider.visible = True
        video_slider.start = np.ceil(start_sec * frame_rate) / frame_rate
        video_slider.end = np.floor(stop_sec * frame_rate) / frame_rate
        video_slider.step = 1/frame_rate
        midpoint_tics = (context_sound['ticks'][0] + context_sound['ticks'][1]) / 2
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

def __context_update(wavi, context_sound, istart_bounded, ilength, npad_sec):
    start_sec = istart_bounded / M.audio_tic_rate
    stop_sec = (istart_bounded+ilength) / M.audio_tic_rate

    if video_toggle.active:
        video_toggle.button_type="primary"
        sound_basename=context_sound['file'][1]
        sound_dirname=os.path.join(groundtruth_folder.value, context_sound['file'][0])
        vidfile = M.video_findfile(sound_dirname, sound_basename)
        if not vidfile:
            bokehlog.info("ERROR: video file corresponding to "+os.path.join(*context_sound['file'])+" not found")
    else:
        vidfile = None
    if vidfile:
        base64vid, height, width, frame_rate = nparray2base64mp4(os.path.join(sound_dirname,
                                                                              vidfile),
                                                                 start_sec, stop_sec, npad_sec)
        labelcounts.styles = {'overflow-y':'hidden', 'overflow-x':'scroll'}
        video_div.styles = {'width':str(width)+'px', 'height':str(height)+'px'}
    else:
        frame_rate = 0
        base64vid = ""
        labelcounts.styles = {'overflow-y':'hidden', 'overflow-x':'scroll'}
        video_div.styles = {'width':'1px', 'height':'1px'}

    base64wav = nparray2base64wav(wavi, M.audio_tic_rate)
    load_multimedia_callback.code = C.load_multimedia_callback_code % (base64wav, base64vid)
    load_multimedia.text = str(np.random.random())

    if vidfile:
        bokeh_document.add_next_tick_callback(lambda: \
                ___context_update(start_sec, stop_sec, frame_rate, context_sound))

def _context_update(wavi, context_sound, istart_bounded, ilength, npad_sec):
    if video_toggle.active:
        video_toggle.button_type="warning"
    else:
        video_toggle.button_type="default"
    bokeh_document.add_next_tick_callback(lambda: \
            __context_update(wavi, context_sound, istart_bounded, ilength, npad_sec))

def context_update():
    global context_cache_file, context_cache_data

    istart = np.nan
    scales = [0]*len(M.context_waveform)
    ywav = [np.full(1,np.nan)]*len(M.context_waveform)
    xwav = [np.full(1,np.nan)]*len(M.context_waveform)
    gram_freq = [np.full(1,np.nan)]*len(M.context_spectrogram)
    gram_time = [np.full(1,np.nan)]*len(M.context_spectrogram)
    gram_image = [np.full((1,1),np.nan)]*len(M.context_spectrogram)
    yprob = [np.full(1,np.nan)]*len(M.used_labels)
    xprob = [np.full(1,np.nan)]*len(M.used_labels)
    ilow = [0]*len(M.context_spectrogram)
    ihigh = [1]*len(M.context_spectrogram)
    xlabel_used, tlabel_used = [], []
    xlabel_annotated, tlabel_annotated = [], []
    left_used, right_used = [], []
    left_annotated, right_annotated = [], []

    if M.context_sound:
        play.disabled=False
        video_toggle.disabled=False
        if M.gui_snippets_spectrogram or M.gui_context_spectrogram:
            spectrogram_length.disabled=False
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
        M.context_midpoint_tic = np.mean(M.context_sound['ticks'], dtype=int)
        istart = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
        if recordings.value != os.path.join(*M.context_sound['file']):
            M.user_changed_recording=False
        recordings.value = os.path.join(*M.context_sound['file'])
        wavfile = os.path.join(groundtruth_folder.value, *M.context_sound['file'])
        if context_cache_file != wavfile:
            context_cache_file = wavfile
            _, _, context_cache_data = M.audio_read(wavfile)
        M.file_nframes = np.shape(context_cache_data)[0]
        probs = [None]*len(M.used_labels)
        for ilabel,label in enumerate(M.used_labels):
            prob_wavfile = M.trim_ext(os.path.join(groundtruth_folder.value,
                                                   *M.context_sound['file']))+'-'+label+'.wav'
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
                wavi = context_cache_data[istart_bounded : istart_bounded+ilength, ichannel]
                npad_sec = 0
                if len(wavi)<M.context_width_tic+1:
                    npad = M.context_width_tic+1-len(wavi)
                    if istart<0:
                        wavi = np.concatenate((np.full((npad,),0, dtype=np.int16), wavi))
                        npad_sec = -npad / M.audio_tic_rate
                    if istart+M.context_width_tic>M.file_nframes:
                        wavi = np.concatenate((wavi, np.full((npad,),0, dtype=np.int16)))
                        npad_sec = +npad / M.audio_tic_rate
                M.context_data[ichannel] = wavi
                M.context_data_istart = istart_bounded

                if ichannel==0:
                    if bokeh_document: 
                        bokeh_document.add_next_tick_callback(lambda: \
                                _context_update(wavi,
                                                M.context_sound,
                                                istart_bounded,
                                                ilength,
                                                npad_sec))

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
                    xwav[idx]=[(istart+i*context_decimate_by) / M.audio_tic_rate / M.context_time_scale \
                               for i in range(len(wavi_trimmed))]

                if ichannel+1 in M.context_spectrogram:
                    idx = M.context_spectrogram.index(ichannel+1)
                    window_length = int(round(M.spectrogram_length_sec[ichannel]*M.audio_tic_rate))
                    gram_freq[idx], gram_time[idx], gram_image[idx] = \
                            spectrogram(wavi,
                                        fs=M.audio_tic_rate,
                                        window=M.spectrogram_window,
                                        nperseg=window_length,
                                        noverlap=round(window_length*M.spectrogram_overlap))
                    gram_time[idx] /= M.context_time_scale
                    ilow[idx] = np.argmin(np.abs(gram_freq[idx] - \
                                                 M.spectrogram_low_hz[ichannel]))
                    ihigh[idx] = np.argmin(np.abs(gram_freq[idx] - \
                                                  M.spectrogram_high_hz[ichannel]))

            for ilabel in range(len(M.used_labels)):
                if not isinstance(probs[ilabel], np.ndarray):  continue
                prob_istart = int(np.rint(istart_bounded*tic_rate_ratio))
                prob_istop = int(np.rint((istart_bounded+ilength)*tic_rate_ratio))
                probi = probs[ilabel][prob_istart : prob_istop : prob_decimate_by]
                if len(probi)<round(M.context_width_tic*tic_rate_ratio)+1:
                    npad = round(M.context_width_tic*tic_rate_ratio)+1-len(probi)
                    if istart<0:
                        probi = np.concatenate((np.full((npad,),0), probi))
                    else:
                        probi = np.concatenate((probi, np.full((npad,),0)))
                probi_trimmed = probi[:prob_pix]
                yprob[ilabel] = probi_trimmed / np.iinfo(np.int16).max
                xprob[ilabel]=[(prob_istart+i*prob_decimate_by)/prob_tic_rate \
                                 for i in range(len(probi_trimmed))]

            if M.context_spectrogram:
                p_spectrogram.yaxis.formatter = CustomJSTickFormatter(
                    args=dict(low_hz=[gram_freq[i][x] / M.context_freq_scale for i,x in enumerate(ilow)],
                              high_hz=[gram_freq[i][x] / M.context_freq_scale for i,x in enumerate(ihigh)]),
                    code="""
                         if (tick==0) {
                             return low_hz[low_hz.length-1] }
                         else if (tick == high_hz.length) {
                             return high_hz[0] }
                         else {
                             return low_hz[low_hz.length-tick-1] + "," + high_hz[high_hz.length-tick] }
                         """)

            ileft = np.searchsorted(M.used_starts_sorted, istart+M.context_width_tic)
            sounds_to_plot = set(range(0,ileft))
            iright = np.searchsorted(M.used_stops, istart, sorter=M.iused_stops_sorted)
            sounds_to_plot &= set([M.iused_stops_sorted[i] for i in \
                    range(iright, len(M.iused_stops_sorted))])
            delete_tapped_sound = False
            if M.context_sound not in M.used_sounds and \
                    M.context_sound['ticks'][0]<istart+M.context_width_tic and \
                    M.context_sound['ticks'][1]>istart:
                M.used_sounds.append(M.context_sound)
                sounds_to_plot |= set([len(M.used_sounds)-1])
                delete_tapped_sound = True

            tapped_wav_in_view = False
            M.remaining_isounds = []
            for isound in sounds_to_plot:
                if M.context_sound['file']!=M.used_sounds[isound]['file']:
                    continue
                L = np.max([istart, M.used_sounds[isound]['ticks'][0]])
                R = np.min([istart+M.context_width_tic,
                            M.used_sounds[isound]['ticks'][1]])
                if L>istart and R<istart+M.context_width_tic and \
                        M.used_sounds[isound]['label'] in M.state['labels']:
                    M.remaining_isounds.append(isound)
                xlabel_used.append((L+R)/2 / M.audio_tic_rate / M.context_time_scale)
                tlabel_used.append(M.used_sounds[isound]['kind']+'\n'+\
                              M.used_sounds[isound]['label'])
                left_used.append(L / M.audio_tic_rate / M.context_time_scale)
                right_used.append(R / M.audio_tic_rate / M.context_time_scale)
                if M.context_sound==M.used_sounds[isound]:
                    if M.context_waveform:
                        waveform_quad_fuchsia.data.update(
                                left=[L / M.audio_tic_rate / M.context_time_scale],
                                right=[R / M.audio_tic_rate / M.context_time_scale],
                                top=[1],
                                bottom=[0])
                    if M.context_spectrogram:
                        spectrogram_quad_fuchsia.data.update(
                                left=[L / M.audio_tic_rate / M.context_time_scale],
                                right=[R / M.audio_tic_rate / M.context_time_scale],
                                top=[len(M.context_spectrogram)],
                                bottom=[len(M.context_spectrogram)/2])
                    tapped_wav_in_view = True
            if delete_tapped_sound:
                M.used_sounds.pop()

            M.remaining_isounds = [i for i in M.remaining_isounds \
                                   if all([i==j or \
                                           M.used_sounds[i]['ticks'][0] > M.used_sounds[j]['ticks'][1] or \
                                           M.used_sounds[i]['ticks'][1] < M.used_sounds[j]['ticks'][0] \
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
                    if M.context_sound['file']!=M.annotated_sounds[isound]['file']:
                        continue

                    M.remaining_isounds = [i for i in M.remaining_isounds \
                                           if M.annotated_sounds[isound]['ticks'][0] > M.used_sounds[i]['ticks'][1] or \
                                              M.annotated_sounds[isound]['ticks'][1] < M.used_sounds[i]['ticks'][0]]
                        
                    L = np.max([istart, M.annotated_sounds[isound]['ticks'][0]])
                    R = np.min([istart+M.context_width_tic,
                                M.annotated_sounds[isound]['ticks'][1]])
                    xlabel_annotated.append((L+R) / 2 / M.audio_tic_rate / M.context_time_scale)
                    tlabel_annotated.append(M.annotated_sounds[isound]['label'])
                    left_annotated.append(L / M.audio_tic_rate / M.context_time_scale)
                    right_annotated.append(R / M.audio_tic_rate / M.context_time_scale)
    else:
        play.disabled=True
        video_toggle.disabled=True
        if M.gui_snippets_spectrogram or M.gui_context_spectrogram:
            spectrogram_length.disabled=True
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
                log_img = np.log10(1e-15 + gram_image[idx][ilow[idx] : 1+ihigh[idx], :])
                clip_vals = np.percentile(log_img, M.spectrogram_clip)
                np.place(log_img, log_img<clip_vals[0], clip_vals[0])
                np.place(log_img, log_img>clip_vals[1], clip_vals[1])
                spectrogram_source[idx].data.update(
                        im = [log_img],
                        x = [istart / M.audio_tic_rate / M.context_time_scale],
                        y = [len(M.context_spectrogram) - 1 - idx],
                        dw = [gram_time[idx][-1] + M.spectrogram_length_sec[ichannel] / M.context_time_scale * M.gui_spectrogram_overlap],
                        dh = [1])
            else:
                spectrogram_source[idx].data.update(im=[], x=[], y=[], dw=[], dh=[])
        if xwav and not np.isnan(xwav[0][-1]):
            spectrogram_range_source.data.update(x=[xwav[0][-1]])

    if M.probability_style=="lines":
        probability_source.data.update(xs=xprob, ys=yprob,
                                       colors=[M.label_colors[x] for x in M.used_labels],
                                       labels=list(M.used_labels))
    else:
        probability_source.data.update(**{'x'+str(i):(xprob[i] if i<len(xprob) else xprob[0] if len(xprob)>0 else [])
                                          for i in list(range(M.nlabels))},
                                       **{'y'+str(i):(yprob[i] if i<len(yprob) else [0]*len(yprob[0]) if len(yprob)>0 else [])
                                          for i in list(range(M.nlabels))})
        if len(xprob)>0 and len(xprob[0])>1:
            for g in probability_glyphs:
                g.glyph.width = xprob[0][1] - xprob[0][0]

    if M.context_waveform:
        waveform_quad_grey_used.data.update(left=left_used,
                                            right=right_used,
                                            top=[1]*len(left_used),
                                            bottom=[0]*len(left_used))
        waveform_quad_grey_annotated.data.update(left=left_annotated,
                                                 right=right_annotated,
                                                 top=[0]*len(left_annotated),
                                                 bottom=[-1]*len(left_annotated))
        waveform_label_source_used.data.update(x=xlabel_used,
                                               y=[1]*len(xlabel_used),
                                               text=tlabel_used)
        waveform_label_source_annotated.data.update(x=xlabel_annotated,
                                                    y=[-1]*len(xlabel_annotated),
                                                    text=tlabel_annotated)
    if M.context_spectrogram:
        spectrogram_quad_grey_used.data.update(left=left_used,
                                               right=right_used,
                                               top=[len(M.context_spectrogram)]*len(left_used),
                                               bottom=[len(M.context_spectrogram)/2]*len(left_used))
        spectrogram_quad_grey_annotated.data.update(left=left_annotated,
                                                    right=right_annotated,
                                                    top=[len(M.context_spectrogram)/2]*len(left_annotated),
                                                    bottom=[0]*len(left_annotated))
        spectrogram_label_source_used.data.update(x=xlabel_used,
                                                  y=[len(M.context_spectrogram)]*len(xlabel_used),
                                                  text=tlabel_used)
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
                                          for i in list(range(1,nlayers-1))],
                                        (str(nlayers-1), "output")]
    else:
        cluster_these_layers.options = []

def recordings_update():
    M.used_sounds = []
    if kinds_touse.value=="" or labels_touse.value=="":
        _wavfiles = []
        for ext in M.audio_read_exts():
            _wavfiles.extend(glob.glob("**/*"+ext,
                                      root_dir=groundtruth_folder.value, recursive=True))
        _wavfiles = list(filter(lambda x: 'oldfiles-' not in x and \
                                          'congruence-' not in x, _wavfiles))
        wavfiles = []
        for wavfile in _wavfiles:
            if len(M.audio_read_rec2ch(wavfile)) > 1:
                wavfiles.extend([wavfile+'-'+k for k in M.audio_read_rec2ch(wavfile).keys()])
            else:
                wavfiles.append(wavfile)
        for wavfile in wavfiles:
            M.used_sounds.append({'file': list(os.path.split(wavfile)),
                                  'ticks': [1, 1], 'kind': '', 'label': ''})
    elif M.dfs:
        wavfiles = set()
        kinds = kinds_touse.value.split(',')
        labels = labels_touse.value.split(',')
        for df,subdir in zip(M.dfs, M.subdirs):
            bidx = np.logical_and(np.array(df[3].apply(lambda x: x in kinds)),
                                  np.array(df[4].apply(lambda x: x in labels)))
            if any(bidx):
                M.used_sounds.extend(list(df[bidx].apply(lambda x:
                                                         {"file": [subdir,x[0]],
                                                          "ticks": [x[1],x[2]],
                                                          "kind": x[3],
                                                          "label": x[4]},
                                                         1)))
                wavfiles |= set(df[bidx].apply(lambda x: os.path.join(subdir,x[0]), 1))

    if M.used_sounds:
        M.used_starts_sorted = [x['ticks'][0] for x in M.used_sounds]
        isort = np.argsort(M.used_starts_sorted)
        M.used_sounds = [M.used_sounds[x] for x in isort]
        M.used_starts_sorted = [M.used_starts_sorted[x] for x in isort]
        M.used_stops = [x['ticks'][1] for x in M.used_sounds]
        M.iused_stops_sorted = np.argsort(M.used_stops)

        recordings.options = sorted(list(wavfiles))
        M.used_recording2firstsound = {}
        for recording in recordings.options:
            M.used_recording2firstsound[recording] = \
                  next(filter(lambda x: os.path.join(*x[1]['file'])==recording,
                              enumerate(M.used_sounds)))[0]
        recordings.options = [""] + recordings.options
        if recordings.value != "":
            M.user_changed_recording=False
        recordings.value = ""
    else:
        M.used_sounds = None
        M.used_starts_sorted = M.used_stops = M.iused_stops_sorted = None
        M.used_recording2firstsound = {}
        recordings.options = []

    M.used_labels = set([x['label'] for x in M.used_sounds]) if M.used_sounds else []
    if M.probability_style=="bars":
        for ilabel,label in enumerate(M.used_labels):
            probability_glyphs[ilabel].name = label
    M.label_colors = { l:c for l,c in zip(M.used_labels, cycle(label_palette)) }
    M.isnippet = -1
    M.context_sound = None
    snippets_update(True)
    context_update()

def _groundtruth_update():
    M.dfs, M.subdirs = labelcounts_update()
    cluster_these_layers_update()
    cluster_reset()
    recordings_update()
    M.save_state_callback()
    recordings.disabled=False
    recordings.stylesheets = [""]
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
        recordings.stylesheets = [".bk-input:disabled { background-color: #FFA500; }"]
        groundtruth_folder_button.disabled=True
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(_groundtruth_update)
        else:
            _groundtruth_update()

def labels_touse_update(other=False, detect=False):
    theselabels_touse = [x.value for x in label_texts if x.value!='']
    if other and 'other' not in theselabels_touse:
        theselabels_touse.append('other')
    if detect:
        for detect_label in M.detect_labels:
            if detect_label not in theselabels_touse:
                theselabels_touse.append(detect_label)
    labels_touse.value=str.join(',',theselabels_touse)

def buttons_update():
    for button in wizard_buttons:
        button.button_type="success" if button==M.wizard else "default"
    for button in action_buttons:
        if button == leaveout:
            button.stylesheets = M.primary_style if button==M.action else M.default_style
        else:
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
        if button in action2parameterbuttons[M.action]:
            if button==prevalences_button:
                button.disabled = loss.value=='overlapped'
            else:
                button.disabled = False
        else:
            button.disabled = True
    okay=True if M.action else False
    for textinput in parameter_textinputs:
        if textinput in action2parametertextinputs[M.action]:
            if textinput==prevalences:
                prevalences.disabled = loss.value=='overlapped'
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
            elif textinput in cluster_parameters.values():
                thislogic = cluster_parameters_enable_logic[textinput]
                if thislogic:
                    textinput.disabled = cluster_parameters[thislogic[0]].value not in thislogic[1]
                else:
                    textinput.disabled = False
            elif textinput in augmentation_parameters.values():
                thislogic = augmentation_parameters_enable_logic[textinput]
                if thislogic:
                    textinput.disabled = augmentation_parameters[thislogic[0]].value not in thislogic[1]
                else:
                    textinput.disabled = False
            elif M.action is leaveout and textinput is kfold:
                textinput.disabled = leaveout.value != "omit some"
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
                elif textinput in cluster_parameters.values():
                    if cluster_parameters_required[textinput]:
                        okay=False
                elif textinput in augmentation_parameters.values():
                    if augmentation_parameters_required[textinput]:
                        okay=False
                else:
                    if textinput not in [test_files, restore_from]:
                        okay=False
        else:
            textinput.disabled=True
    if M.action==classify and \
            loss.value=='exclusive' and prevalences.value!='' and labels_touse.value=='':
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
        names=[os.path.basename(x) + (os.sep if os.path.isdir(x) else '') for x in files]
    else:
        names=[x + (os.sep if os.path.isdir(x) else '') for x in files]
    sizes=[os.path.getsize(f) for f in files]
    dates=[datetime.fromtimestamp(os.path.getmtime(f)) for f in files]
    file_dialog_source.selected.indices = []
    file_dialog_source.data.update(names=names, sizes=sizes, dates=dates)

def labelcounts_update():
    dfs, subdirs = [], []
    if not os.path.isdir(groundtruth_folder.value):
        labelcounts.text = ""
        return dfs, subdirs

    def _labelcounts_update(curdir):
        for entry in os.listdir(curdir):
            if os.path.isdir(os.path.join(curdir, entry)):
                if not re.fullmatch('congruence-[0-9]{8}T[0-9]{6}', entry) and \
                   not re.fullmatch('oldfiles-[0-9]{8}T[0-9]{6}', entry):
                    _labelcounts_update(os.path.join(curdir, entry))
            elif entry.endswith('.csv'):
                filepath = os.path.join(curdir, entry)
                if os.path.getsize(filepath) > 0:
                    try:
                        df = pd.read_csv(filepath, header=None, index_col=False)
                    except:
                        bokehlog.info("WARNING: "+entry+" is not in the correct format")
                    if 5<=len(df.columns)<=6:
                        dfs.append(df)
                        subdirs.append(curdir[len(groundtruth_folder.value):].lstrip(os.path.sep))
                    else:
                        bokehlog.info("WARNING: "+entry+" is not in the correct format")
    _labelcounts_update(groundtruth_folder.value)

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

def model_summary_update():
    model_settings = {'nlabels': len(labels_touse.value.split(',')),
                      'time_units': M.time_units,
                      'freq_units': M.freq_units,
                      'time_scale': M.time_scale,
                      'freq_scale': M.freq_scale,
                      'audio_tic_rate': M.audio_tic_rate,
                      'audio_nchannels': M.audio_nchannels,
                      'video_frame_rate': M.video_frame_rate,
                      'video_frame_width': M.video_frame_width,
                      'video_frame_height': M.video_frame_height,
                      'video_channels': [int(x)-1 for x in M.video_channels.split(',')],
                      'parallelize': int(parallelize.value),
                      'batch_size': int(mini_batch.value),
                      'context': float(context.value) }
    tf.keras.backend.clear_session()
    out = io.StringIO()
    try:
        thismodel = M.model.create_model(model_settings,
                                         {k:v.value for k,v in model_parameters.items()},
                                         out)
    except Exception as e:
        print(e, file=out)
        thismodel =  tf.keras.Model(inputs=[], outputs=[], name="error")
    model_summary.value = out.getvalue()+'\n'
    def update_model_summary(x):
        if not x.isspace():
            model_summary.value += x + '\n'
    thismodel.summary(line_length = int(M.gui_width_pix/18), print_fn = update_model_summary)

def init(_bokeh_document):
    global bokeh_document, configuration_file
    global p_cluster, cluster_dots, precomputed_dots, dot_size_cluster, dot_alpha_cluster, cluster_circle_fuchsia, label_palette, circle_radius, dot_size, dot_alpha
    global p_snippets, snippet_palette, snippets_dy, snippets_both, snippets_label_sources_clustered, snippets_label_sources_annotated, snippets_wave_sources, snippets_wave_glyphs, snippets_gram_sources, snippets_gram_glyphs, snippets_quad_grey, snippets_quad_fuchsia
    global p_waveform, waveform_span_red, waveform_quad_grey_used, waveform_quad_grey_annotated, waveform_quad_grey_pan, waveform_quad_fuchsia, waveform_source, waveform_glyph, waveform_label_source_used, waveform_label_source_annotated
    global p_spectrogram, spectrogram_span_red, spectrogram_quad_grey_used, spectrogram_quad_grey_annotated, spectrogram_quad_grey_pan, spectrogram_quad_fuchsia, spectrogram_source, spectrogram_glyph, spectrogram_label_source_used, spectrogram_label_source_annotated, spectrogram_range_source, spectrogram_length
    global p_probability, probability_span_red, probability_source
    global which_layer, which_species, which_word, which_nohyphen, which_kind
    global color_picker
    global zoom_width, zoom_offset, zoomin, zoomout, reset, panleft, panright, allleft, allout, allright, firstlabel, nextlabel, prevlabel, lastlabel
    global save_indicator, nsounds_per_label_buttons, label_texts
    global load_multimedia, play, video_slider, load_multimedia_callback, play_callback, video_slider_callback, video_toggle, video_div
    global undo, redo, remaining
    global recordings
    global detect, misses, train, leaveout, xvalidate, mistakes, activations, cluster, visualize, accuracy, freeze, ensemble, classify, ethogram, compare, congruence
    global status_ticker, waitfor, deletefailures
    global file_dialog_source, configuration_contents
    global logs_folder_button, logs_folder, model_file_button, model_file, wavcsv_files_button, wavcsv_files, groundtruth_folder_button, groundtruth_folder, validation_files_button, test_files_button, validation_files, test_files, labels_touse_button, labels_touse, kinds_touse_button, kinds_touse, prevalences_button, prevalences, delete_ckpts, copy, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, generalize, tunehyperparameters, findnovellabels, examineerrors, testdensely, doit, nsteps, restore_from, save_and_validate_period, validate_percentage, mini_batch, kfold, activations_equalize_ratio, activations_max_sounds, cluster_these_layers, precision_recall_ratios, congruence_portion, congruence_convolve, congruence_measure, context, parallelize, shiftby, optimizer, loss, learning_rate, nreplicates, batch_seed, weights_seed, file_dialog_string, file_dialog_table, readme_contents, model_summary, labelcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs, status_ticker_update, status_ticker_pre, status_ticker_post
    global detect_parameters, detect_parameters_enable_logic, detect_parameters_required, detect_parameters_partitioned, detect_parameters_width
    global doubleclick_parameters, doubleclick_parameters_enable_logic, doubleclick_parameters_required
    global model_parameters, model_parameters_enable_logic, model_parameters_required, model_parameters_partitioned, model_parameters_width
    global cluster_parameters, cluster_parameters_enable_logic, cluster_parameters_required, cluster_parameters_partitioned, cluster_parameters_width
    global augmentation_parameters, augmentation_parameters_enable_logic, augmentation_parameters_required, augmentation_parameters_partitioned, augmentation_parameters_width
    global context_cache_file, context_cache_data

    context_cache_file, context_cache_data = None, None

    bokeh_document = _bokeh_document

    M.cluster_circle_color = M.cluster_circle_color

    if '#' in M.label_palette:
      label_palette = ast.literal_eval(M.label_palette)
    else:
      label_palette = getattr(palettes, M.label_palette)

    snippet_palette = getattr(palettes, M.snippets_colormap)

    dot_size_cluster = ColumnDataSource(data=dict(ds=[M.state["dot_size"]]))
    dot_alpha_cluster = ColumnDataSource(data=dict(da=[M.state["dot_alpha"]]))

    cluster_dots = ColumnDataSource(data=dict(dx=[], dy=[], dz=[], dl=[], dc=[]))
    cluster_circle_fuchsia = ColumnDataSource(data=dict(cx=[], cy=[], cz=[], cr=[], cc=[]))
    p_cluster = ScatterNd(dx='dx', dy='dy', dz='dz', dl='dl', dc='dc',
                          dots_source=cluster_dots,
                          cx='cx', cy='cy', cz='cz', cr='cr', cc='cc',
                          click_position=[0,0],
                          circle_fuchsia_source=cluster_circle_fuchsia,
                          ds='ds',
                          dot_size_source=dot_size_cluster,
                          da='da',
                          dot_alpha_source=dot_alpha_cluster)
    p_cluster.on_change("click_position", lambda a,o,n: C.cluster_tap_callback(n))

    precomputed_dots = None

    snippets_dy = 2*((len(M.snippets_waveform)>0) + (len(M.snippets_spectrogram)>0))
    snippets_both = 2*((len(M.snippets_waveform)>0) and (len(M.snippets_spectrogram)>0))

    p_snippets = figure(background_fill_color='#FFFFFF', toolbar_location=None)
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
            snippets_gram_sources[ixy][idx] = ColumnDataSource(data=dict(im=[], x=[], y=[], dw=[], dh=[]))
            snippets_gram_glyphs[ixy][idx] = p_snippets.image(
                    image='im', x='x', y='y', dw='dw', dh='dh',
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
             for i in list(range(M.snippets_nx*M.snippets_ny))]
    ydata = [-(i//M.snippets_nx*snippets_dy-1)
             for i in list(range(M.snippets_nx*M.snippets_ny))]
    text = ['' for i in list(range(M.snippets_nx*M.snippets_ny))]
    snippets_label_sources_clustered = ColumnDataSource(data=dict(x=xdata, y=ydata, text=text))
    p_snippets.text('x', 'y', source=snippets_label_sources_clustered, text_font_size='6pt',
                    text_baseline='top',
                    text_color='black' if M.snippets_waveform else 'white')

    xdata = [(i%M.snippets_nx)*(M.snippets_gap_pix+M.snippets_pix)
             for i in list(range(M.snippets_nx*M.snippets_ny))]
    ydata = [-(i//M.snippets_nx*snippets_dy+1+snippets_both)
             for i in list(range(M.snippets_nx*M.snippets_ny))]
    text_annotated = ['' for i in list(range(M.snippets_nx*M.snippets_ny))]
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

    p_waveform = figure(width=M.gui_width_pix,
                        height=M.context_waveform_height_pix,
                        background_fill_color='#FFFFFF', toolbar_location=None)
    p_waveform.toolbar.active_drag = None
    p_waveform.grid.visible = False
    if M.context_spectrogram:
        p_waveform.xaxis.visible = False
    else:
        p_waveform.xaxis.axis_label = "Time ("+M.context_time_units+")"
    p_waveform.yaxis.axis_label = ""
    p_waveform.yaxis.ticker = []
    p_waveform.x_range.range_padding = p_waveform.y_range.range_padding = 0.0
    p_waveform.y_range.start = -1
    p_waveform.y_range.end = 1

    waveform_span_red = Span(location=0, dimension='height', line_color='red')
    p_waveform.add_layout(waveform_span_red)
    waveform_span_red.visible=False

    waveform_quad_grey_used = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_waveform.quad('left','right','top','bottom',source=waveform_quad_grey_used,
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

    waveform_label_source_used = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_waveform.text('x', 'y', source=waveform_label_source_used,
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

    p_spectrogram = figure(width=M.gui_width_pix,
                           height=M.context_spectrogram_height_pix,
                           background_fill_color='#FFFFFF', toolbar_location=None)
    p_spectrogram.toolbar.active_drag = None
    p_spectrogram.x_range.range_padding = p_spectrogram.y_range.range_padding = 0
    p_spectrogram.xgrid.visible = False
    p_spectrogram.ygrid.visible = True
    p_spectrogram.xaxis.axis_label = "Time ("+M.context_time_units+")"
    p_spectrogram.yaxis.axis_label = 'Frequency (' + M.context_freq_units + ')'
    p_spectrogram.yaxis.ticker = list(range(1+len(M.context_spectrogram)))

    spectrogram_source = [None]*len(M.context_spectrogram)
    spectrogram_glyph = [None]*len(M.context_spectrogram)
    for idx in range(len(M.context_spectrogram)):
        spectrogram_source[idx] = ColumnDataSource(data=dict(im=[], x=[], y=[], dw=[], dh=[]))
        spectrogram_glyph[idx] = p_spectrogram.image(image='im', x='x', y='y', dw='dw', dh='dh',
                                                     source=spectrogram_source[idx],
                                                     palette=M.spectrogram_colormap,
                                                     level="image")
    spectrogram_range_source = ColumnDataSource(data=dict(x=[]))
    spectrogram_range_glyph = p_spectrogram.scatter('x', marker='square', y=0, size=1, alpha=0,
                                                   source=spectrogram_range_source)

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

    spectrogram_quad_grey_used = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_spectrogram.quad('left','right','top','bottom',source=spectrogram_quad_grey_used,
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

    spectrogram_label_source_used = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_spectrogram.text('x', 'y', source=spectrogram_label_source_used,
                   text_font_size='6pt', text_align='center', text_baseline='top',
                   text_line_height=0.8, level='underlay', text_color='white')
    spectrogram_label_source_annotated = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_spectrogram.text('x', 'y', source=spectrogram_label_source_annotated,
                   text_font_size='6pt', text_align='center', text_baseline='bottom',
                   text_line_height=0.8, level='underlay', text_color='white')

    if M.probability_style=="lines":
        TOOLTIPS = """
            <div><div><span style="color:@colors;">@labels,$y</span></div></div>
        """
    else:
        TOOLTIPS = ""

    p_probability = figure(width=M.gui_width_pix,
                           height=M.context_probability_height_pix,
                           tooltips=TOOLTIPS, toolbar_location=None,
                           background_fill_color='#FFFFFF')
    p_probability.toolbar.active_drag = None
    p_probability.grid.visible = False
    p_probability.yaxis.axis_label = "Probability"
    p_probability.x_range.range_padding = p_probability.y_range.range_padding = 0.0
    p_probability.y_range.start = 0
    p_probability.y_range.end = 1
    p_probability.xaxis.visible = False

    if M.probability_style=="lines":
        probability_source = ColumnDataSource(data=dict(xs=[], ys=[], colors=[], labels=[]))
        global probability_glyph 
        probability_glyph = p_probability.multi_line(xs='xs', ys='ys',
                                                     source=probability_source, color='colors')
    else:
        xs = {'x'+str(i):[] for i in list(range(M.nlabels))}
        ys = {'y'+str(i):[] for i in list(range(M.nlabels))}
        probability_source = ColumnDataSource(data=xs|ys)
        global probability_glyphs
        probability_glyphs = []
        for i in range(M.nlabels):
            probability_glyphs.append(p_probability.vbar(
                    x='x'+str(i),
                    bottom=stack(*['y'+str(j) for j in list(range(i))]),
                    top=stack(*['y'+str(j) for j in list(range(i+1))]),
                    line_width = 0,
                    fill_color=label_palette[i],
                    source=probability_source))
            p_probability.add_tools(HoverTool(renderers=[probability_glyphs[i]],
                                              tooltips=[("", "$name, @y"+str(i))]))

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
                           disabled=True,
                           sizing_mode='stretch_width')
    circle_radius.on_change("value_throttled", C.circle_radius_callback)

    dot_size = Slider(start=1, end=24, step=1,
                      value=M.state["dot_size"],
                      title="dot size",
                      disabled=True, sizing_mode='stretch_width')
    dot_size.on_change("value", C.dot_size_callback)

    dot_alpha = Slider(start=0.01, end=1.0, step=0.01, \
                       value=M.state["dot_alpha"], \
                       title="dot alpha", \
                       disabled=True,
                       sizing_mode='stretch_width')
    dot_alpha.on_change("value", C.dot_alpha_callback)

    cluster_update()

    if M.gui_snippets_spectrogram or M.gui_context_spectrogram:
        spectrogram_length = TextInput(value=','.join([str(x / M.time_scale)
                                                       for x in M.spectrogram_length_sec]), \
                                       title="length ("+M.time_units+")", \
                                       disabled=False,
                                       sizing_mode='stretch_width')
        spectrogram_length.on_change('value', C.spectrogram_window_callback)

    zoom_width = TextInput(value=str(M.context_width_sec / M.time_scale),
                             title="width ("+M.time_units+"):",
                             disabled=True,
                             sizing_mode='stretch_width')
    zoom_width.on_change("value", C.zoom_width_callback)

    zoom_offset = TextInput(value=str(M.context_offset_sec / M.time_scale),
                            title="offset ("+M.time_units+"):",
                            disabled=True,
                            sizing_mode='stretch_width')
    zoom_offset.on_change("value", C.zoom_offset_callback)

    zoomin = Button(label='\u2191', disabled=True, align="center")
    zoomin.on_click(C.zoomin_callback)

    zoomout = Button(label='\u2193', disabled=True, align="center")
    zoomout.on_click(C.zoomout_callback)

    reset = Button(label='\u25ef', disabled=True)
    reset.on_click(C.zero_callback)

    panleft = Button(label='\u2190', disabled=True, align="center")
    panleft.on_click(C.panleft_callback)

    panright = Button(label='\u2192', disabled=True, align="center")
    panright.on_click(C.panright_callback)

    allleft = Button(label='\u21e4', disabled=True)
    allleft.on_click(C.allleft_callback)

    allout = Button(label='\u2913', disabled=True, align="center")
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

    save_indicator = Button(label='0', align=("center","end"))

    nsounds_per_label_callbacks=[]
    nsounds_per_label_buttons=[]
    label_callbacks=[]
    label_texts=[]

    for i in range(M.nlabels):
        nsounds_per_label_callbacks.append(lambda i=i: C.nsounds_per_label_callback(i))
        nsounds_per_label_buttons.append(Button(label='0', css_classes=['hide-label']))
        nsounds_per_label_buttons[-1].on_click(nsounds_per_label_callbacks[-1])

        label_callbacks.append(lambda a,o,n,i=i: C.label_callback(n,i))
        label_texts.append(TextInput(value=M.state['labels'][i],
                                     css_classes=['hide-label'], height=32))
        label_texts[-1].on_change("value", label_callbacks[-1])

    C.nsounds_per_label_callback(M.ilabel)

    load_multimedia = Paragraph(visible=False)
    load_multimedia_callback = CustomJS(code=C.load_multimedia_callback_code % ("",""))
    load_multimedia.js_on_change('text', load_multimedia_callback)

    play = Button(label='play', disabled=True, align=("center","end"))
    play_callback = CustomJS(args=dict(waveform_span_red=waveform_span_red,
                                       spectrogram_span_red=spectrogram_span_red,
                                       probability_span_red=probability_span_red,
                                       p=p_waveform if M.context_waveform else p_spectrogram),
                             code=C.play_callback_code)
    play.js_on_event(ButtonClick, play_callback)
    play.on_change('disabled', lambda a,o,n: reset_video())

    video_toggle = Toggle(label='video', active=False, disabled=True, align=("center","end"))
    video_toggle.on_click(lambda x: context_update())

    video_div = Div(text="""<video id="context_video"></video>""",
                    styles={'width':'1px'})

    video_slider = Slider(title="", show_value=False, visible=False,
                          start=0, end=1, value=0, step=1)
    video_slider_callback = CustomJS(args=dict(waveform_span_red=waveform_span_red,
                                               spectrogram_span_red=spectrogram_span_red,
                                               probability_span_red=probability_span_red),
                                     code=C.video_slider_callback_code)
    video_slider.js_on_change("value", video_slider_callback)

    undo = Button(label='undo', disabled=True, align=("center","end"))
    undo.on_click(C.undo_callback)

    redo = Button(label='redo', disabled=True, align=("center","end"))
    redo.on_click(C.redo_callback)

    remaining = Button(label='add remaining', disabled=True)
    remaining.on_click(C.remaining_callback)

    recordings = Select(title="recording:", height=48)
    recordings.on_change('value', C.recordings_callback)

    detect = Button(label='detect', width_policy='fit')
    detect.on_click(lambda: C.action_callback(detect, C.detect_actuate))

    misses = Button(label='misses', width_policy='fit')
    misses.on_click(lambda: C.action_callback(misses, C.misses_actuate))

    train = Button(label='train', width_policy='fit')
    train.on_click(lambda: C.action_callback(train, C.train_actuate))

    leaveout = Select(title="", value="omit one", margin = 4,
                      options=["omit one", "omit some", "omit all"],
                      width_policy='fit', stylesheets=M.default_style)
    leaveout.on_change('value',
                       lambda a,o,n: C.action_callback(leaveout, lambda: C.leaveout_actuate(n)))

    xvalidate = Button(label='x-validate', width_policy='fit')
    xvalidate.on_click(lambda: C.action_callback(xvalidate, C.xvalidate_actuate))

    mistakes = Button(label='mistakes', width_policy='fit')
    mistakes.on_click(lambda: C.action_callback(mistakes, C.mistakes_actuate))

    activations = Button(label='activations', width_policy='fit')
    activations.on_click(lambda: C.action_callback(activations, C.activations_actuate))

    cluster = Button(label='cluster', width_policy='fit')
    cluster.on_click(lambda: C.action_callback(cluster, C.cluster_actuate))

    visualize = Button(label='visualize', width_policy='fit')
    visualize.on_click(lambda: C.action_callback(visualize, C.visualize_actuate))

    accuracy = Button(label='accuracy', width_policy='fit')
    accuracy.on_click(lambda: C.action_callback(accuracy, C.accuracy_actuate))

    freeze = Button(label='freeze', width_policy='fit')
    freeze.on_click(lambda: C.action_callback(freeze, C.freeze_actuate))

    ensemble = Button(label='ensemble', width_policy='fit')
    ensemble.on_click(lambda: C.action_callback(ensemble, C.ensemble_actuate))

    classify = Button(label='classify', width_policy='fit')
    classify.on_click(C.classify_callback)

    ethogram = Button(label='ethogram', width_policy='fit')
    ethogram.on_click(lambda: C.action_callback(ethogram, C.ethogram_actuate))

    compare = Button(label='compare', width_policy='fit')
    compare.on_click(lambda: C.action_callback(compare, C.compare_actuate))

    congruence = Button(label='congruence', width_policy='fit')
    congruence.on_click(lambda: C.action_callback(congruence, C.congruence_actuate))

    status_ticker_pre="<div style='overflow:auto; white-space:nowrap; width:"+str(M.gui_width_pix-236)+"px'>status: "
    status_ticker_post="</div>"
    status_ticker = Div(text=status_ticker_pre+status_ticker_post)

    deletefailures = Toggle(label='delete failures', active=False, disabled=True)
    deletefailures.on_click(C.deletefailures_callback)

    waitfor = Toggle(label='wait for last job', active=False, disabled=True)
    waitfor.on_click(C.waitfor_callback)

    logs_folder_button = Button(label='logs folder:', min_width=110)
    logs_folder_button.on_click(C.logs_callback)
    logs_folder = TextInput(value=M.state['logs_folder'], title="", disabled=False,
                            sizing_mode="stretch_width")
    logs_folder.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    model_file_button = Button(label='checkpoint file:', min_width=110)
    model_file_button.on_click(C.model_callback)
    model_file = TextInput(value=M.state['model_file'], title="", disabled=False,
                           sizing_mode="stretch_width")
    model_file.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    wavcsv_files_button = Button(label='wav,csv files:', min_width=110)
    wavcsv_files_button.on_click(C.wavcsv_files_callback)
    wavcsv_files = TextInput(value=M.state['wavcsv_files'], title="", disabled=False,
                             sizing_mode="stretch_width")
    wavcsv_files.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    groundtruth_folder_button = Button(label='ground truth:', min_width=110)
    groundtruth_folder_button.on_click(C.groundtruth_callback)
    groundtruth_folder = TextInput(value=M.state['groundtruth_folder'], title="", disabled=False,
                                   sizing_mode="stretch_width")
    groundtruth_folder.on_change('value', lambda a,o,n: groundtruth_update())

    validation_files_button = Button(label='validation files:', min_width=110)
    validation_files_button.on_click(C.validationfiles_callback)
    validation_files = TextInput(value=M.state['validation_files'], title="", disabled=False,
                                 sizing_mode="stretch_width")
    validation_files.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    test_files_button = Button(label='test files:', min_width=110)
    test_files_button.on_click(C.test_files_callback)
    test_files = TextInput(value=M.state['test_files'], title="", disabled=False,
                           sizing_mode="stretch_width")
    test_files.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    labels_touse_button = Button(label='labels to use:', min_width=110)
    labels_touse_button.on_click(C.labels_touse_callback)
    labels_touse = TextInput(value=M.state['labels_touse'], title="", disabled=False,
                             sizing_mode="stretch_width")
    labels_touse.on_change('value', lambda a,o,n: C.touse_callback(n,labels_touse_button))

    kinds_touse_button = Button(label='kinds to use:', min_width=110)
    kinds_touse = TextInput(value=M.state['kinds_touse'], title="", disabled=False,
                            sizing_mode="stretch_width")
    kinds_touse.on_change('value', lambda a,o,n: C.touse_callback(n,kinds_touse_button))

    prevalences_button = Button(label='prevalences:')
    prevalences_button.on_click(C.prevalences_callback)
    prevalences = TextInput(value=M.state['prevalences'], title="", disabled=False,
                            sizing_mode="stretch_width")
    prevalences.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    delete_ckpts = Button(label='delete ckpts')
    delete_ckpts.on_click(lambda: C.action_callback(delete_ckpts, C.delete_ckpts_actuate))

    copy = Button(label='copy')
    copy.on_click(C.copy_callback)

    labelsounds = Button(label='label sounds', sizing_mode="stretch_width")
    labelsounds.on_click(lambda: C.wizard_callback(labelsounds))

    makepredictions = Button(label='make predictions', sizing_mode="stretch_width")
    makepredictions.on_click(lambda: C.wizard_callback(makepredictions))

    fixfalsepositives = Button(label='fix false positives', sizing_mode="stretch_width")
    fixfalsepositives.on_click(lambda: C.wizard_callback(fixfalsepositives))

    fixfalsenegatives = Button(label='fix false negatives', sizing_mode="stretch_width")
    fixfalsenegatives.on_click(lambda: C.wizard_callback(fixfalsenegatives))

    generalize = Button(label='test generalization', sizing_mode="stretch_width")
    generalize.on_click(lambda: C.wizard_callback(generalize))

    tunehyperparameters = Button(label='tune h-parameters', sizing_mode="stretch_width")
    tunehyperparameters.on_click(lambda: C.wizard_callback(tunehyperparameters))

    findnovellabels = Button(label='find novel labels', sizing_mode="stretch_width")
    findnovellabels.on_click(lambda: C.wizard_callback(findnovellabels))

    examineerrors = Button(label='examine errors', sizing_mode="stretch_width")
    examineerrors.on_click(lambda: C.wizard_callback(examineerrors))

    testdensely = Button(label='test densely', sizing_mode="stretch_width")
    testdensely .on_click(lambda: C.wizard_callback(testdensely))

    doit = Button(label='do it!', disabled=True)
    doit.on_click(C.doit_callback)

    nsteps = TextInput(value=M.state['nsteps'], title="# steps", disabled=False,
                       sizing_mode="stretch_width")
    nsteps.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    restore_from = TextInput(value=M.state['restore_from'], title="restore from",
                             disabled=False, sizing_mode="stretch_width")
    restore_from.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    save_and_validate_period = TextInput(value=M.state['save_and_validate_period'],
                                         title="validate period",
                                         disabled=False, sizing_mode='stretch_width')
    save_and_validate_period.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    validate_percentage = TextInput(value=M.state['validate_percentage'],
                                    title="validate %",
                                    disabled=False, sizing_mode='stretch_width')
    validate_percentage.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    mini_batch = TextInput(value=M.state['mini_batch'], title="mini-batch",
                                  disabled=False, sizing_mode="stretch_width")
    mini_batch.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    kfold = TextInput(value=M.state['kfold'], title="k-fold",
                      disabled=False, sizing_mode='stretch_width')
    kfold.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    activations_equalize_ratio = TextInput(value=M.state['activations_equalize_ratio'],
                                           title="equalize ratio",
                                           disabled=False, sizing_mode='stretch_width')
    activations_equalize_ratio.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    activations_max_sounds = TextInput(value=M.state['activations_max_sounds'], title="max sounds",
                                       disabled=False, sizing_mode='stretch_width')
    activations_max_sounds.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    precision_recall_ratios = TextInput(value=M.state['precision_recall_ratios'], title="P/Rs",
                                               disabled=False, sizing_mode='stretch_width')
    precision_recall_ratios.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))
    
    congruence_portion = Select(title="portion", height=48,
                                value=M.state['congruence_portion'],
                                options=["union", "intersection"],
                                sizing_mode='stretch_width')
    congruence_portion.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    congruence_convolve = TextInput(value=M.state['congruence_convolve'],
                                    title="convolve ("+M.time_units+")",
                                    disabled=False, sizing_mode='stretch_width')
    congruence_convolve.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))
    
    congruence_measure = Select(title="measure", height=48,
                                value=M.state['congruence_measure'],
                                options=["label", "tic", "both"],
                                sizing_mode='stretch_width')
    congruence_measure.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    context = TextInput(value=M.state['context'], title="context ("+M.time_units+")",
                        disabled=False, sizing_mode="stretch_width")
    context.on_change('value', lambda a,o,n: C.context_parallelize_callback(n))

    parallelize = TextInput(value=M.state['parallelize'], title="parallelize",
                        disabled=False, sizing_mode="stretch_width")
    parallelize.on_change('value', lambda a,o,n: C.context_parallelize_callback(n))

    shiftby = TextInput(value=M.state['shiftby'], title="shift by ("+M.time_units+")",
                        disabled=False, sizing_mode='stretch_width')
    shiftby.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    optimizer = Select(title="optimizer", height=48, value=M.state['optimizer'],
                       options=["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSProp", "SGD"],
                       sizing_mode="stretch_width")
    optimizer.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    loss = Select(title="loss", height=48, value=M.state['loss'],
                  options=["exclusive", "overlapped", "autoencoder"],
                  sizing_mode="stretch_width")
    loss.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))

    learning_rate = TextInput(value=M.state['learning_rate'], title="learning rate",
                                     disabled=False, sizing_mode='stretch_width')
    learning_rate.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    V = sys.modules[__name__]

    def get_callback(f, msu):
        def callback(a,o,n):
            f(n,M,V,C) if f else C.generic_parameters_callback(n)
            if msu:  model_summary_update()
        return callback

    def parse_plugin_parameters(Mparameters, width, msu=False):
        parameters = OrderedDict()
        parameters_enable_logic = {}
        parameters_required = {}
        for parameter in Mparameters:
            if parameter[2]=='':
                thisparameter = TextInput(value=M.state[parameter[0]],
                                          title=parameter[1],
                                          disabled=False,
                                          sizing_mode='stretch_width')
            else:
                thisparameter = Select(value=M.state[parameter[0]],
                                       title=parameter[1],
                                       options=parameter[2],
                                       height=48,
                                       sizing_mode='stretch_width')
            thisparameter.on_change('value', get_callback(parameter[6], msu))
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

        return parameters, parameters_enable_logic, parameters_required, parameters_partitioned, parameters_width

    detect_parameters, detect_parameters_enable_logic, detect_parameters_required, detect_parameters_partitioned, detect_parameters_width = parse_plugin_parameters(M.detect_parameters, 2)
    doubleclick_parameters, doubleclick_parameters_enable_logic, doubleclick_parameters_required, _, _ = parse_plugin_parameters(M.doubleclick_parameters, 1)
    model_parameters, model_parameters_enable_logic, model_parameters_required, model_parameters_partitioned, model_parameters_width = parse_plugin_parameters(M.model_parameters, 5, True)
    cluster_parameters, cluster_parameters_enable_logic, cluster_parameters_required, cluster_parameters_partitioned, cluster_parameters_width = parse_plugin_parameters(M.cluster_parameters, 1)
    augmentation_parameters, augmentation_parameters_enable_logic, augmentation_parameters_required, augmentation_parameters_partitioned, augmentation_parameters_width = parse_plugin_parameters(M.augmentation_parameters, 2)

    file_dialog_source = ColumnDataSource(data=dict(names=[], sizes=[], dates=[]))
    file_dialog_source.selected.on_change('indices', C.file_dialog_callback)

    file_dialog_columns = [
        TableColumn(field="names", title="Name"),
        TableColumn(field="sizes", title="Size", \
                    formatter=NumberFormatter(format="0 b")),
        TableColumn(field="dates", title="Date", \
                    formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
    ]
    file_dialog_table = DataTable(source=file_dialog_source, \
                                  columns=file_dialog_columns, \
                                  height=800-20*len(detect_parameters_partitioned),
                                  index_position=None,
                                  fit_columns=False,
                                  sizing_mode='stretch_width')

    configuration_contents = TextAreaInput(rows=20, max_length=50000,
                                           styles={'font-family': 'Courier New'},
                                           disabled=True, sizing_mode='stretch_width')
    if M.configuration_file:
        with open(M.configuration_file, 'r') as fid:
            configuration_contents.value = fid.read()


    cluster_these_layers = MultiSelect(title='layers', \
                                       value=M.state['cluster_these_layers'], \
                                       options=[],
                                       sizing_mode='stretch_both')
    cluster_these_layers.on_change('value', lambda a,o,n: C.generic_parameters_callback(''))
    cluster_these_layers_update()

    nreplicates = TextInput(value=M.state['nreplicates'], title="# replicates",
                            disabled=False, sizing_mode='stretch_width')
    nreplicates.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    batch_seed = TextInput(value=M.state['batch_seed'], title="batch seed",
                           disabled=False, sizing_mode='stretch_width')
    batch_seed.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    weights_seed = TextInput(value=M.state['weights_seed'], title="weights seed",
                             disabled=False, sizing_mode="stretch_width")
    weights_seed.on_change('value', lambda a,o,n: C.generic_parameters_callback(n))

    file_dialog_string = TextInput(disabled=False, sizing_mode='stretch_width')
    file_dialog_string.on_change("value", C.file_dialog_path_callback)
    file_dialog_string.value = M.state['file_dialog_string']
     
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','README.md'), 'r', encoding='utf-8') as fid:
        contents = fid.read()
    html = markdown.markdown(contents, extensions=['tables','toc'])
    readme_contents = Div(text=html, height=1140, width=M.gui_width_pix//2,
                          stylesheets=["p { margin: 10px; }"],
                          styles={'overflow':'scroll', 'display':'flex', 'flex-direction':'column'})

    model_summary = TextAreaInput(rows=49-3*len(model_parameters_partitioned),
                                  max_length=50000,
                                  styles={'font-family': 'Courier New'},
                                  disabled=True, sizing_mode='stretch_width')

    labelcounts = Div(text="",
                      styles={'overflow-y':'hidden', 'overflow-x':'scroll'})
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
        leaveout,
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
        congruence,
        delete_ckpts])

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
        cluster_these_layers,
        precision_recall_ratios,
        congruence_portion,
        congruence_convolve,
        congruence_measure,
        nreplicates,
        batch_seed,
        weights_seed,

        context,
        parallelize,
        shiftby,
        optimizer,
        loss,
        learning_rate] +

        list(detect_parameters.values()) +
        list(model_parameters.values()) +
        list(cluster_parameters.values()) +
        list(augmentation_parameters.values()))

    wizard2actions = {
            labelsounds: [detect, train, activations, cluster, visualize, delete_ckpts],
            makepredictions: [train, accuracy, freeze, classify, ethogram, delete_ckpts],
            fixfalsepositives: [activations, cluster, visualize, delete_ckpts],
            fixfalsenegatives: [detect, misses, activations, cluster, visualize, delete_ckpts],
            generalize: [leaveout, accuracy, delete_ckpts],
            tunehyperparameters: [xvalidate, accuracy, compare, delete_ckpts],
            findnovellabels: [detect, train, activations, cluster, visualize, delete_ckpts],
            examineerrors: [detect, mistakes, activations, cluster, visualize, delete_ckpts],
            testdensely: [detect, activations, cluster, visualize, classify, ethogram, congruence, delete_ckpts],
            None: action_buttons }

    action2parameterbuttons = {
            detect: [wavcsv_files_button],
            train: [logs_folder_button, groundtruth_folder_button, labels_touse_button, test_files_button, kinds_touse_button],
            leaveout: [logs_folder_button, groundtruth_folder_button, validation_files_button, test_files_button, labels_touse_button, kinds_touse_button],
            xvalidate: [logs_folder_button, groundtruth_folder_button, test_files_button, labels_touse_button, kinds_touse_button],
            mistakes: [groundtruth_folder_button],
            activations: [logs_folder_button, model_file_button, groundtruth_folder_button, labels_touse_button, kinds_touse_button],
            cluster: [groundtruth_folder_button],
            visualize: [groundtruth_folder_button],
            accuracy: [logs_folder_button],
            delete_ckpts: [logs_folder_button],
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
            train: [context, parallelize, shiftby, optimizer, loss, learning_rate, nreplicates, batch_seed, weights_seed, logs_folder, groundtruth_folder, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, validate_percentage, mini_batch] + list(model_parameters.values()) + list(augmentation_parameters.values()),
            leaveout: [context, parallelize, shiftby, optimizer, loss, learning_rate, batch_seed, weights_seed, logs_folder, groundtruth_folder, validation_files, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch, kfold] + list(model_parameters.values()) + list(augmentation_parameters.values()),
            xvalidate: [context, parallelize, shiftby, optimizer, loss, learning_rate, batch_seed, weights_seed, logs_folder, groundtruth_folder, test_files, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch, kfold] + list(model_parameters.values()) + list(augmentation_parameters.values()),
            mistakes: [groundtruth_folder],
            activations: [context, parallelize, shiftby, logs_folder, model_file, groundtruth_folder, labels_touse, kinds_touse, activations_equalize_ratio, activations_max_sounds, mini_batch, batch_seed] + list(model_parameters.values()),
            cluster: [groundtruth_folder] + list(cluster_parameters.values()),
            visualize: [groundtruth_folder],
            accuracy: [logs_folder, precision_recall_ratios, loss],
            delete_ckpts: [logs_folder],
            freeze: [context, parallelize, logs_folder, model_file, loss] + list(model_parameters.values()),
            ensemble: [context, parallelize, logs_folder, model_file] + list(model_parameters.values()),
            classify: [context, parallelize, shiftby, logs_folder, model_file, wavcsv_files, labels_touse, prevalences, loss],
            ethogram: [model_file, wavcsv_files],
            misses: [wavcsv_files],
            compare: [logs_folder, loss],
            congruence: [groundtruth_folder, validation_files, test_files, congruence_portion, congruence_convolve, congruence_measure],
            None: parameter_textinputs }

    groundtruth_update()
    model_summary_update()
