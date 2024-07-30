#!/usr/bin/env python3

# to have songexplorer automatically convert the values stored in EPG .aq8
# files into voltages, export the binary data in some .DOx files as plain text
# .A0x files using the “Save measured data as ASCII” function in the stylet+
# software.  then use this script to generate a lookup table, and specify the
# path to the generated .npy file in the keyword arguments to the load-epg
# audio_read plugin in configuration.py.

# e.g. src/load-epg-make-lut.py <path-to-folder-of-DOx-and-A0x-files>

import sys
import os
import numpy as np

_, path2data = sys.argv

lut = np.empty((0,2))
for asciifile in filter(lambda x: os.path.splitext(x)[1].startswith('.A'),
                        os.listdir(path2data)):
    print(asciifile)
    asciidata = np.loadtxt(os.path.join(path2data, asciifile), delimiter=';')
    binaryfile = asciifile[:-3]+'D'+asciifile[-2:]
    binarydata = np.fromfile(os.path.join(path2data, binaryfile), dtype=np.uint32)
    this_lut = np.hstack((np.expand_dims(binarydata, axis=1), asciidata[:,[1]]))
    lut = np.unique(np.vstack((lut, np.unique(this_lut, axis=0))), axis=0)

isort = np.argsort(lut[:,0])
lut = lut[isort,:]

np.save(path2data+".npy", lut)
