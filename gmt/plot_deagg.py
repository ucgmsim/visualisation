#!/usr/bin/env python2
"""
Plots deagg data.

Requires:
numpy >= 1.13 for numpy.unique(axis)
gmt from qcore
"""

from argparse import ArgumentParser
from io import BytesIO
import math
import os
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
# requires fairly new version of numpy for axis parameter in np.unique
npv = map(int, np.__version__.split('.'))
if npv[0] < 1 or (npv[0] == 1 and npv[1] < 13):
    print('requires numpy >= 1.13')
    exit(1)

from qcore import gmt

X_LEN = 4.5
Y_LEN = 4.0
Z_LEN = 2.5
ROT = 30
TILT = 60
LEGEND_SPACE = 0.7
EPSILON_LEGEND_EXPAND = 1.25
EPSILON_COLOURS = ['215/38/3', '252/94/62', '252/180/158', '254/220/210', \
                   '217/217/255', '151/151/255', '0/0/255', '0/0/170']
EPSILON_LABELS = ['@~e@~<-2', '-2<@~e@~<-1', '-1<@~e@~<-0.5', '-0.5<@~e@~<0', \
                  '0<@~e@~<0.5', '0.5<@~e@~<1', '1<@~e@~<2', '2<@~e@~']
TYPE_LEGEND_EXPAND = 0.35
TYPE_COLOURS = ['blue', 'red', 'green']
TYPE_LABELS = ['A', 'B', 'DS']

###
### LOAD DATA
###
parser = ArgumentParser()
parser.add_argument('deagg_file', help = 'deagg file to plot')
parser.add_argument('--out-name', help = 'basename excluding extention', \
                    default = 'deagg')
parser.add_argument('--out-dir', help = 'directory to store output', \
                    default = '.')
parser.add_argument('--dpi', help = 'dpi of raster output', \
                    type = int, default = 300)
parser.add_argument('--mag-min', help = 'minimum magnitude', \
                    type = float, default = 5.0)
parser.add_argument('--mag-max', help = 'maximum magnitude', \
                    type = float, default = 9.0)
parser.add_argument('-z', help = '"epsilon" or "type"', default = 'epsilon')
args = parser.parse_args()
assert(os.path.exists(args.deagg_file))
assert(args.mag_min < args.mag_max)
if not os.path.exists(args.out_dir):
    try:
        os.makedirs(args.out_dir)
    except OSError:
        if not os.path.isdir(args.out_dir):
            raise
rrup_mag_z_c = np.loadtxt(args.deagg_file, skiprows = 4, usecols = (2, 1, 5, 4))
# modifications based on plot type selection
if args.z == 'type':
    t = np.loadtxt(args.deagg_file, skiprows = 4, usecols = (3), dtype = '|S2')
    colours = TYPE_COLOURS
    labels = TYPE_LABELS
    legend_expand = TYPE_LEGEND_EXPAND
else:
    colours = EPSILON_COLOURS
    labels = EPSILON_LABELS
    legend_expand = EPSILON_LEGEND_EXPAND

###
### PROCESS DATA
###
# x axis
x_max = max(rrup_mag_z_c[:, 0])
if x_max < 115:
    x_inc = 10
elif x_max < 225:
    x_inc = 20
elif x_max < 335:
    x_inc = 30
elif x_max < 445:
    x_inc = 40
else:
    x_inc = 50
dx = x_inc / 2.0
x_max = math.ceil(x_max / float(dx)) * dx

# y axis
y_min = args.mag_min
y_max = args.mag_max
if y_max - y_min < 5:
    y_inc = 0.5
else:
    y_inc = 1.0
dy = y_inc / 2.0

# bins to put data in
bins_x = (np.arange(int(x_max / dx)) + 1) * dx
bins_y = (np.arange(int((y_max - y_min) / dy)) + 1) * dy + y_min
bins_e = np.array([-2, -1, -0.5, 0, 0.5, 1, 2, \
                   max(3, np.max(rrup_mag_z_c[:, 2]) + 1)])

# convert data into bin indexes
rrup_mag_z_c[:, 0] = np.digitize(rrup_mag_z_c[:, 0], bins_x)
rrup_mag_z_c[:, 1] = np.digitize(rrup_mag_z_c[:, 1], bins_y)
if args.z == 'type':
    rrup_mag_z_c[:, 2] = np.float32(t == 'B') + np.float32(t == 'DS') * 2
else:
    rrup_mag_z_c[:, 2] = np.digitize(rrup_mag_z_c[:, 2], bins_e)

# combine duplicate bins
blocks = np.zeros(tuple(map(int, \
                  np.append(np.max(rrup_mag_z_c[:, :3], axis = 0) + 1, 2))))
unique = np.unique(rrup_mag_z_c[:, :3], axis = 0)
for rrup, mag, z in unique[unique[:, 2].argsort()]:
    # get base
    blocks[int(rrup), int(mag), int(z), 1] = \
            np.max(blocks[int(rrup), int(mag), :, 0])
    # current top = value + base
    value = np.add.reduce(rrup_mag_z_c[ \
            np.ix_(np.minimum.reduce( \
            rrup_mag_z_c[:, :3] == (rrup, mag, z), axis = 1), (3,))])
    if value:
        blocks[int(rrup), int(mag), int(z), 0] = \
                value + blocks[int(rrup), int(mag), int(z), 1]
del rrup_mag_z_c, unique

# move indexes into array
top, base = blocks.reshape((-1, 2)).T
cpt = np.tile(np.arange(blocks.shape[2]), np.prod(blocks.shape[0:2]))
y = np.tile(np.repeat(np.arange(blocks.shape[1]) * dy + 0.5 * dy + y_min, \
                      blocks.shape[2]), blocks.shape[0])
x = np.repeat(np.arange(blocks.shape[0]) * dx + 0.5 * dx, \
              np.prod(blocks.shape[1:3]))
gmt_rows = np.column_stack((x, y, top, cpt, base))
del x, y, top, cpt, base
# don't plot if top == 0
gmt_rows = np.delete(gmt_rows, \
                     np.argwhere(gmt_rows[:, 2] == 0).flatten(), axis = 0)

# z axis depends on max contribution tower
z_inc = int(math.ceil(np.max(np.add.reduce(blocks, axis = 2)) / 5.0))
z_max = z_inc * 5
del blocks

###
### PLOT AXES
###
wd = mkdtemp()
p = gmt.GMTPlot('%s.ps' % os.path.join(wd, args.out_name))
os.remove(os.path.join(wd, 'gmt.conf'))
# setup axes
p.spacial('X', (0, x_max, y_min, y_max, 0, z_max), \
        sizing = '%si/%si' % (X_LEN, Y_LEN), z = 'Z%si' % (Z_LEN), \
        p = '%s/%s' % (180 - ROT, 90 - TILT), x_shift = '5', y_shift = 5)
p.ticks_multi(['xa%s+lRupture Distance (km)' % (x_inc), \
               'ya%s+lMagnitude' % (y_inc), \
               'za%sg%s+l%%Contribution' % (z_inc, z_inc), \
               'wESnZ' ])
# GMT will not plot gridlines without box, manually add gridlines
gridlines = []
for z in xrange(z_inc, z_max + z_inc, z_inc):
    gridlines.append('0 %s %s\n0 %s %s\n%s %s %s' \
                     % (y_min, z, y_max, z, x_max, y_max, z))
gridlines.append('0 %s 0\n0 %s %s' % (y_max, y_max, z_max))
gridlines.append('%s %s 0\n%s %s %s' % (x_max, y_max, x_max, y_max, z_max))
p.path('\n>\n'.join(gridlines), is_file = False, width = '0.5p', z = True)

###
### PLOT CONTENTS
###
cpt = os.path.join(wd, 'z.cpt')
gmt.makecpt(','.join(colours), cpt, \
            0, len(colours), inc = 1, wd = wd)
gmt_in = BytesIO()
np.savetxt(gmt_in, gmt_rows, fmt = '%.6f')
p.points(gmt_in.getvalue(), is_file = False, z = True, line = 'black', \
        shape = 'o', size = '%si/%sib' % (float(X_LEN) / len(bins_x) - 0.05, \
                                          float(Y_LEN) / len(bins_x) - 0.05), \
        line_thickness = '0.5p', cpt = cpt)

###
### PLOT LEGEND
###
# x y diffs from start to end, alternatively run multiple GMT commands with -X
angle = math.radians(ROT)
map_width = math.cos(angle) * X_LEN + math.sin(angle) * Y_LEN
x_end = (X_LEN + math.cos(angle) * math.sin(angle) \
                 * (Y_LEN - math.tan(angle) * X_LEN)) / X_LEN \
                 * x_max * legend_expand
y_end = math.tan(angle) * x_end / x_max * X_LEN * (y_max - y_min) / Y_LEN
# x y diffs at start, alternatively set -D(dz)
x_shift = map_width * (legend_expand - 1) * -0.5
y_shift = (LEGEND_SPACE) / math.cos(math.radians(TILT)) \
          + X_LEN * math.sin(angle)
x0 = (y_shift * math.sin(angle) + x_shift * math.cos(angle)) * (x_max / X_LEN)
y0 = y_min + (-y_shift * math.cos(angle) + x_shift * math.sin(angle)) \
             * ((y_max - y_min) / Y_LEN)
# legend definitions
legend_boxes = []
legend_labels = []
for i, x in enumerate(np.arange(0, 1.01, 1.0 / (len(colours) - 1.0))):
    legend_boxes.append('%s %s %s %s' % (x0 + x * x_end, y0 + x * y_end, \
                                         z_inc / 2.0, i))
    legend_labels.append('%s 0 %s' % (x, labels[i]))
# cubes and labels of legend
p.points('\n'.join(legend_boxes), is_file = False, z = True, line = 'black', \
        shape = 'o', size = '%si/%sib0' % (Z_LEN / 10.0, Z_LEN / 10.0), \
        line_thickness = '0.5p', cpt = cpt, clip = False)
p.spacial('X', (0, 1, 0, 1), sizing = '%si/1i' % (map_width * legend_expand), \
          x_shift = '%si' % (x_shift), \
          y_shift = '-%si' % (LEGEND_SPACE + 0.2))
p.text_multi('\n'.join(legend_labels), is_file = False, justify = 'CT')

###
### SAVE
###
p.finalise()
p.png(portrait = True, background = 'white', \
      dpi = args.dpi, out_dir = args.out_dir, margin = [0.618, 1])
rmtree(wd)
