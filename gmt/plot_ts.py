#!/usr/bin/env python2

"""
Created: 21 November 2016 from bash version (created 21 April 2016)
Purpose: Generate visualisations of timeslices (PNG)
Replacing Purpose 21-04-2016: Use bash instead of csh. Source vars from e3d.par
Replacing Purpose 21-11-2016: Use python instead of bash for greater flexibility.
Authors: Viktor Polak <viktor.polak@canterbury.ac.nz>

USAGE:
Execute with python: "$ ./plot_ts.py" or "$ python2 plot_ts.py"

USE CASES:
1. create timeslice png files
execute with 1st parameter being number of processes else interactive choice
2. create render with custom input same format as timeslice ('3f4')
execute with 1st parameter = single, second = source file

NOTES:
Processes seem to be IO limited (will see speedup with more processes with SSD).

ISSUES:
could validate parameters/check if folders/files exist.
"""

import multiprocessing as mp
import os
from shutil import copy, rmtree
import sys
from time import time

# params.py, params_plot.py and params_base.py in path
sys.path.insert(0, os.path.abspath(os.path.curdir))
# copy params_plot template from repo if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists('params_plot.py'):
    copy('%s/params_plot.template.py' % (script_dir), './params_plot.py')

import qcore.geo as geo
from params_base import *
srf_cnrs = srf_cnrs[0]
xyts_file = xyts_files[0]
import params_plot as plot
tsplot = plot.TS
base_dir = os.path.abspath(sim_dir)
import qcore.gmt as gmt
from qcore.xyts import XYTSFile

###
### DIRECTORIES
###
# render
png_dir = os.path.join(base_dir, 'PNG_timeslices')
# temporary working directories for gmt are within here
# prevent multiprocessing issues by isolating processes
gmt_temp = os.path.join(base_dir, 'GMT_WD_TS')
# clear
for out_dir in [png_dir, gmt_temp]:
    if os.path.isdir(out_dir):
        rmtree(out_dir)
    os.makedirs(out_dir)

###
### MULTIPROCESSING: first parameter set for non-interactive choice
###
if len(sys.argv) > 1:
    processes = int(sys.argv[1])
    print('Using %s processes from cmdline.' % (processes))
else:
    virtual_cores = mp.cpu_count()
    if virtual_cores > 8:
        # performance issues with more threads (disk IO)
        print('%d virtual cores found, limit to 8 if GMT installed on HDD' \
                % (virtual_cores))
        virtual_cores = 8
    print "Run on how many processes? [%d]: " % (virtual_cores),
    wanted_procs = raw_input()
    if wanted_procs == '':
        processes = virtual_cores
        print('Using default, %d processes.' % (virtual_cores))
    else:
        try:
            if int(wanted_procs) > 0:
                processes = int(wanted_procs)
                print('Running on %d processes.' % (processes))
        except ValueError:
            print('Invalid input. Exiting.')
            exit()


###
### RESOURCES
###
# dpi is very important for keeping adjusted map aspect ratios within same pixel
# 80 -> 720p, 120 -> 1080p
dpi = 120
# don't change following to keep 16:9 movie ratio
page_width = 16
page_height = 9
# space around map for titles, tick labels and scales etc
margin_top = 1.0
margin_bottom = 0.4
margin_left = 1.0
margin_right = 1.7
map_width = page_width - margin_left - margin_right
map_height = page_height - margin_top - margin_bottom

xyts = XYTSFile(xyts_file)
# simulation boundaries as as 2D list and GMT string
corners, cnr_str = xyts.corners(gmt_format = True)

# region to plot, sites to display
if plot.region != None:
    x_min, x_max, y_min, y_max, region_sites = \
            get_region(plot.region, as_components = True)
else:
    # use timeslice domain
    print('Using timeslice domain as plotting region.')
    x_min = min([xy[0] for xy in corners])
    x_max = max([xy[0] for xy in corners])
    y_min = min([xy[1] for xy in corners])
    y_max = max([xy[1] for xy in corners])
# extend to fit in map area
ll_region = (x_min, x_max, y_min, y_max)
map_width, map_height, ll_region = \
        gmt.fill_space(map_width, map_height, ll_region, \
                proj = 'M', dpi = dpi, wd = gmt_temp)
x_min, x_max, y_min, y_max = ll_region
# avg lon/lat (midpoint of plotting region)
ll_avg = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
# extend map to cover margins
if tsplot.border != None:
    borderless = True
    map_width_a, map_height_a, borderless_region = gmt.fill_margins( \
            ll_region, map_width, dpi, left = margin_left, \
            right = margin_right, top = margin_top, bottom = margin_bottom)
else:
    borderless = False

###
### PLOTTING STARTS HERE - TEMPLATE
###
######################################################

print('========== CREATING TEMPLATE ==========')
cpt_overlay = '%s/motion.cpt' % (gmt_temp)
mask = '%s/modelmask.grd' % (gmt_temp)
template_bottom = '%s/bottom.ps' % (gmt_temp)
template_top = '%s/top.ps' % (gmt_temp)

###
### create resources that are used throughout the process
###
t0 = time()
# AUTOPARAMS - sites
if tsplot.sites == None:
    region_sites = []
elif tsplot.sites == 'auto':
    if x_max - x_min > 3:
        region_sites = gmt.sites_major
    else:
        region_sites = gmt.sites.keys()
elif tsplot.sites == 'major':
    region_sites = gmt.sites_major
elif tsplot.sites == 'all':
    region_sites = gmt.sites.keys()
else:
    region_sites = tsplot.sites
# AUTOPARAMS - tick labels
if tsplot.major_tick == None:
    tsplot.major_tick, tsplot.minor_tick = \
            gmt.auto_tick(x_min, x_max, map_width)
elif tsplot.minor_tick == None:
    tsplot.minor_tick = tsplot.tick_major / 5.
# AUTOPARAMS - overlay spacing
if tsplot.grd_dx == None or tsplot.grd_dy == None:
    # TODO: work out x, y spacing considering rotation of data
    tsplot.grd_dx = '%sk' % (xyts.dx / 2.0)
    tsplot.grd_dy = tsplot.grd_dx
# AUTOPARAMS - colour scale
if tsplot.cpt_inc == None or tsplot.cpt_max == None:
    pgv_path = '%s/PGV.bin' % (gmt_temp)
    xyts.pgv(pgvout = pgv_path)
    tsplot.cpt_inc, tsplot.cpt_max = \
            gmt.xyv_cpt_range(pgv_path, \
            my_inc = tsplot.cpt_inc, my_max = tsplot.cpt_max)[1:3]
# AUTOPARAMS - convergence limit
if tsplot.convergence_limit == None:
    tsplot.convergence_limit = tsplot.cpt_inc * 0.2
# AUTOPARAMS - low cutoff
if tsplot.lowcut == 'auto':
    tsplot.lowcut = tsplot.cpt_max * 0.02
# AUTOPARAMS - title text generation
plot.vel_model = plot.vel_model.replace('<HH>', str(xyts.hh))
# overlay colour scale
gmt.makecpt(tsplot.cpt, cpt_overlay, tsplot.cpt_min, tsplot.cpt_max, \
        inc = tsplot.cpt_inc, invert = tsplot.cpt_inv, \
        bg = tsplot.cpt_bg, fg = tsplot.cpt_fg)
# simulation area mask
geo.path_from_corners(corners = corners, min_edge_points = 100, \
        output = '%s/sim.modelpath_hr' % (gmt_temp))
gmt.grd_mask('%s/sim.modelpath_hr' % (gmt_temp), mask, \
        dx = tsplot.grd_dx, dy = tsplot.grd_dy, region = ll_region)
print('Created resources (%.2fs)' % (time() - t0))

###
### create a basemap template which all maps start with
###
t1 = time()
b = gmt.GMTPlot(template_bottom)
if borderless:
    b.spacial('M', borderless_region, sizing = map_width_a)
    # topo, water, overlay cpt scale
    b.basemap()
    # map margins are semi-transparent
    b.background(map_width_a, map_height_a, \
            colour = tsplot.border, spacial = True, \
            window = (margin_left, margin_right, margin_top, margin_bottom))
else:
    # background can be larger as whitespace is later cropped
    b.background(page_width, page_height, colour = 'white')
# leave space for left tickmarks and bottom colour scale
b.spacial('M', ll_region, sizing = map_width, \
        x_shift = margin_left, y_shift = margin_bottom)
if not borderless:
    # topo, water, overlay cpt scale
    b.basemap(topo_cpt = 'grey1')
# title, fault model and velocity model subtitles
b.text(ll_avg[0], y_max, plot.event_title, size = 20, dy = 0.6)
b.text(x_min, y_max, plot.fault_model, size = 14, align = 'LB', dy = 0.3)
b.text(x_min, y_max, plot.vel_model, size = 14, align = 'LB', dy = 0.1)
b.cpt_scale('R', 'B', cpt_overlay, tsplot.cpt_inc, tsplot.cpt_inc, \
        label = tsplot.cpt_legend, length = map_height, horiz = False, \
        pos = 'rel_out', align = 'LB', thickness = 0.3, dx = 0.3, \
        arrow_f = tsplot.cpt_max > 0, arrow_b = tsplot.cpt_min < 0)
# stations - split into real and virtual
with open(stat_file, 'r') as sf:
    stations = sf.readlines()
stations_real = []
stations_virtual = []
for i in xrange(len(stations)):
    if len(stations[i].split()[-1]) == 7:
        stations_virtual.append(stations[i])
    else:
        stations_real.append(stations[i])
b.points(''.join(stations_real), is_file = False, \
        shape = 't', size = 0.08, fill = None, \
        line = 'white', line_thickness = 0.8)
b.points(''.join(stations_virtual), is_file = False, \
        shape = 'c', size = 0.02, fill = 'black', line = None)
b.leave()
print('Created bottom template (%.2fs)' % (time() - t1))

###
### create map data which all maps will have on top
###
t2 = time()
t = gmt.GMTPlot(template_top, append = True)
# locations in NZ
t.sites(region_sites)
t.coastlines()
# simulation domain
t.path(cnr_str, is_file = False, split = '-', \
        close = True, width = '0.4p', colour = 'black')
# fault file - creating direct from SRF is slower
#t OK if only done in template - more reliable
t.fault(srf_cnrs, is_srf = False, plane_width = 0.5, \
        top_width = 1, hyp_width = 0.5)
# ticks on top otherwise parts of map border may be drawn over
t.ticks(major = tsplot.major_tick, minor = tsplot.minor_tick, sides = 'ws')
t.leave()
print('Created top template (%.2fs)' % (time() - t2))

###
### estimate time savings
###
t3 = time()
print('Time saved per timeslice: ~%.2fs' % (time() - t0))
print('Time saved over %d timeslices: ~%.2fs' \
        % (xyts.nt - xyts.t0, (xyts.nt - xyts.t0) * (time() - t0)))
print('========== TEMPLATE COMPLETE ==========')


###
### PLOTTING CONTINUES - TIME SLICE LOOPING
###
######################################################

def render_slice(n):
    t0 = time()

    # prepare resources in separate folder
    # prevents GMT IO errors on its conf/history files
    swd = '%s/ts%.4d' % (gmt_temp, n)
    os.makedirs(swd)
    # name of slice postscript
    ps = '%s/ts%.4d.ps' % (swd, n)

    # copy GMT setup and basefile
    copy('%s/gmt.conf' % (gmt_temp), swd)
    copy('%s/gmt.history' % (gmt_temp), swd)
    copy(template_bottom, ps)
    s = gmt.GMTPlot(ps, append = True)
    # add timeslice timestamp
    s.text(x_max, y_max, 't=%.2fs' % (n * xyts.dt), \
            align = 'RB', size = '14p', dy = 0.1)

    xyts.tslice_get(n, comp = tsplot.component, \
            outfile = '%s/ts%.4d.X' % (swd, n))
    s.overlay('%s/ts%.4d.X' % (swd, n), cpt_overlay, \
            dx = tsplot.grd_dx, dy = tsplot.grd_dy, \
            climit = tsplot.convergence_limit, \
            min_v = tsplot.lowcut, max_v = tsplot.highcut, crop_grd = mask, \
            contours = tsplot.cpt_inc, land_crop = tsplot.land_crop)

    # append top file
    s.leave()
    with open(ps, 'a') as c:
        with open(template_top, 'r') as t:
            c.write(t.read())
    s.enter()

    # add seismograms if wanted
    if os.path.exists(os.path.abspath(tsplot.seis_data)):
        s.seismo(os.path.abspath(tsplot.seis_data), n, \
                fmt = tsplot.seis_fmt, \
                colour = tsplot.seis_colour, \
                width = tsplot.seis_line)

    # create PNG
    s.finalise()
    s.png(dpi = dpi, out_dir = png_dir)

    print('timeslice %d complete in %.2fs' % (n, time() - t0))

###
### start rendering each timeslice
###
ts0 = time()
#for i in xrange(xyts.t0, xyts.nt - xyts.t0):
#    render_slice(i)
pool = mp.Pool(processes)
pool.map(render_slice, xrange(xyts.t0, xyts.nt - xyts.t0))
print('FINISHED TIMESLICE SEGMENT IN %.2fs' % (time() - ts0))
print('AVERAGE TIMESLICE TIME %.2fs' % \
        ((time() - ts0) / (xyts.nt - xyts.t0)))

# images -> animation
gmt.make_movie('%s/ts%%04d.png' % (png_dir), \
        os.path.join(base_dir, 'animation.m4v'), fps = 20, codec = 'libx264')

# temporary files can be quite large
rmtree(gmt_temp)
rmtree(png_dir)
