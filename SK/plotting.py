#!/usr/bin/env python
# Tiaan Bezuidenhout, 2020. For inquiries: bezmc93@gmail.com
# NB: REQUIRES Python 3

"""
Plotting tools.
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from bokeh.models import LinearColorMapper
from bokeh.plotting import figure

import SK.utils as ut
import SK.coordinates as co


def plot_known(w,known_coords):
    """
    Makes a cross on the localisation plot where a known source is located.
    w must be a WCS object pre-set with SK_coordinates.buildWCS
    """
    known_coords = known_coords.split(',')

    try:
        known_coords = [float(known_coords[0]), float(known_coords[1])]
    except:
        known_coords = SkyCoord(known_coords[0], known_coords[1], frame='icrs', 
                                unit=(u.hourangle, u.deg))
        known_coords = [known_coords.ra.deg, known_coords.dec.deg]

    known_px = w.all_world2pix([known_coords], 1)

    ax.scatter(known_px[0, 0],known_px[0, 1], c='cyan', marker='x', s=20, zorder=999)


def make_ticks(p2, array_width, array_height, w, fineness):
    """
    Adds ticks and labels in sky coordinates to the plot.
    """

    ticks = ut.getTicks(array_width, array_height, w, fineness)
    labels = co.pix2deg(ticks, w)
    ra_deg= list(np.around(labels[:, 0], 4))
    dec_deg = list(np.around(labels[:, 1], 4))

    tick_vals = list(np.arange(0,len(ra_deg)))
    xtick_keys = {}
    ytick_keys = {}
    for key in tick_vals:
        for i in range(0,len(ra_deg)):
            xtick_keys[key] = str(ra_deg[i])
            ytick_keys[key] = str(dec_deg[i])
            ra_deg.remove(ra_deg[i])
            dec_deg.remove(dec_deg[i])

            break

    p2.xaxis.major_label_overrides = xtick_keys
    p2.yaxis.major_label_overrides = ytick_keys


def likelihoodPlot(p2, w, loglikelihood, options):
    """
    Creates the localisation plot
    """
    
    likelihood = ut.norm_likelihood(loglikelihood)

    from bokeh.palettes import Blues256
    color = LinearColorMapper(palette = Blues256[::-1],low = np.min(likelihood), high = np.max(likelihood))
    p2.image(image=[likelihood], x=0, y=0, dw=likelihood.shape[1], dh=likelihood.shape[0], color_mapper = color,level='image')


    p2.x(np.where(likelihood==np.amax(likelihood))[1],
                np.where(likelihood==np.amax(likelihood))[0], size=15, color="red")


    # Printing location of maximum likelihood
    max_loc = np.where(loglikelihood==np.nanmax(loglikelihood))
    message = ut.printCoords(max_loc, w)
    # print(max_loc)
    if len(max_loc[0]) == 2:
        ax.axhline(max_loc[0], lw=1.2, c='#a8dadc', ls='--')
        ax.axvline(max_loc[1], lw=1.2, c='#a8dadc', ls='--')

    ## Calculating the interval values in 2D
    print('Error levels:')
    sigs = np.asarray(np.arange(1,options.nsig[0]+1))

    header = ['# Region file format: DS9 version 4.1',
          ('global color=blue dashlist=8 3 '
           'width=2 font="helvetica 10 '
           'normal roman" select=1 highlite=1 '
           'dash=0 fixed=0 edit=1 move=1 '
           'delete=1 include=1 source=1'),
          'fk5']

    x, y = np.meshgrid(np.linspace(0, likelihood.shape[1], likelihood.shape[1]), np.linspace(0, likelihood.shape[0], likelihood.shape[0]))
    for s in sigs:
        level, error = ut.calc_error(likelihood, s)

        if s == 1:
            ls = 'solid'
            lc = '#e63946'
            lw = 3
            c1 = p2.contour(x,y,likelihood, levels=[level], line_color=lc, line_dash=ls, line_width=lw)
        else:
            ls = 'dashed'
            lc = '#E65476'
            lw = 2

        p2.contour(x,y,likelihood, levels=[level], line_color=lc, line_dash=ls, line_width=lw)

        message += '<br>---- %i sigma error----<br>' % (s)
        message += ut.printError(max_loc, w, error, s)
        
    # for con in c1.collections[0].get_paths():
    #     xs = []
    #     ys = []
    #     v = con.vertices
    #     #print '-------'
    #     #print v

    #     xs = v[:,0]
    #     ys = v[:,1]

    #     v_degs = co.pix2deg((xs,ys),w)
    #     #print v_degs
    #     line = 'polygon'
    #     for q in range(0,len(v_degs)):
    #         line += ' ' + str(v_degs[q,0])
    #         line += ' ' + str(v_degs[q,1])
    #     header.append(line)
    # with open('FRB.reg', 'w') as rf:
    #     for listitem in header:
    #         rf.write('{}\n'.format(listitem))
    # rf.close()


    ## Making axis histograms

    p3 = figure(width=600, height=100, x_range=p2.x_range, title='Likelihood')
    p3.line(np.arange(0,likelihood.shape[1]),np.sum(likelihood, axis=0), line_color='black')
    p3.xaxis.major_label_text_font_size = '0pt'
    p3.background_fill_color = "#F7FBFF"
    p3.yaxis[0].ticker.desired_num_ticks = 3

    p4 = figure(width=100, height=600, y_range=p2.y_range)
    p4.line(np.sum(likelihood, axis=1),np.arange(0,likelihood.shape[0]), line_color='black')
    p4.yaxis.major_label_text_font_size = '0pt'
    p4.background_fill_color = "#F7FBFF"
    p4.xaxis[0].ticker.desired_num_ticks = 3

    p2.yaxis.axis_label = 'Dec (deg)'
    p2.xaxis.axis_label = 'RA (deg)'


    print('---------------------------------------------------------------')
    print(message)
    print('---------------------------------------------------------------')

    return p3,p4,message