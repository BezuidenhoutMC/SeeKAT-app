#!/usr/bin/env python
# Tiaan Bezuidenhout, 2023. For inquiries: bezmc93@gmail.com

import argparse
import numpy as np
import sys
import os
from werkzeug.utils import secure_filename

import SK.utils as ut
import SK.coordinates as co
import SK.plotting as Splot

from flask import Flask, render_template, request, session
from flask_socketio import SocketIO
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.embed import components
from bokeh.resources import INLINE
np.seterr(divide='ignore', invalid='ignore')

def parseOptions(parser):
    '''Options:
    -f    Input file with each line a different CB detection.
        Should have 3 columns: RA (h:m:s), Dec (d:m:s), S/N
    -p    PSF of a CB in fits format
    --o    Fractional sensitivity level at which CBs are tiled to overlap
    --r    Resolution of PSF in units of arcseconds per pixel
    --n    Number of beams to consider when creating overlap contours. Will
        pick the specified number of beams with the highest S/N values.
    --nsig Sets the number of standard deviation contours drawn.
    --s Draws known coordinates onto the plot for comparison.
    --ticks Sets the spacing of ticks on the localisation plot.
    --clip Sets level below which CB PSF is set equal to zero.
    '''

    parser.add_argument('-f', dest='file', 
                nargs = 1, 
                type = str, 
                help="Detections file",
                required=False)
    parser.add_argument('-c', dest='config', 
                nargs = 1, 
                type = str, 
                help="Configuration (json) file",
                required=False)
    parser.add_argument('-p',dest='psf',
                nargs=1,
                type=str,
                help="PSF file",
                required=False)
    parser.add_argument('--o', dest='overlap',
                type = float,
                help = "Fractional sensitivity level at which the coherent beams overlap",
                default = 0.25,
                required = False)
    parser.add_argument('--r', dest='res',
                nargs = 1,
                type=float,
                help="Distance in arcseconds represented by one pixel of the PSF",
                default = 1,
                required = False)
    parser.add_argument('--n',dest='npairs',
                nargs = 1,
                type = int,
                help='Number of beams to use',
                default = [1000000])
    parser.add_argument('--nsig',dest='nsig',
                nargs = 1,
                type = int,
                help='Draws uncertainty contours up to this number of standard deviations.',
                default = [2])
    parser.add_argument('--s', dest='source',
                nargs = 1,
                type=str,
                help="Draws given coordinate location (format: hms,dms) on localisation plot",
                required = False)
    parser.add_argument('--ticks', dest='tickspacing',
                        nargs = 1,
                        type = float,
                        help = "Sets the number of pixels between ticks on the localisation plot",
                        default = [100],
                        required = False)
    parser.add_argument('--clip', dest='clipping',
                        nargs = 1,
                        type = float,
                        help = "Sets values of the PSF below this number to zero. Helps minimise the influence of low-level sidelobes",
                        default = [0.08],
                        required = False)      
    parser.add_argument('--fits', dest='fitsOut',
                        help = "Outputs .fits file of localisation region",
                        action = 'store_true')   

    options,unknown= parser.parse_known_args()

    return options

def place_beam(p2,j, npairs, c, data, array_height, array_width, psf_ar, options, beam_ar):
    sys.stdout.write("\rAdding beam %d/%d..." % (j + 1, npairs + 1))
    sys.stdout.flush()

    # ax.scatter(c.ra.px, c.dec.px, color='black', s=0.2)

    comparison_snr = data["SN"][j]

    comparison_ar = np.zeros((array_height, array_width))

    dec_start = int(np.round(c.dec.px[j])) - int(psf_ar.shape[1] / 2)
    dec_end = int(np.round(c.dec.px[j])) + int(psf_ar.shape[1] / 2)
    ra_start = int(np.round(c.ra.px[j])) - int(psf_ar.shape[0] / 2)
    ra_end = int(np.round(c.ra.px[j])) + int(psf_ar.shape[0] / 2)
    comparison_ar[dec_start : dec_end, ra_start : ra_end] = psf_ar
            

    x, y = np.meshgrid(np.linspace(0, comparison_ar.shape[1], comparison_ar.shape[1]), np.linspace(0, comparison_ar.shape[0], comparison_ar.shape[0]))
    p2.contour(x, y, comparison_ar, [options.overlap], line_color="black")

    x, y = np.meshgrid(np.linspace(0, beam_ar.shape[1], beam_ar.shape[1]), np.linspace(0, beam_ar.shape[0], beam_ar.shape[0]))
    p2.contour(x, y, beam_ar, [options.overlap], line_color="black")

    return comparison_ar / beam_ar


def make_map(p2, array_height, array_width, c, psf_ar, options, data):
    if options.npairs[0] > 2 and options.npairs[0] + 1 <= len(c):
        npairs = options.npairs[0] - 1
    else:
        npairs = len(c) - 1

    loglikelihood = np.zeros((array_height, array_width))

    nit = 1000  # number of iterations for covariance matrix

    fake_snrs = data["SN"][None, :] + np.random.randn(nit * len(c)).reshape(nit, len(c))

    # make covariance matrix
    beam_snr = data["SN"][0]
    beam_snrs_fake = fake_snrs[:, 0]

    sim_ratios = np.transpose([fake_snrs[:, j] / beam_snrs_fake for j in np.arange(1, npairs + 1)])
    obs_ratios = np.transpose([data["SN"][j] / beam_snr for j in np.arange(1, npairs + 1)])

    C = np.cov(sim_ratios, rowvar=False)

    # make model and get residuals
    sys.stdout.write("\rAdding beam %d/%d..." % (1, npairs + 1))
    sys.stdout.flush()


    beam_ar = np.zeros((array_height, array_width))
    beam_snr = data["SN"][0]  # NB, beams must be sorted by S/N; highest first!

    dec_start = int(np.round(c.dec.px[0])) - int(psf_ar.shape[1] / 2)
    dec_end = int(np.round(c.dec.px[0])) + int(psf_ar.shape[1] / 2)
    ra_start = int(np.round(c.ra.px[0])) - int(psf_ar.shape[0] / 2)
    ra_end = int(np.round(c.ra.px[0])) + int(psf_ar.shape[0] / 2)

    beam_ar[dec_start: dec_end, ra_start: ra_end] = psf_ar

    x, y = np.meshgrid(np.linspace(0, beam_ar.shape[1], beam_ar.shape[1]), np.linspace(0, beam_ar.shape[0], beam_ar.shape[0]))
    p2.contour(x, y, beam_ar, [options.overlap], line_color="black")

    psf_ratios = np.transpose([place_beam(p2, j, npairs, c, data, array_height,
                                           array_width, psf_ar, options, beam_ar)
                               for j in np.arange(1, npairs + 1)], axes=(1, 2, 0))

    resids = np.transpose([obs_ratios[i] - psf_ratios[:, :, i] for i in np.arange(0, npairs)],
                          axes=(1, 2, 0))

    chi2 = np.sum(resids * np.sum(np.linalg.inv(C)[None, None, :, :] *
                                  resids[:, :, :, None], axis=2), axis=2)

    chi2[chi2 == np.inf] = np.nan
    loglikelihood = -0.5 * chi2

    return loglikelihood



#############
# APP
#############

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management
app.config['UPLOAD_FOLDER'] = 'path/to/your/upload/folder'

socketio = SocketIO(app)


@app.route("/", methods=['GET', 'POST'])
def index():
    parser = argparse.ArgumentParser()
    options = parseOptions(parser)

    if 'user_data' not in session:
        session['user_data'] = [
                        {'ra': '4:08:23.82', 'dec': '-18:16:49.0', 'sn': '47.87'},
                        {'ra': '4:08:31.77', 'dec': '-18:17:20.3', 'sn': '56.78'},
                        {'ra': '4:08:30.71', 'dec': '-18:15:23.8', 'sn': '13.45'},
                        ]

    if request.method == 'POST':
        p2 = figure(width=600, height=600)
        p2.background_fill_color = "#F7FBFF"

        # Get the user-entered data from the form
        user_data = []

        for i in range(len(request.form.getlist('ra[]'))):
            if request.form.getlist('ra[]')[i] and request.form.getlist('dec[]')[i] and request.form.getlist('sn[]')[i]:
                ra = request.form.getlist('ra[]')[i]
                dec = request.form.getlist('dec[]')[i]
                sn = request.form.getlist('sn[]')[i]

            set_data = {
                'RA': ra,
                'Dec': dec,
                'SN': sn
            }
            user_data.append(set_data)
        
        if len(user_data) > 2:
            session['user_data'] = user_data


    
        options.res = [float(value) for value in request.form.getlist('res[]')]
        options.overlap = float(request.form.get('overlap[]', 0.25))
        options.npairs = [0]
        options.clipping = [float(value) for value in request.form.getlist('clipping[]')]
        options.nsig = [float(value) for value in request.form.getlist('nsig[]')]
        options.tickspacing = [float(value) for value in request.form.getlist('tickspacing[]')]
        options.config = False

        # Retrieve the last uploaded file path from the session
        psf_file_path = session.get('psf_file_path', None)

        # Check if a new file has been uploaded
        new_psf_file = request.files.get('psf_file')
        if new_psf_file:
            # Save the uploaded file to a temporary location
            psf_file_path = os.path.join("/tmp", secure_filename(new_psf_file.filename))
            new_psf_file.save(psf_file_path)
            # Update the session with the new file path
            session['psf_file_path'] = psf_file_path

        # Set the PSF option to the file path
        options.psf = [psf_file_path] if psf_file_path else ['default_psf.fits']

        data, c, boresight = ut.readCoords(options,session['user_data'])

        psf_ar = ut.readPSF(options.psf[0], options.clipping[0])

        c, w, array_width, array_height = co.deg2pix(c, psf_ar, 
                                    boresight, options.res[0])


        if options.source:
            Splot.plot_known(w, options.source[0])
        
        loglikelihood = make_map(p2,array_height, array_width,
                                 c, psf_ar, options, data)

        Splot.make_ticks(p2, array_width, array_height, 
                        w, fineness=1)

        print("\nPlotting...")
        p3,p4,message = Splot.likelihoodPlot(p2, w, loglikelihood, options)

        layout = gridplot([
            [p3],
            [p2, p4],
            ])
  

        # Embed the Bokeh plot components into the HTML
        script, div = components(layout, INLINE)
        return render_template('index.html', script=script, div=div,message=message)

    return render_template('index.html', img_data=None, user_data=session['user_data'])

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == "__main__":
    socketio.run(app, port=2000, debug=True)
