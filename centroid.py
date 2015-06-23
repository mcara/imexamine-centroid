"""
Python portal of the centroid algorithm used in IRAF's imexamine task.
According to IRAF's documentation, IRAF's implementation of the cedntroid
algorithm uses the Mountain Photometry Code Algorithm as outlined in
Stellar Magnitudes from Digital Images.

Porting to Python performed by Mihai Cara.
Date: 06/20/2015

THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL I BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**COPYRIGHT: ** Because this software uses an algorithm from the IRAF package,
we provide a copy of the IRAF's original copyright notice:

Copyright(c) 1986 Association of Universities for Research in Astronomy Inc.

The IRAF software is publicly available, but is NOT in the public domain.
The difference is that copyrights granting rights for unrestricted use and
redistribution have been placed on all of the software to identify its authors.
You are allowed and encouraged to take this software and use it as you wish,
subject to the restrictions outlined below.

Permission to use, copy, modify, and distribute this software and its
documentation is hereby granted without fee, provided that the above copyright
notice appear in all copies and that both that copyright notice and this
permission notice appear in supporting documentation, and that references to
the Association of Universities for Research in Astronomy Inc. (AURA),
the National Optical Astronomy Observatories (NOAO), or the Image Reduction
and Analysis Facility (IRAF) not be used in advertising or publicity
pertaining to distribution of the software without specific, written prior
permission from NOAO.  NOAO makes no representations about the suitability
of this software for any purpose.  It is provided "as is" without express or
implied warranty.

NOAO DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL NOAO
BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

import numpy as np
from astropy.io import fits
from os import path
import re
__version__ = '0.1.0'
__author__ = 'Mihai Cara'
__vdate__ = '20-July-2015'


def imexcentroid(image, ext, xyin, radius=5, xyout=None, fmt='.3f', sep=' '):
    """
    Provided with an input image and either a Python list of initial coordinates
    (and optionally the "radius" of the extraction box) or a list of coordinates
    (and optional radii) saved in a text file, the code will output "improved"
    coordinates of the sources in the image computed using the original IRAF's
    centroid algorithm.

    Parameters
    ----------
        image : str, numpy.ndarray
            Either a string file name of an image file or a numpy.ndarrray
            containing image data.

        ext : int, tuple
            Either an integer extension number or a tuple of extension name and
            extension extension version, e.g., ``('sci', 1)``.

            .. note::
                This parameter is ignored when input `image` argument
                is a `numpy.ndarray`.

        xyin : str, list, tuple, numpy.ndarray
            Initial guess for source coordinates and optional radius of the
            extraction box to be used for that specific source. If radius is
            omitted, then the default value specified by the `radius` parameter
            will be used. This parameter can be provided in several forms:

            * a comma, semi-colon, space, or tab-separated list of source
              coordinates (one source per line) followed by an optional radius
              of the extraction box;

            * a list or tuple of lists or tuples containing two coordinates for
              each source and an optional extraction box radius for a given
              source.

            * a `numpy.ndarray` of shapes `Nx2` or `Nx3` describing initial
              source coordinates and the optional radius of the extraction box
              for each source.

        radius : int, float (Default : 5)
            Determines the dimension of the extraction box which will be the
            bounding box of a circle of radius specified by `radius`.

        xyout : str, None (Default : None)
            The name of the output text file to which the computed centroid
            coordinates should be saved. If `xyout` is `None` the output
            coordinates will not be written to a file.

        fmt : str (Default : '.3f')
            Numeric format to be used when writting out centroid coordinates to
            the text file specified by `xyout`. This parameter is ignored if
            `xyout` is `None`.

        sep : str (Default : ' ')
            Separator string to be used to separate the coordinates written out
            to the text file specified by the `xyout` parameter. This parameter
            is ignored if `xyout` is `None`.


    Returns
    -------
        centroid_xy : list
            A list of lists of centroid coordinates.


    Examples
    --------
    >>> from centroid import imexcentroid
    >>> imexcentroid('j8ki01f4q_flt.fits', ('sci',1), [[698,2030,7],[303,2036],
    ... [49,1961]], 5, xyout='centroids.txt')
    [[699.13499669315183, 2030.8008360207496],
     [303.02899254608371, 2036.8800981069467],
     [49.824582149004847, 1961.944597170836]]

    If the coordinates are in a text file, e.g., 'initial_coordinates.txt' then
    provide that file name:
    >>> imexcentroid('j8ki01f4q_flt.fits', ('sci',1), 'initial_coordinates.txt',
    ... 5, xyout='centroids.txt')
    [[699.13499669315183, 2030.8008360207496],
     [303.02899254608371, 2036.8800981069467],
     [49.824582149004847, 1961.944597170836]]

    The `radius` parameter (5 in the above examples) specifies the **default**
    radius of the extraction box to be used by the centroid algorithm. However,
    one can override this value by providing a radius following the x and y
    coordinates of a source. In the first example, the extraction box for the
    first source will be of radius 7 pixels while for the rest of the sources it
    will be 5 pixels (default value specified by `radius`).

    Default format for output coordinates is `fmt`='.3f' and the default
    separator (between x and y coordinates) is `sep`=' '.

    """
    # get image data:
    if isinstance(image, str):
        image = fits.getdata(image, ext=ext)
    elif not isinstance(image, np.ndarray):
        raise TypeError("Unsupported type for the 'image' parameter")

    # make sure we are dealing with 2D images:
    if len(image.shape) != 2:
        raise ValueError("Input image must 2-dimensional")

    # make sure radius > 0:
    if radius <= 0.0:
        raise ValueError("Parameter 'radius' must be a strictly positive number")

    # prepare format string for the output coordinates:
    if xyout is not None:
        if isinstance(xyout, str):
            fmt = "{{:{:s}}}{:s}{{:{:s}}}\n".format(fmt, sep, fmt)
        else:
            raise TypeError("'xyout' must be either None or a valid file name")

    # if input (initial) coordinates are in a text file, read the file and extract x,y,[r];
    # for numpy.ndarray or list input - make sure it has radius set.
    if isinstance(xyin, str):
        r = '[,; \t\r\n]*'

        f = open(xyin)
        lines = f.readlines()
        f.close()

        xyin = []

        for line in lines:
            l = line.strip()
            # skip over empty lines:
            if len(l) < 1 or l[0] == '#':
                continue
            # split the string into "words"
            xyr = map(float, [x for x in re.split(r, l) if len(x) > 0][:3])
            if len(xyr) == 2:
                xyr.append(radius)
                xyin.append(xyr)
            elif len(xyr) >= 3:
                xyin.append(xyr[:3])
            else:
                raise ValueError("Incorrect format encountered while reading "
                                 "input coordinate file: '{}'".format(l))

    elif isinstance(xyin, np.ndarray):
        if xyin.shape[1] == 2:
            lrad = [ radius ]
            xyin = [ list(i) + lrad for i in xyin ]
        elif xyin.shape[1] > 2:
            xyin = [ list(i)[:3] for i in xyin ]
        else:
            raise ValueError("Initial coordinates must be an array of NxM "
                             "values where M >= 2")

    elif isinstance(xyin, list):
        xyinnew = []
        for xy in xyin:
            xyi = list(xy)
            if len(xyi) == 2:
                xyi.append(radius)
            if len(xyi) > 2:
                xyi = xyi[:3]
            else:
                raise ValueError("Initial coordinates must be an list of lists "
                                 "of M values where M >= 2")
            xyinnew.append(xyi)
        xyin = xyinnew

    else:
        raise TypeError("Unsuported type for input coordinates")

    ################################################
    ##  Main algorithm for computing centroids    ##
    ##  for all input source coordinates:         ##
    ################################################

    centroid_xy = []
    ymax = image.shape[0] - 1
    xmax = image.shape[1] - 1
    niter = range(3)

    for xi, yi, radius in xyin:
        xc0 = xi
        yc0 = yi

        for k in niter:
            # find the bounding box for extraction
            x1 = int(xc0 - radius + 0.5)
            x2 = int(xc0 + radius + 0.5)
            y1 = int(yc0 - radius + 0.5)
            y2 = int(yc0 + radius + 0.5)

            (x1, x2, y1, y2) = _inbounds_box(x1-1, x2-1, y1-1, y2-1, xmax, ymax)
            box = image[y1:y2,x1:x2]

            # create lists of coordinates
            xs = np.arange(x1+1,x2+1)
            ys = np.arange(y1+1,y2+1)

            # compute marginal distribution for x-axis
            margx = box.sum(axis=0, dtype=np.float64)
            meanx = margx.mean()
            margx -= meanx
            goodx = margx > 0.0 # no data
            if not goodx.any():
                xc = np.nan
                yc = np.nan
                break

            # compute marginal distribution for y-axis
            margy = box.sum(axis=1, dtype=np.float64)
            meany = margy.mean()
            margy -= meany
            goody = margy > 0.0
            if not goody.any(): # no data
                xc = np.nan
                yc = np.nan
                break

            # compute centroid
            margx_good = margx[goodx]
            margy_good = margy[goody]
            xc = np.dot(xs[goodx], margx_good) / margx_good.sum()
            yc = np.dot(ys[goody], margy_good) / margy_good.sum()

            # check that the centroid is in the same pixel as the initial guess:
            if int(xc) == int(xc0) and int(yc) == int(yc0):
                break
            else:
                xc0 = xc
                yc0 = yc

        centroid_xy.append([xc, yc])

    # save results to a file (if provided) and also return the list of coordinates:
    if xyout is not None:
        lines = []
        for xy in centroid_xy:
            lines.append(fmt.format(xy[0], xy[1]))
        f = open(xyout, 'w')
        f.writeline(line)
        f.close()

    return centroid_xy


def _inbounds_box(x1, x2, y1, y2, xmax, ymax):
    # xmax, ymax - upper bound for indeces (should be image size along a
    #              dimension - 1)
    # x1, x2, y1, y2 - bounds of an image slice
    # Assumtions: x1 < x2, y1 < y2, xmax >= 0, ymax >= 0
    if x2 < 0 or x1 > xmax or y2 < 0 or y1 > ymax:
        return (0, 0, 0, 0, 0)
    if x1 < 0:
        x1 = 0
    if x2 > xmax:
        x2 = xmax
    if y1 < 0:
        y1 = 0
    if y2 > ymax:
        ymax
    return (x1, x2+1, y1, y2+1)
