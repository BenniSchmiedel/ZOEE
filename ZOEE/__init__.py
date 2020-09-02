def Tutorial_copy(*args,**kwargs):
    import shutil, sys, os
    path=kwargs.get('path',os.getcwd())
    possible_paths=[]
    if path!=os.getcwd():
        subdir=['..','../..','../../..','../../../..','../../../../..']
        for i in subdir:
            possible_paths.append(i+path)
            
    data_paths=[]
    for i in sys.path:
        data_paths.append(i+'/lowEBMs/tutorials')
    for data in data_paths:
        exists = os.path.isdir(data)
        if exists:
            location=data
            break
    exit=False
    for trypath in possible_paths:
        exists= os.path.isdir(trypath)
        if exists:
            
            if os.path.exists(trypath+'/tutorials/Notebooks') and os.path.exists(trypath+'/tutorials/config'):
                shutil.rmtree(trypath+'/tutorials/Notebooks')
                shutil.rmtree(trypath+'/tutorials/config')
            shutil.copytree(data+'/Notebooks',trypath+'/tutorials/Notebooks')
            shutil.copytree(data+'/config',trypath+'/tutorials/config')
            exit=True
            break
    if exit:
        print('Copy tutorial files to:'+trypath)
    else:
        print('Output path could not be found, please insert a valid path')
        
def Forcing_copy(*args,**kwargs):
    import shutil, sys, os
    path=kwargs.get('path',os.getcwd())
    possible_paths=[]
    if path!=os.getcwd():
        subdir=['..','../..','../../..','../../../..','../../../../..']
        for i in subdir:
            possible_paths.append(i+path)
            
    data_paths=[]
    for i in sys.path:
        data_paths.append(i+'/lowEBMs/Forcings')
    for data in data_paths:
        exists = os.path.isdir(data)
        if exists:
            location=data
            break
    exit=False
    for trypath in possible_paths:
        exists= os.path.isdir(trypath)
        if exists:
            
            if os.path.exists(trypath+'/Forcings/TSI') and os.path.exists(trypath+'/Forcings/Orbital') and os.path.exists(trypath+'/Forcings/Volcanic'):
                shutil.rmtree(trypath+'/Forcings/TSI')
                shutil.rmtree(trypath+'/Forcings/Orbital')
                shutil.rmtree(trypath+'/Forcings/Volcanic')
            shutil.copytree(data+'/TSI',trypath+'/Forcings/TSI')
            shutil.copytree(data+'/Orbital',trypath+'/Forcings/Orbital')
            shutil.copytree(data+'/Volcanic',trypath+'/Forcings/Volcanic')
            exit=True
            break
    if exit:
        print('Copy forcing data to:'+trypath)
    else:
        print('Output path could not be found, please insert a valid path')
        

def update_plotstyle():
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [16,9]
    matplotlib.rcParams['axes.titlesize']=24
    matplotlib.rcParams['axes.labelsize']=20
    matplotlib.rcParams['lines.linewidth']=2.5
    matplotlib.rcParams['lines.markersize']=10
    matplotlib.rcParams['xtick.labelsize']=16
    matplotlib.rcParams['ytick.labelsize']=16
    matplotlib.rcParams['ytick.labelsize']=16
    matplotlib.rcParams['ytick.minor.visible']=True
    matplotlib.rcParams['ytick.direction']='inout'
    matplotlib.rcParams['ytick.major.size']=10
    matplotlib.rcParams['ytick.minor.size']=5
    matplotlib.rcParams['xtick.minor.visible']=True
    matplotlib.rcParams['xtick.direction']='inout'
    matplotlib.rcParams['xtick.major.size']=10
    matplotlib.rcParams['xtick.minor.size']=5

def moving_average(signal, period):
    import numpy as np
    #buffer = [np.nan] * period
    buffer=[signal[:12].mean()]*12
    for i in range(period,len(signal)):
        buffer.append(signal[i-period:i].mean())
    return buffer


import numpy as np
from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator

class LatitudeScale(mscale.ScaleBase):
    name = 'latarea'

    def __init__(self, axis, *, thresh=np.deg2rad(89.5), **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        thresh: The degree above which to crop the data.
        """
        super().__init__(axis)
        if thresh >= np.pi / 2:
            raise ValueError("thresh must be less than pi/2")
        self.thresh = thresh

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        transform = self.InvertedMercatorLatitudeTransform(self.thresh)
        return transform

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in ``ticker.py``.

        In our case, the Mercator example uses a fixed locator from
        -90 to 90 degrees and a custom formatter class to put convert
        the radians to degrees and put a degree symbol after the
        value::
        """
        class DegreeFormatter(Formatter):
            def __call__(self, x, pos=None):
                number=np.degrees(x)
                label="%d\N{DEGREE SIGN}" % np.round(np.degrees(x))
                return label

        axis.set_major_locator(FixedLocator(
            np.deg2rad([-80,-60,-40,-20,0,20,40,60,80])))#np.linspace(-60, 60, 5))))
        axis.set_major_formatter(DegreeFormatter())
        #axis.set_minor_formatter(DegreeFormatter())
        
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, -self.thresh), min(vmax, self.thresh)
    
    class MercatorLatitudeTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the Mercator scale
            is limited by the user-specified threshold, the input
            array must be masked to contain only valid values.
            ``matplotlib`` will handle masked arrays and remove the
            out-of-range data from the plot.  Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            if masked.mask.any():
                return ma.log(np.abs(ma.tan(masked) + 1.0 / ma.cos(masked)))
            else:
                return np.log(np.abs(np.tan(a) + 1.0 / np.cos(a)))

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return LatitudeScale.InvertedMercatorLatitudeTransform(
                self.thresh)

    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            forward =np.arctan(np.sinh(a))
            return forward

        def inverted(self):
            return LatitudeScale.MercatorLatitudeTransform(self.thresh)

