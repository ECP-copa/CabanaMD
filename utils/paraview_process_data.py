import os

# get all the .pvtu files in current directory
# not very pythonic, but it gets the job done
dumps = []
domain_act = []
domain_lb = []
dirname = os.getcwd()
for f in os.listdir(dirname):
    if f.endswith(".pvtu"):
        if f.startswith("dump"):
            dumps.append(os.path.join(dirname, f))
        elif f.startswith("domain_act"):
            domain_act.append(os.path.join(dirname, f))
        elif f.startswith("domain_lb"):
            domain_lb.append(os.path.join(dirname, f))

# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Partitioned Unstructured Grid Reader'
particles = XMLPartitionedUnstructuredGridReader(
        registrationName='particles', FileName=dumps)

# Properties modified on dump_10pvtu
particles.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

renderView1.ApplyIsometricView()
renderView1.AxesGrid.Visibility = 1

# get display properties
particlesDisplay = GetDisplayProperties(particles, view=renderView1)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph', Input=particles,
    GlyphType='Arrow')

# show data in view
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'

# set scalar coloring
ColorBy(glyph1Display, ('POINTS', 'Velocity'))

# rescale color and/or opacity maps used to include current data range
glyph1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction('Velocity')

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction('Velocity')

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D('Velocity')

# Properties modified on glyph1
glyph1.GlyphMode = 'All Points'
glyph1.GlyphType = 'Sphere'

glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 1.0

# Rescale transfer function
velocityLUT.RescaleTransferFunction(-1.42767, 1.41958)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(-1.42767, 1.41958)

# create a new 'XML Partitioned Unstructured Grid Reader'
domain_actual = XMLPartitionedUnstructuredGridReader(
        registrationName='domain_actual', FileName=domain_act)

# Properties modified on domain_act_10pvtu
domain_actual.TimeArray = 'None'

# show data in view
domain_actualDisplay = Show(domain_actual, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
domain_actualDisplay.Representation = 'Surface'

# Rescale transfer function
velocityLUT.RescaleTransferFunction(-1.42767, 1.41958)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(-1.42767, 1.41958)

# change representation type
domain_actualDisplay.SetRepresentationType('Wireframe')

# set scalar coloring
ColorBy(domain_actualDisplay, ('CELLS', 'rank'))

# rescale color and/or opacity maps used to include current data range
domain_actualDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
domain_actualDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'rank'
rankLUT = GetColorTransferFunction('rank')

# get opacity transfer function/opacity map for 'rank'
rankPWF = GetOpacityTransferFunction('rank')

# get 2D transfer function for 'rank'
rankTF2D = GetTransferFunction2D('rank')

# Properties modified on domain_act_10pvtuDisplay
domain_actualDisplay.LineWidth = 2.0

# reset view to fit data bounds
renderView1.ResetCamera(False, 0.9)

# update the view to ensure updated data information
renderView1.Update()
