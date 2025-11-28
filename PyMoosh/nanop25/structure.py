import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm 
from PyMoosh.modes import steepest
from scipy.special import erf
from scipy.linalg import toeplitz, inv
import matplotlib

font = {"family" : "DejaVu Serif", "weight" : "normal", "size": 15}
matplotlib.rc("font", **font)

i = complex(0,1)

material_list = [1., 1.54, 'Silver', 'Gold']
stack = [2,0,3,1]

thicknesses = [20,5,10,20]
Layers = pm.Structure(material_list, stack, thicknesses, verbose=False)

# Clear any old figures first
plt.close('all')

Layers.plot_stack(600)   # this draws the plot but returns None

## Not working - empty figures
# # Find the last figure that actually has axes (not an empty one)
# figs = [f for f in map(plt.figure, plt.get_fignums()) if len(f.axes) > 0]
# if len(figs) == 0:
#     raise RuntimeError("No figures with axes found after plot_stack()")

# fig = figs[-1]     # last non-empty figure
# ax = fig.axes[0]   # first axis in that figure

# # Modify labels
# ax.set_title("Representation of the structure")
# #ax.set_xlabel("Pixel X")
# ax.set_ylabel("Thickness (nm)")

# # Save the figure
#plt.savefig("nanop25/stack_plot.svg", format="svg", dpi=300, bbox_inches="tight")
# # or PDF
# #fig.savefig("stack_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")

# plt.show()
