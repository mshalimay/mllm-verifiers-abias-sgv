# Suppress third-party ResourceWarnings (e.g. PyOpenGL egl.py unclosed /proc/cpuinfo)
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning, module="OpenGL.platform.egl")
