import importlib
import glob
import sys
sys.path.append("..")
from Registry import ARCH_REGISTRY

arch_filenames = glob.glob("model/*_arch.py")
arch_filenames = [i.replace("model\\","").replace(".py","") for i in arch_filenames]
_arch_modules = [importlib.import_module(f'.{file_name}',package='model') for file_name in arch_filenames]

