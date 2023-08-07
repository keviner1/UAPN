import importlib
import glob
import sys
sys.path.append("..")
from Registry import CONFIG_REGISTRY

config_filenames = glob.glob("config/config_*.py")
config_filenames = [i.replace("config\\","").replace(".py","") for i in config_filenames]
_config_modules = [importlib.import_module(f'.{file_name}',package='config') for file_name in config_filenames]

