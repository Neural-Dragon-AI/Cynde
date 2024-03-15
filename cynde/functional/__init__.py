# This is the __init__.py file for the package.
from .embed import *
from .prompt import *
from .cv import *
from .classify import *
from .generate import *
from .results import *

from dotenv import load_dotenv

load_dotenv()


def set_directories(root_dir):
    if root_dir is None:
        raise ValueError("CYNDE_DIR environment variable must be set before using cynde")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "cynde_mount"), exist_ok=True)
    os.environ['CACHE_DIR'] = os.path.join(root_dir, "cache")
    os.environ['OUTPUT_DIR'] = os.path.join(root_dir, "output")
    os.environ['MODAL_MOUNT'] = os.path.join(root_dir, "cynde_mount")

root_dir = os.getenv('CYNDE_DIR')
set_directories(root_dir)