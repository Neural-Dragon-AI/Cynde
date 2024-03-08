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
    os.makedirs(os.path.join(root_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "modal_sharing"), exist_ok=True)
    os.environ['CACHE_DIR'] = os.path.join(root_dir, "cache")
    os.environ['OUTPUT_DIR'] = os.path.join(root_dir, "output")
    os.environ['MODAL_SHARING_DIR'] = os.path.join(root_dir, "modal_sharing")

root_dir = os.getenv('ROOT_DIR')
set_directories(root_dir)