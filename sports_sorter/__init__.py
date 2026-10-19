"""sports-image-sorter: classify sports photos into per-sport folders.

Public entry points are re-exported here so that library users can do::

    from sports_sorter import Config, Store, scan_directory
"""

from .config import Config, Label, load_config
from .db import Store
from .images import ImageRecord, scan_directory, sha256_file, dhash
from .safety import SafetyError, assert_input_allowed, assert_safe_output

__all__ = [
    "Config",
    "Label",
    "load_config",
    "Store",
    "ImageRecord",
    "scan_directory",
    "sha256_file",
    "dhash",
    "SafetyError",
    "assert_input_allowed",
    "assert_safe_output",
]

__version__ = "0.1.0"
