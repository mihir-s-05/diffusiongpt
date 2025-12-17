"""
Poor Man's Configurator (from nanoGPT), with a small tweak:
if a config path isn't found, we also try it relative to this file.
"""

import os
import sys
from ast import literal_eval


THIS_DIR = os.path.dirname(__file__)

for arg in sys.argv[1:]:
    if "=" not in arg:
        assert not arg.startswith("--")
        config_file = arg
        if not os.path.exists(config_file):
            config_file = os.path.join(THIS_DIR, config_file)
        print(f"Overriding config with {config_file}:")
        with open(config_file, encoding="utf-8") as f:
            print(f.read())
        exec(open(config_file, encoding="utf-8").read())
    else:
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

