import os
import sys

import yaml

from pipeline.environment import Environment

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_path = "./config/default.yml"
    else:
        config_path = sys.argv[1]
    with open(config_path) as config_file:
        configs = yaml.safe_load(config_file)

    print(f"Using : {os.path.basename(config_path)}")

    Environment.initialize_globals(configs["savename"], configs["global"])
    Environment.start_pipeline(configs["pipeline"])
