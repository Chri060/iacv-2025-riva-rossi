import os, sys, yaml
from pipeline.environment import Environment

def main():
    """
    Main entry point for running a pipeline.

    This script:
    1. Loads a configuration file (YAML format).
    2. Initializes global settings in the Environment.
    3. Starts the pipeline defined in the configuration.

    Usage:
        python main.py [config_path]

    If no config_path is provided, defaults to './config/dev.yml'.
    """

    # Select the configuration file:
    if len(sys.argv) == 1:
        # If no argument is provided, use the default dev config
        config_path = "./config/dev.yml"
    else:
        # Use the path passed as the first command-line argument
        config_path = sys.argv[1]

    # Load YAML configuration
    with open(config_path) as config_file:
        configs = yaml.safe_load(config_file)

    # Print which configuration file is being used
    print(f"Using : {os.path.basename(config_path)}")

    # Initialize the environment with global settings
    Environment.initialize_globals(configs["save_name"], configs["global"])

    # Start pipeline steps defined in config
    Environment.start_pipeline(configs["pipeline"])

if __name__ == "__main__":
    main()