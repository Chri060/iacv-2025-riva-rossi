import os, sys, dash, yaml
from dash import Dash, Input, Output, html
from flask import send_file
from pipeline.environment import Environment
from pipeline.pipes.intrinsic_calibration import IntrinsicCalibration
from pipeline.pipes.extrinsic_calibration import ExtrinsicCalibration
from pipeline.pipes.lane_detection import DetectLane
from pipeline.pipes.ball_tracking import TrackBall
from pipeline.pipes.ball_localization import LocalizeBall
from pipeline.pipes.ball_motion import SpinBall
from pipeline.pipes.video_synchronization import SynchronizeVideo
from pipeline.pipes.video_undistortion import UndistortVideo

# Define the static directory path for resources
STATIC_DIR = f"{os.getcwd()}/resources"

# Initialize Dash app
app = Dash()


@app.server.route("/video/<path:video_path>")
def serve_video(video_path: str):
    """
    Serve a video file from the server's static directory.

    This endpoint takes a video path relative to the server's "videos" folder
    inside the STATIC_DIR, verifies that the file exists, and returns it
    as an HTTP response with the correct MIME type. Supports conditional
    requests for efficient streaming.

    Args:
        video_path (str): Relative path to the video file under the "videos"
            directory in STATIC_DIR.

    Returns:
        Response: A Flask response object containing the video file if found.
    """

    # Define the path
    full_path = os.path.join(STATIC_DIR, "videos", video_path)

    return send_file(full_path, mimetype="video/mp4", conditional=True)


def main():
    """
    The main entry point for creating and running the interactive Dash visualization app.

    Args:
        None. Command-line arguments are used for configuration selection.
    """

    # Determine config the file path from command-line argument or default
    if len(sys.argv) == 1:
        config_path = "./config/dev.yml"
    else:
        config_path = sys.argv[1]

    # Load configuration from YAML file
    with open(config_path) as config_file:
        configs = yaml.safe_load(config_file)

    print(f"Loading Dashboard for : {os.path.basename(config_path)}")

    # Initialize global environment variables
    Environment.initialize_globals(configs["save_name"], configs["global"])
    pipe_configs: list[dict] = configs["pipeline"]

    # Map pipe names to classes for dynamic instantiation
    pipes: dict = {
        "intrinsic": IntrinsicCalibration,
        "video_synchronization": SynchronizeVideo,
        "video_undistortion": UndistortVideo,
        "lane_detection": DetectLane,
        "extrinsic": ExtrinsicCalibration,
        "ball_tracker": TrackBall,
        "ball_localization": LocalizeBall,
        "ball_rotation": SpinBall
    }

    # Initialize the pipeline modules
    print("Initializing Pipes :")
    for key in pipes.keys():
        pipes.update({key: pipes.get(key)(Environment.save_name)})
        print(f"> {key}")

    pages = {}
    welcome_page = None
    welcome_page_name = None

    # Iterate over pipeline configuration and generate Dash pages
    for i, pipe_conf in enumerate(pipe_configs):
        pipe = pipes[pipe_conf["name"]]
        params = pipe_conf.get("params", None)
        page = pipe.plotly_page(params)
        if page is not None:
            pages.update(page)
            if welcome_page is None:
                welcome_page_name = next(iter(page))
                welcome_page = page[welcome_page_name]

    # Title prefix for the Dash app
    title_text = "Dash Visualization : "

    # Define the Dash layout
    app.layout = [
        html.Div(id="title", children=title_text + welcome_page_name),
        html.Hr(),
        html.Div([html.Button(key, id=key, n_clicks=0) for key in pages.keys()]),
        html.Hr(),
        html.Div(id="page", children=welcome_page),
    ]

    # Callback to handle button clicks and update the displayed page
    @app.callback(
        [Output("title", "children"), Output("page", "children")],
        [Input(name, "n_clicks") for name in pages.keys()],
    )
    def update_output(*args):
        # Determine which button triggered the callback
        ctx = dash.callback_context
        if not ctx.triggered:
            # Default: show welcome page
            return title_text + welcome_page_name, welcome_page
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            return title_text + button_id, pages[button_id]

    # Run the Dash server
    app.run(debug=False)


if __name__ == "__main__":
    main()
