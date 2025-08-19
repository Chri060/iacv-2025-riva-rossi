import os
import sys

import dash
import yaml
from dash import Dash, Input, Output, html
from flask import send_from_directory

from pipeline.environment import Environment

# Define the absolute path to the static directory
STATIC_DIR = f"{os.getcwd()}/resources"

app = Dash()


@app.server.route("/video/<path:video_path>")
def serve_video(video_path):
    video_folder = f"{STATIC_DIR}/videos"
    return send_from_directory(video_folder, video_path, mimetype="video/mp4")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_path = "./config/default.yml"
    else:
        config_path = sys.argv[1]
    with open(config_path) as config_file:
        configs = yaml.safe_load(config_file)

    print(f"Loading Dashboard for : {os.path.basename(config_path)}")

    Environment.initialize_globals(configs["savename"], configs["global"])
    pipe_configs: list[dict] = configs["pipeline"]

    import pipeline.pipes.calibration as cal
    import pipeline.pipes.localization as loc
    import pipeline.pipes.video_processing as vp
    from pipeline.pipe import Pipe

    pipes: dict[str, Pipe] = {
        "intrinsic": cal.IntrinsicCalibration,
        "video_synchronization": vp.SynchronizeVideo,
        "video_stabilization": vp.Stabilizer,
        "video_undistortion": vp.UndistortVideo,
        "lane_detection": loc.LaneDetector,
        "extrinsic": cal.ExtrinsicCalibration,
        "ball_tracker": loc.Ball_Tracker,
        "ball_localization": loc.Ball_Localization,
    }

    print("Initializing Pipes :")
    for key in pipes.keys():
        pipes.update({key: pipes.get(key)(Environment.savename)})
        print(f"> {key}")

    pages = {}
    welcome_page = None
    welcome_page_name = None
    for i, pipe_conf in enumerate(pipe_configs):
        pipe = pipes[pipe_conf["name"]]
        params = pipe_conf.get("params", None)
        page = pipe.plotly_page(params)
        if page is not None:
            pages.update(page)
            if welcome_page is None:
                welcome_page_name = next(iter(page))
                welcome_page = page[welcome_page_name]

    title_text = "Dash Visualization : "

    app.layout = [
        html.Div(id="title", children=title_text + welcome_page_name),
        html.Hr(),
        html.Div([html.Button(key, id=key, n_clicks=0) for key in pages.keys()]),
        html.Hr(),
        html.Div(id="page", children=welcome_page),
    ]

    # Callback to handle button clicks
    @app.callback(
        [Output("title", "children"), Output("page", "children")],
        [
            Input(name, "n_clicks") for name in pages.keys()
        ],  # triggers when whenever one of the input changes
    )
    def update_output(*args):
        # Get the button that was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return title_text + welcome_page_name, welcome_page
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[
                0
            ]  # get the id of the clicked button
            return title_text + button_id, pages[button_id]

    app.run(debug=False)
