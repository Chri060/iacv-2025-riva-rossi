from dash import html

class Pipe:
    """
    Base class for all pipeline steps.

    Each subclass of Pipe represents a processing unit in the workflow,
    such as calibration, undistortion, feature extraction, etc.
    A Pipe defines three main responsibilities:
    - `execute`: Run the main functionality (process data, save results).
    - `load`: Reload previously computed results into the Environment.
    - `plotly_page`: Provide a Plotly Dash visualization of the results.

    Attributes:
        save_name (str): Unique name used when saving results.
                         Defaults to "<ClassName>_save".
    """

    def __init__(self, save_name: str | None = None):
        """
        Initialize the pipeline step.

        Parameters:
           save_name (str | None, optional): Custom save name for results.
                                            If None, defaults to "save".
        """

        if save_name is None:
            save_name = "save"
        self.save_name = f"{save_name}_{self.__class__.__name__}"

    def execute(self, params: dict) -> None:
        """
        Executes the pipeline main functionality.

        Subclasses should override this method with the actual implementation.

        Parameters:
            params (dict): Dictionary of parameters needed for execution.
        """

        print(f"Missing Implementation for {self.__class__.__name__} : execute")

    def load(self, params: dict) -> None:
        """
        Loads previously computed results into the Environment.

        Subclasses should override this method to reload saved data/results.

        Parameters:
            params (dict): Dictionary of parameters for loading results.
        """

        print(f"Missing Implementation for {self.__class__.__name__} : load")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Prepares the Plotly Dash components for visualization in the dashboard.

        Subclasses should override this method to define custom layouts.

        Parameters:
            params (dict): Dictionary of parameters for visualization.

        Returns:
            dict[str, html.Div]: A mapping of class name to Dash HTML component(s).
        """
        
        text = f"Missing Implementation for {self.__class__.__name__} : plotly"
        return {self.__class__.__name__: html.Div(children=text)}