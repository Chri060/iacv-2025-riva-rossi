from dash import html


class Pipe:
    """
    Base class for all pipeline steps in a data processing workflow.

    Each subclass of Pipe represents a modular processing unit that performs
    specific operations on data. A Pipe has three main responsibilities:

        1. `execute(params: dict)`: Run the main processing functionality.
        2. `load(params: dict)`: Reload previously computed results from storage
           into the environment.
        3. `plotly_page(params: dict) -> dict[str, html.Div]`: Provide a Plotly
           Dash visualization of the results for the dashboard.

    Attributes:
        save_name (str): Unique name used when saving results, combining
                         the provided `save_name` with the class name.
    """

    def __init__(self, save_name: str | None = None):
        """
        Initialize the pipeline step.

        Args:
            save_name (str | None, optional): Custom base name for saving results.
                If None, defaults to "save". The final `save_name` combines
                this base name with the class name.
        """

        if save_name is None:
            save_name = "save"
        self.save_name = f"{save_name}_{self.__class__.__name__}"

    def execute(self, params: dict) -> None:
        """
        Execute the main functionality of this pipeline step.

        Subclasses should override this method with actual processing logic.

        Args:
            params (dict): Parameters required to perform the execution.
        """

        print(f"Missing Implementation for {self.__class__.__name__} : execute")

    def load(self, params: dict) -> None:
        """
        Load previously computed results into the environment.

        Subclasses should override this method to load saved data or results.

        Args:
            params (dict): Parameters required for loading results.
        """

        print(f"Missing Implementation for {self.__class__.__name__} : load")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Prepare Plotly Dash components for visualization in a dashboard.

        Subclasses should override this method to define custom layouts
        and return interactive components.

        Args:
            params (dict): Parameters for generating the visualization.

        Returns:
            dict[str, html.Div]: A mapping from the step's class name to
            a Dash HTML component representing the visualization. Defaults
            to a placeholder message if not implemented.
        """

        text = f"Missing Implementation for {self.__class__.__name__} : plotly"
        return {self.__class__.__name__: html.Div(children=text)}
