from dash import html


class Pipe:
    def __init__(self, savename: str | None = None):
        if savename is None:
            savename = "save"
        self.save_name = f"{self.__class__.__name__}_{savename}"

    def execute(self, params: dict) -> None:
        """
        Executes the pipes main functionality
        """
        print(f"Missing Implementation for {self.__class__.__name__} : execute")

    def load(self, params: dict) -> None:
        """
        Loads into the Environment the previously computed results according to the global savename
        """
        print(f"Missing Implementation for {self.__class__.__name__} : load")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Prepares the dash components to be displayed into the dashboard
        """
        text = f"Missing Implementation for {self.__class__.__name__} : plotly"
        return {self.__class__.__name__: html.Div(children=text)}
