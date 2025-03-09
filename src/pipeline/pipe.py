class Pipe:
    def __init__(self, save_name=None):
        if save_name is None:
            save_name = "save"
        self.save_name = f"{self.__class__.__name__}_{save_name}"

    def execute(self, params):
        print(f"Missing Implementation for {self.__class__.__name__} : execute")

    def load(self, params):
        print(f"Missing Implementation for {self.__class__.__name__} : load")
