from pathlib import Path
from howtrader.trader.app import BaseApp
from howtrader.trader.engine import MainEngine


__version__ = "1.0.0"


class QuantMlApp(BaseApp):
    """"""

    app_name = "QuantMl"
    app_module = __module__
    app_path = Path(__file__).parent
    display_name: str = "量化机器学习"
    engine_class = MainEngine
    widget_name: str = "QuantMlManager"
    icon_name: str = "quantml.ico"

    def __init__(self, main_engine: MainEngine):
        """"""
        super().__init__(main_engine)

    def get_setting(self):
        """"""
        return {}

    def start(self, main_engine: MainEngine):
        """"""
        pass