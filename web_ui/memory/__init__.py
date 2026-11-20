"""Experience database for the browser UI (`slic serve --memory-dir <dir>`): records runs, replays them, and offers stored
designs as warm starts. See web_ui/README.md; routes.py describes the endpoints, store.py the files, ranking.py the ranking."""
from .routes import enable_memory

__all__ = ["enable_memory"]
