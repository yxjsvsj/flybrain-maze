from pathlib import Path

from flask import Flask

from .ground import GroundControl, register_ground_routes
from .routes import register_routes


def create_app(controller, guard_runner, sound_player=None) -> Flask:
    root = Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static"),
    )
    app.config["SERIAL_CONTROLLER"] = controller
    app.config["GUARD_RUNNER"] = guard_runner
    app.config["SOUND_PLAYER"] = sound_player
    register_routes(app)
    ground = GroundControl(controller)
    app.config["GROUND_CONTROL"] = ground
    register_ground_routes(app, ground)
    return app
