"""Where a recording goes: a file, a running viewer, or a freshly spawned one."""

from pathlib import Path

import rerun as rr

APP_ID = "glaswacht"


def init(
    out: str | Path | None = None,
    connect: str | None = None,
    blueprint: rr.blueprint.Blueprint | None = None,
    spawn: bool = True,
    app_id: str = APP_ID,
) -> None:
    """Start a recording. `connect` wins over `out`; neither means spawn a viewer.

    The blueprint is also sent as the *active* one: Rerun otherwise reopens whatever
    layout it last used for this application id.
    """
    rr.init(app_id, spawn=False, default_blueprint=blueprint)
    if connect:
        rr.connect_grpc(connect, default_blueprint=blueprint)
    elif out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        rr.save(out, default_blueprint=blueprint)
    elif spawn:
        rr.spawn(default_blueprint=blueprint)
    if blueprint is not None:
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
