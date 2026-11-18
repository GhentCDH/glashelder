class UserError(Exception):
    """A mistake the user can fix, like a setting that points at the wrong file.

    The CLI shows only its message, without a traceback.
    """
