def _package_version() -> str:
    from importlib.metadata import version

    return version(__package__)


__version__ = _package_version()
