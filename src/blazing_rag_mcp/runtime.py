"""Compatibility module for older imports.

`Application` is the production lifecycle owner. `Runtime` remains as an alias so existing code
that imports it does not break.
"""

from .application import Application

Runtime = Application

__all__ = ["Application", "Runtime"]
