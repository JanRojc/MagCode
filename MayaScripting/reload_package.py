import sys
import importlib

def reload_package(pkg_name):
    """Reload a package and all its submodules (depth-first)."""
    # collect all module names that belong to this package
    to_reload = [name for name in sys.modules
                 if name == pkg_name or name.startswith(pkg_name + ".")]

    # reload children first
    for name in sorted(to_reload, key=len, reverse=True):
        mod = sys.modules.get(name)
        if mod is not None:
            try:
                importlib.reload(mod)
                print("reloaded:", name)
            except Exception as e:
                print("FAILED to reload", name, "->", e)


reload_package("smpl_torch")