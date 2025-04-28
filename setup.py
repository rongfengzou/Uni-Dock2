import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext



class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "", namespace: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.namespace = namespace


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, "-DBUILD_TEST=FALSE"], cwd=build_temp, check=True
        )
        subprocess.run(
            ["make", "-j"], cwd=build_temp, check=True
        )

        shutil.copytree(build_temp, Path(self.build_lib) / ext.namespace / ext.name)


setup(
    ext_modules=[CMakeExtension("unidock_engine", "unidock/unidock_engine")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
