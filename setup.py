# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import Path

from setuptools import find_namespace_packages, setup  # type: ignore

metadata: dict = {}
with open("_metadata.py") as fp:
    exec(fp.read(), metadata)
shutil.copy(
    "_metadata.py",
    os.path.join("pytket", "extensions", "qiskit", "_metadata.py"),
)


setup(
    name="pytket-qiskit",
    version=metadata["__extension_version__"],
    author="TKET development team",
    author_email="tket-support@quantinuum.com",
    python_requires=">=3.10",
    project_urls={
        "Documentation": "https://docs.quantinuum.com/tket/extensions/pytket-qiskit/index.html",
        "Source": "https://github.com/CQCL/pytket-qiskit",
        "Tracker": "https://github.com/CQCL/pytket-qiskit/issues",
    },
    description="Extension for pytket, providing translation to and from the Qiskit "
    "framework",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_namespace_packages(include=["pytket.*"]),
    include_package_data=True,
    install_requires=[
        "pytket >= 2.0.1",
        "qiskit >= 1.4.2, < 2",
        "qiskit-ibm-runtime >= 0.37.0",
        "qiskit-aer >= 0.15.1",
        "numpy >= 1.26.4",
    ],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
