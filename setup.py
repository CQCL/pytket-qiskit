# Copyright 2020-2024 Quantinuum
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

import shutil
import os
from setuptools import setup, find_namespace_packages  # type: ignore

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
        "Documentation": "https://tket.quantinuum.com/extensions/pytket-qiskit/index.html",
        "Source": "https://github.com/CQCL/pytket-qiskit",
        "Tracker": "https://github.com/CQCL/pytket-qiskit/issues",
    },
    description="Extension for pytket, providing translation to and from the Qiskit "
    "framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_namespace_packages(include=["pytket.*"]),
    include_package_data=True,
    install_requires=[
        "pytket ~= 1.29",
        "qiskit ~= 1.1",
        "qiskit-algorithms ~= 0.3.0",
        "qiskit-ibm-runtime ~= 0.24.1",
        "qiskit-aer ~= 0.14.2",
        "numpy < 2",
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
