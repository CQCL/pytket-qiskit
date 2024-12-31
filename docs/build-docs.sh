#!/bin/bash
rm -rf build/

# This build script is only used for local docs build.
# The docs build for the website uses a different script.

# Move theming elements into the docs folder
cp -R pytket-docs-theming/_static .
cp -R pytket-docs-theming/quantinuum-sphinx .
cp pytket-docs-theming/conf.py .

# Get the name of the project
PACKAGE="$(basename "$(dirname `pwd`)")"

# Get pytket extension version
VERSION="$(pip show $PACKAGE | grep Version | awk '{print $2}')"

# Output package version
echo extension version $VERSION

# Combine to set title
PACKAGE+=" $VERSION"

# Build the docs setting the html_title
sphinx-build -b html . build -D html_title="$PACKAGE API documentation" -W

# Find and replace all generated links that use _tket in the built html.
# Note that MACOS and linux have differing sed syntax.
if [[ "$OSTYPE" == "darwin"* ]]; then
    find build/ -type f -name "*.html" | xargs sed -e 's/pytket._tket/pytket/g' -i ""
    sed -i '' 's/pytket._tket/pytket/g' build/searchindex.js
else
    find build/ -type f -name "*.html" | xargs sed -i 's/pytket._tket/pytket/g'
    sed -i 's/pytket._tket/pytket/g' build/searchindex.js
fi

# Remove copied files after build is done. This ensures reusability.
rm -r _static 
rm -r quantinuum-sphinx
rm conf.py