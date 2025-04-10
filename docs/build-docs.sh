#!/bin/bash
rm -rf build/

# Move theming elements into the docs folder
cp -R pytket-docs-theming/_static .
cp -R pytket-docs-theming/quantinuum-sphinx .
cp pytket-docs-theming/conf.py .

# Get the name of the project
EXTENSION_NAME="$(basename "$(dirname `pwd`)")"

# Correct github link in navbar
sed -i '' 's#CQCL/tket#CQCL/'$EXTENSION_NAME'#' _static/nav-config.js

# Build the docs. Ensure we have the correct project title.
sphinx-build -b html -D html_title="$EXTENSION_NAME" . build 

if [[ "$OSTYPE" == "darwin"* ]]; then
    find build/ -type f -name "*.html" | xargs sed -e 's/pytket._tket/pytket/g' -i ""
    sed -i '' 's/pytket._tket/pytket/g' build/searchindex.js
else
    find build/ -type f -name "*.html" | xargs sed -i 's/pytket._tket/pytket/g'
    sed -i 's/pytket._tket/pytket/g' build/searchindex.js
fi

# Remove copied files. This ensures reusability.
rm -r _static 
rm -r quantinuum-sphinx
rm conf.py