#!/bin/bash
rm -rf build/

# Move theming elements into the docs folder
cp -R pytket-docs-theming/_static .
cp pytket-docs-theming/conf.py .

# Get the name of the project
EXTENSION_NAME="$(basename "$(dirname `pwd`)")"

# Build the docs. Ensure we have the correct project title.
sphinx-build -W -b html -D html_title="$EXTENSION_NAME" . build || exit 1

sphinx-build -W -v -b coverage . build/coverage || exit 1

if [[ "$OSTYPE" == "darwin"* ]]; then
    find build/ -type f -name "*.html" | xargs sed -e 's/pytket._tket/pytket/g' -i ""
    sed -i '' 's/pytket._tket/pytket/g' build/searchindex.js
else
    find build/ -type f -name "*.html" | xargs sed -i 's/pytket._tket/pytket/g'
    sed -i 's/pytket._tket/pytket/g' build/searchindex.js
fi

# Remove copied files. This ensures reusability.
rm -r _static
rm conf.py