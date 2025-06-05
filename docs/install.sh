set -e


# Copy over poetry dependencies from theming repository
cp docs/pytket-docs-theming/extensions/pyproject.toml .
cp docs/pytket-docs-theming/extensions/poetry.lock .

# Install the docs dependencies. Creates a .venv directory in docs
poetry install

# NOTE: Editable wheel should be installed separately.
set +e