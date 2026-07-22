# precon3d

## Getting Started

To use the features of this template, you will need to modify several aspects after import, including:

- Update pyproject.toml to reflect the name of your project ("python_template" for this repo) and include dependencies. This will also affect paths for command line entry points near the bottom of the .toml file
- Modify the name of the subdirectory in src/ to match the name of your project ("python_template" for this repo). The included CI/CD configuration expect the standard src layout employed by the template and will not work otherwise.
- Modify the imports for test files to point to the name of your project ("python_template" for this repo). Only an example function in the "sum" module is included as a test.
- Update links for the badges above here in the readme, as well as for the project badges under repo -> General Settings -> Badges. Note that this template is in the 3DMatChar group and the template lives at the following group subdirectory: templates/python_template. Adjust links accordingly. Badges will not render correctly until the above other changes are made and the CI/CD pipeline runs successfully.

## Installing a package

- setup your git keys
- install python to make sure you have a working version

```bash
python --version
python3 --version
python3 -m pip install --upgrade pip setuptools #global upgrade
```

- setup proxy stuff
- create virtual environment

```bash
python3 -m venv .venv #make venv
```

- activate the environment

```bash
source .venv/bin/activate       # for bash shell
source .venv/bin/activate.csh   # for c shell
source .venv/bin/activate.fish  # for fish shell
.\.venv\Scripts\activate        # for powershell
```

- install the package with pip

```bash
pip install -e .[dev]  # install in dev mode, with editable
```

## Package Notes

- src structure: https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html
- test coverage

```bash
pytest --cov=src --cov-report=xml:coverage_reports/coverage.xml --cov-report=html:coverage_reports/htmlcov --cov-report=term tests/
```

- mkdocs
  For full documentation visit [mkdocs.org](https://www.mkdocs.org).

```bash
mkdocs new [dir-name] # Create a new project.
mkdocs serve # Start the live-reloading docs server.
mkdocs build # Build the documentation site.
mkdocs -h # Print help message and exit.

```

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

- api docs

```bash
env:
  PROJECT_NAME: python_template
echo "Project Name: $PROJECT_NAME"
pdoc ./src/$PROJECT_NAME/  -o ./docs/api
pdoc ./src/python_template/  -o ./docs/api
```
