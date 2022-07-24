virtualenv venv --python=python3.7
source venv/bin/activate
# Activating venv in script does change results of `which pip` and `which python` but installation was still occurring in global environment. Explicitly referencing pip and python within the venv subdir seems to fix this.
venv/bin/pip3 install -r requirements.txt
venv/bin/pip3 install --no-deps -r requirements-no-deps.txt
venv/bin/python3 -m nltk.downloader punkt

