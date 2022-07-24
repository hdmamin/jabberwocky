# Run from current directory.
virtualenv venv
source venv/bin/activate
brew install portaudio
# Activating venv in script does change results of `which pip` and `which python` but installation was still occurring in global environment. Explicitly referencing pip and python within the venv subdir seems to fix this.
venv/bin/pip install -r requirements.txt
venv/bin/python3 -m nltk.downloader punkt

