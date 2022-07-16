# Run from current directory.
virtualenv venv
source venv/bin/activate
brew install portaudio
pip install -r requirements.txt
python -m nltk.downloader punkt

