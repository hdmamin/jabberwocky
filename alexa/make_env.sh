virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-no-deps.txt
python -m nltk.downloader punkt

