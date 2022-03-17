.PHONY: todo nb clean lib pypi readmes run

# Convention: add _ between comment sign and TODO to hide an item that you don't want to delete entirely. This will still be findable if you run `ack TODO`.
todo:
		ack -R '# TODO' {gui,lib,notebooks,notes,reports} || :

nb:
		cp notebooks/TEMPLATE.ipynb notebooks/nb000-untitled.ipynb

clean:
		cd lib && rm -rf dist
 
lib: clean
		cd lib && python setup.py sdist
 
pypi: lib
		cd lib && twine upload --repository pypi dist/*

readmes:
	htools update_readmes "['gui', 'notebooks', 'lib/jabberwocky']"

install:
	brew install portaudio
	pip install -r requirements.txt
	python -m nltk.downloader punkt

install_dev:
	brew install portaudio
	pip install --upgrade --force-reinstall -r requirements-dev.txt
	python -m nltk.downloader punkt
 
run:
	python gui/main.py
