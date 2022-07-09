.PHONY: todo nb clean lib pypi readmes run prompt persona install install_dev install_alexa

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
	htools update_readmes "['gui', 'notebooks', 'lib/jabberwocky', 'alexa']"
	python ./lib/jabberwocky/cli.py update_prompt_readme

install:
	brew install portaudio
	pip install -r requirements.txt
	python -m nltk.downloader punkt

install_dev:
	brew install portaudio
	pip install --upgrade --force-reinstall -r requirements-dev.txt
	python -m nltk.downloader punkt

install_alexa:
	chmod u+x ./alexa/make_env.sh
	./alexa/make_env.sh
 
run:
	python gui/main.py

ngrok:
	ngrok http 5000

run_alexa:
	python alexa/app.py

hooks:
	ln -sf ~/jabberwocky/pre-commit.py .git/hooks/pre-commit
	chmod u+x .git/hooks/pre-commit

prompt:
	cp data/templates/prompt.yaml data/prompts/NEW_PROMPT.yaml
	vi data/prompts/NEW_PROMPT.yaml

persona:
	mkdir data/conversation_personas_custom/NEW_PERSONA
	cp data/templates/persona.yaml data/conversation_personas_custom/NEW_PERSONA/meta.yaml
	vi data/conversation_personas_custom/NEW_PERSONA/meta.yaml
