.PHONY: todo nb clean lib pypi readmes run_gui prompt persona gui_env alexa_env run_alexa alexa_logs ping
.SILENT: ngrok_url run_alexa alexa_logs persona prompt hooks ngrok

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

gui_env:
	cd gui && chmod u+x make_env.sh && ./make_env.sh

alexa_env:
	cd alexa && chmod u+x make_env.sh && ./make_env.sh
 
run_gui:
	# Backslash makes makefile run the app inside the env we just activated. It uses a new shell otherwise.
	# Don't use backticks in echo (habit from markdown) - this tries to execute the command inside.
	. gui/venv/bin/activate; \
	python gui/main.py
	# nohup python gui/main.py > /dev/null 2>&1 &
	# echo "You can use 'make alexa_logs' to view the latest logs from your running skill."

# Print public URL of the flask app behind your alexa skill. (This assumes it's already running publicly - if that's not the case, use `make ngrok` to both launch ngrok and print the URL it creates.)
ngrok_url:
	curl localhost:4040/api/tunnels | jq '.tunnels[0].public_url'

# Start ngrok in background and print the url in your terminal. Sleep briefly in between to avoid errors - I guess it can take some time for the URL to become available.
ngrok:
	nohup ngrok http 5000 > /dev/null &
	sleep 1
	$(MAKE) ngrok_url

run_alexa:
	# Backslash makes makefile run the app inside the env we just activated. It uses a new shell otherwise.
	# Don't use backticks in echo (habit from markdown) - this tries to execute the command inside.
	. alexa/venv/bin/activate; \
	nohup python alexa/app.py > /dev/null 2>&1 &
	echo "You can use 'make alexa_logs' to view the latest logs from your running skill."

ping:
	# Check that alexa skill is running and exposed via ngrok.
	# Had trouble reusing `make ngrok_url` inside makefile so just repeated logic here. This is a
	# make variable, not a shell variable.
	$(eval public_url := $(shell curl localhost:4040/api/tunnels | jq '.tunnels[0].public_url'))
	curl -X GET "$(public_url)/health"

alexa_logs:
	watch -n 1 -d tail alexa/app.log

hooks:
	ln -sf ~/jabberwocky/pre-commit.py .git/hooks/pre-commit
	chmod u+x .git/hooks/pre-commit

prompt:
	scripts/makeprompt.sh

persona:
	mkdir data/conversation_personas_custom/NEW_PERSONA
	cp data/templates/persona.yaml data/conversation_personas_custom/NEW_PERSONA/meta.yaml
	vi data/conversation_personas_custom/NEW_PERSONA/meta.yaml
