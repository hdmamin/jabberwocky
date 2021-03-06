set -a
. setenv.sh
docker run -it -v $CONVERSATIONS_PATH:/gui/data/conversations \
    -v $COMPLETIONS_PATH:/gui/data/completions \
    -v $PROMPTS_PATH:/gui/data/prompts \
    -v $PERSONAS_PATH:/gui/data/conversation_personas \
    -v $CUSTOM_PERSONAS_PATH:/gui/data/conversation_personas_custom \
    -v $LOGS_PATH:/gui/data/logs \
    -v $FONTS_PATH:/gui/data/fonts \
    -v $BIN_PATH:/gui/gui \
    -v ~/.openai:/root/.openai \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$IP:0 \
    jabberwocky /gui/bash
