# Jabberwocky-GUI

## Usage Notes

The first time you speak, the speech transcription backend will take a few seconds to calibrate to the level of ambient noise in your environment. You will know it's ready for transcription when you see a "Listening..." message appear below the Record button. Calibration only occurs once to save time.

## Built-in Tasks

Task Mode provides a miscellaneous assortment of built-in prompts for the following tasks:

- Summarization
- Explain like I'm 5
- Translation
- How To (step by step instructions for performing everyday tasks)
- Writing Style Analysis
- Explain machine learning concepts in simple language
- Generate ML paper abstracts
- MMA Fight Analysis and Prediction

## Hotkeys

**CTRL + SHIFT**: Start recording audio (same as pressing the "Record" button).  
**CTRL + a**: Get GPT-3's response to whatever input you've recorded (same as pressing the "Get Response" button).

## Getting Started

Use the following steps to set up your environment and run the GUI.

1. Clone the repo.

```
git clone https://github.com/hdmamin/jabberwocky.git
```

2. Install the necessary packages. I recommend using a virtual environment of some kind (virtualenv, conda, etc). If you're using Mac OS and virtualenv, you can use the command

```
make gui_env
```

to create a virtual environment for the GUI. If you're not using Mac OS or prefer to use a different environment manager, you can view `gui/make_env.sh` to see what logic is actually being executed. Note: the alexa skill uses a newer version of jabberwocky that is not backward compatible, so you can't run the GUI and the skill in the same virtual environment.


3. Add your openai API key somewhere the program can access it. There are two ways to do this (make sure to use your actual key, not the literal text `your_openai_api_key`):

```
echo your_openai_api_key > ~/.openai
```

or

```
export OPENAI_API_KEY=your_openai_api_key
```

If you plan to use other backends like huggingface, goose.ai, or banana.dev, you should make their api key(s) available too.

```
echo your_gooseai_api_key > ~/.gooseai
echo your_huggingface_api_key > ~/.huggingface
echo your_banana_api_key > ~/.banana
```

4. Run the gui from inside your virtual environment.

```
source gui/venv/bin/activate
python gui/main.py
```

`make run_gui` also works.

**Developer Tip:** If you plan to push new changes to github, I'd recommend using the command `make hooks` to install a git pre-commit hook to prevent you from accidentally exposing your openai API key. (This shouldn't happen regardless but the hook provides some layer of safety in case you print your key in a notebook or something.) You only need to run this once. You can also use the file `pre-commit.py` in the project root as a reference for creating your own hook.


---
Start of auto-generated file data.<br/>Last updated: 2022-06-25 11:03:46

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>File</th>
      <th>Summary</th>
      <th>Line Count</th>
      <th>Last Modified</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>callbacks.py</td>
      <td>Callbacks for our dearpygui app. Note that this module relies on a number of<br/>global variables which are defined after import in main.py:<br/>-APP<br/>-SPEAKER<br/>-CHUNKER<br/>-CONV_MANAGER<br/>-MANAGER<br/>-GENDER2VOICE<br/>-MODEL_NAMES<br/>-NAME2TASK<br/><br/>Pycharm highlights these as errors since they appear to be undefined when<br/>looking at this module in isolation.</td>
      <td>995</td>
      <td>2022-05-21 17:43:01</td>
      <td>43.53 kb</td>
    </tr>
    <tr>
      <td>main.py</td>
      <td>Launches a GUI providing an audio interface to GPT3.<br/><br/>Example usage:<br/>--------------<br/>python gui/main.py</td>
      <td>765</td>
      <td>2022-05-21 17:43:01</td>
      <td>40.10 kb</td>
    </tr>
    <tr>
      <td>utils.py</td>
      <td>General utilities used in our GUI that aren't callbacks. Note: this relies<br/>on the SPEAKER variable defined in main.py (main.py makes this variable<br/>available after importing utils - just making note of it here since pycharm<br/>will show errors since SPEAKER is undefined at definition time).</td>
      <td>364</td>
      <td>2021-09-12 13:54:19</td>
      <td>14.33 kb</td>
    </tr>
  </tbody>
</table>
<br/>End of auto-generated file data. Do not add anything below this.
