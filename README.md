<p align='center'>
<img src='data/icons/icon.png' height='100'>
<h1 align='center'>Jabberwocky</h1>
</p>

https://user-images.githubusercontent.com/40480855/132139847-0d0014b9-022e-4684-80bf-d46031ca4763.mp4

# Project Description

This GUI provides an audio interface to GPT-3. My main goal was to provide a convenient way to interact with various experts or public figures: imagine discussing physics with Einstein or hip hop with Kanye (or hip hop with Einstein? 🤔). I often find writing and speaking to be wildly different experiences and I imagined the same would be true when interacting with GPT-3. This turned out to be partially true - the default Mac text-to-speech functionality I'm using here is certainly not as lifelike as I'd like. Perhaps more powerful audio generation methods will pop up in a future release...

We also provide Task Mode containing built-in prompts for a number of sample tasks:

- Summarization
- Explain like I'm 5
- Translation
- How To (step by step instructions for performing everyday tasks)
- Writing Style Analysis
- Explain machine learning concepts in simple language
- Generate ML paper abstracts
- MMA Fight Analysis and Prediction

## Getting Started

1. Clone the repo.

```
git clone https://github.com/hdmamin/jabberwocky.git
```

2. Install the necessary packages. I recommend using a virtual environment of some kind (virtualenv, conda, etc). If you're not using Mac OS, you could try installing portaudio with whatever package manager you're using, but app behavior on other systems is unknown.

```
brew install portaudio
pip install -r requirements.txt
python -m nltk.downloader punkt
```

If you have `make` installed you can simply use the command:

```
make install
```

3. Add your openai API key somewhere the program can access it. There are two ways to do this:

```
echo your_openai_api_key > ~/.openai
```

or

```
export OPENAI_API_KEY=your_openai_api_key
```

(Make sure to use your actual key, not the literal text `your_openai_api_key`.)

4. Run the app.

```
python gui/main.py
```

Or with `make`:

```
make run
```

## Usage

### Conversation Mode

In conversation mode, you can chat with a number of pre-defined personas or add new ones. New personas can be autogenerated or defined manually. 

![](data/clips/demo/add_persona.gif)

See `data/conversation_personas` for examples of autogenerated prompts. You can likely achieve better results using custom prompts though.

Conversation mode only supports spoken input, though you can edit flawed transcriptions manually. Querying GPT-3 with nonsensical or ungrammatical text will negatively affect response quality.

### Task Mode

In task mode, you can ask GPT-3 to perform a number pre-defined tasks. Written and spoken input are both supported. By default, GPT-3's response is both typed out and read aloud.

![](data/clips/demo/punctuation.gif)
Transcripts of responses from a small subset of non-conversation tasks can be found in the `data/completions` directory. You can also save your own completions while using the app.

## Usage Notes

The first time you speak, the speech transcription back end will take a few seconds to calibrate to the level of ambient noise in your environment. You will know it's ready for transcription when you see a "Listening..." message appear below the Record button. Calibration only occurs once to save time.

### Hotkeys

**CTRL + SHIFT**: Start recording audio (same as pressing the "Record" button).  
**CTRL + a**: Get GPT-3's response to whatever input you've recorded (same as pressing the "Get Response" button).

### Project Members
* Harrison Mamin

### Repo Structure
```
jabberwocky/
├── data         # Raw and processed data. Some files are excluded from github but the ones needed to run the app are there.
├── notes        # Miscellaneous notes from the development process stored as raw text files.
├── notebooks    # Jupyter notebooks for experimentation and exploratory analysis.
├── reports      # Markdown reports (performance reports, blog posts, etc.)
├── gui          # GUI scripts. The main script should be run from the project root directory. 
└── lib          # Python package. Code can be imported in analysis notebooks, py scripts, etc.
```

The `docker` and `setup` dirs contain remnants from previous attempts to package the app. While I ultimately decided to go with a simpler approach, I left them in the repo so I have the option of picking up where I left off if I decide to work on a new version.
