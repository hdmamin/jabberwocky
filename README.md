# jabberwocky

<img src='data/icons/icon.png' height='100'>

# Project Description
A GUI providing an audio interface to GPT3. We support conversations with most public figures and provide a number of other pre-built tasks including:
- Summarization
- Explain like I'm 5
- Translation
- How To (step by step instructions for performing everyday tasks)
- Writing Style Analysis
- MMA Fight Analysis and Prediction
- Explain machine learning concepts in simple language
    - Generate ML paper abstracts

### Examples

In conversation mode, you can chat with a number of pre-defined personas or add new ones:

![](data/clips/demo/add_persona.gif)

In task mode, you can ask GPT3 to perform a number pre-defined tasks. Written and spoken input are both supported. By default, GPT3's response is both displayed in writing and read aloud.

![](data/clips/demo/punctuation.gif)
Transcripts of responses from a subset of non-conversation tasks can be found in the `data/completion` directory.

### Project Members
* Harrison Mamin

### Repo Structure
```
jabberwocky/
├── data         # Raw and processed data. Actual files are excluded from github.
├── notes        # Miscellaneous notes stored as raw text files.
├── notebooks    # Jupyter notebooks for experimentation and exploratory analysis.
├── reports      # Markdown reports (performance reports, blog posts, etc.)
├── bin          # Executable scripts to be run from the project root directory.
├── lib          # Python package. Code can be imported in analysis notebooks, py scripts, etc.
└── services     # Serve model predictions through a Flask/FastAPI app.
```
