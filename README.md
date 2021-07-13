# jabberwocky

# Project Description
A GUI providing an audio interface to GPT3. We support conversations with most public figures (basically anyone with a Wikipedia page) and provide a number of other modes including:
- Summarization
- Explain like I'm 5
- Translation
- How To (step by step instructions for performing everyday tasks)
- Writing Style Analysis
- MMA Fight Analysis and Prediction
- Explain machine learning concepts in simple language
    - Generate ML paper abstracts

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
