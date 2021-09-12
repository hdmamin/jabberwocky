<p align='center'>
<img src='https://github.com/hdmamin/jabberwocky/raw/347e1514769264219393abf8a62b1a808cb70421/data/icons/icon.png' height='100'>
<h1 align='center'>Jabberwocky</h1>
</p>

https://user-images.githubusercontent.com/40480855/132139847-0d0014b9-022e-4684-80bf-d46031ca4763.mp4

This was not really designed to be used as a standalone library - it was mostly used as a convenient way to structure and import code in other parts of the [project](https://github.com/hdmamin/jabberwocky). Some components may be reusable for other projects combining GPT-3 with audio, however.

# Project Description
The core library powering a GUI that provides an audio interface to GPT3. We support conversations with most public figures and provide a number of other pre-built tasks including:
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

![](https://github.com/hdmamin/jabberwocky/raw/347e1514769264219393abf8a62b1a808cb70421/data/clips/demo/add_persona.gif)

In task mode, you can ask GPT3 to perform a number pre-defined tasks. Written and spoken input are both supported. By default, GPT3's response is both displayed in writing and read aloud.

![](https://github.com/hdmamin/jabberwocky/raw/c48600f88d8127911c96de138ce09f6ef97377eb/data/clips/demo/punctuation.gif)
Transcripts of responses from a subset of non-conversation tasks can be found in the `data/completions` directory.

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

---
Start of auto-generated file data.<br/>Last updated: 2021-09-12 13:53:34

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
      <td>__init__.py</td>
      <td>_</td>
      <td>1</td>
      <td>2021-09-04 13:32:08</td>
      <td>22.00 b</td>
    </tr>
    <tr>
      <td>config.py</td>
      <td>Define constants used throughout the project.</td>
      <td>21</td>
      <td>2021-07-22 20:29:41</td>
      <td>564.00 b</td>
    </tr>
    <tr>
      <td>core.py</td>
      <td>Core functionality that ties together multiple APIs.</td>
      <td>609</td>
      <td>2021-09-06 13:39:53</td>
      <td>24.21 kb</td>
    </tr>
    <tr>
      <td>external_data.py</td>
      <td>Functionality to fetch and work with YouTube transcripts.</td>
      <td>281</td>
      <td>2021-08-06 20:25:01</td>
      <td>9.97 kb</td>
    </tr>
    <tr>
      <td>openai_utils.py</td>
      <td>Utility functions for interacting with the gpt3 api.</td>
      <td>1320</td>
      <td>2021-09-01 20:12:51</td>
      <td>55.00 kb</td>
    </tr>
    <tr>
      <td>speech.py</td>
      <td>Module to help us interact with mac's speech command. This lets the GUI read<br/>responses out loud.</td>
      <td>117</td>
      <td>2021-08-20 21:05:11</td>
      <td>4.16 kb</td>
    </tr>
    <tr>
      <td>utils.py</td>
      <td>General purpose utilities.</td>
      <td>337</td>
      <td>2021-08-04 20:02:18</td>
      <td>10.93 kb</td>
    </tr>
  </tbody>
</table>
<br/>End of auto-generated file data. Do not add anything below this.
