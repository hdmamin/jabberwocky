# Jabberwocky-GUI

### Usage Notes

The first time you speak, the speech transcription backend will take a few seconds to calibrate to the level of ambient noise in your environment. You will know it's ready for transcription when you see a "Listening..." message appear below the Record button. Calibration only occurs once to save time.

### Built-in Tasks

Task Mode provides a miscellaneous assortment of built-in prompts for the following tasks:

- Summarization
- Explain like I'm 5
- Translation
- How To (step by step instructions for performing everyday tasks)
- Writing Style Analysis
- Explain machine learning concepts in simple language
- Generate ML paper abstracts
- MMA Fight Analysis and Prediction

### Hotkeys

**CTRL + SHIFT**: Start recording audio (same as pressing the "Record" button).  
**CTRL + a**: Get GPT-3's response to whatever input you've recorded (same as pressing the "Get Response" button).


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
