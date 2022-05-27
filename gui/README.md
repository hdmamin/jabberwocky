# GUI

This folder is for scripts that can be run from the command line (usually python or shell scripts).


---
Start of auto-generated file data.<br/>Last updated: 2022-05-26 20:57:07

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
