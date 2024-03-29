# Notebooks

Jupyter notebooks for data exploration and experimentation.


---
Start of auto-generated file data.<br/>Last updated: 2022-07-24 14:12:51

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>File</th>
      <th>Summary</th>
      <th>Code Cell Count</th>
      <th>Markdown Cell Count</th>
      <th>Last Modified</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TEMPLATE.ipynb</td>
      <td>`# TODO: write summary`</td>
      <td>3</td>
      <td>1</td>
      <td>2021-05-25 21:31:16</td>
      <td>1.30 kb</td>
    </tr>
    <tr>
      <td>nb01-openai-api-playground.ipynb</td>
      <td>Try out openai api and develop a few convenience functions for the project.</td>
      <td>69</td>
      <td>3</td>
      <td>2021-05-08 14:16:31</td>
      <td>53.58 kb</td>
    </tr>
    <tr>
      <td>nb014-redo-static-stream-func.ipynb</td>
      <td>Try rewriting generator that streams from static backends. Want to do stopword truncation upfront. Might also be able to make this MUCH simpler.</td>
      <td>13</td>
      <td>2</td>
      <td>2022-05-09 20:59:57</td>
      <td>15.84 kb</td>
    </tr>
    <tr>
      <td>nb014-test-batch-queries.ipynb</td>
      <td>Test out new batch query functionality.</td>
      <td>25</td>
      <td>4</td>
      <td>2022-04-18 20:48:26</td>
      <td>67.54 kb</td>
    </tr>
    <tr>
      <td>nb015-fix-openai-api-issue.ipynb</td>
      <td>Recently (possibly since upgrading to openai 0.18.1?), querying openai causes invalid URL errors. Here are some notes from yesterday's troubleshooting session.<br/><br/>- curl works, python doesn't.<br/>- Not just codex, now nox openai engines work w/ python. Maybe due to updating pip package? Temporarily upped billing limit to try other models.<br/>- Restarted kernel and gpt query works again w/ ada. BUT after I import openai explicitly, that fails too. That must be a clue.<br/>- Tried uninstalling, reinstalling, opened new tmux pane. Still same error.<br/>- Tried deleting 'openai' object and then importing jabberwocky. This does work!?<br/>- If I re-import openai after that, GPT.query still works. But openai.completion while codex does not.<br/>- If I import openai FROM jabberwocky openai_utils, codex query still fails. But GPT.query works. And openai.Completion works w/ engine ada!<br/>- Conclusion: maybe it is codex-specific then?</td>
      <td>146</td>
      <td>16</td>
      <td>2022-05-09 20:53:26</td>
      <td>1.58 mb</td>
    </tr>
    <tr>
      <td>nb016-prompt-object.ipynb</td>
      <td>Experimenting with making a Prompt object. Goals:<br/>- simplify the call to resolve template + arg(s)<br/>- allow computed values? (e.g. accept arg x and then fill another field with x+3 or x.upper()}<br/>- maybe define postprocessing/completion validation steps?</td>
      <td>65</td>
      <td>7</td>
      <td>2022-05-27 15:41:34</td>
      <td>66.36 kb</td>
    </tr>
    <tr>
      <td>nb017-nationality-extraction-and--emotion-detection.ipynb</td>
      <td>SMS dataset from here:<br/>https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset<br/>It looks ver messy.</td>
      <td>96</td>
      <td>7</td>
      <td>2022-05-28 17:44:03</td>
      <td>527.64 kb</td>
    </tr>
    <tr>
      <td>nb018-persona-name-fuzzy-matching.ipynb</td>
      <td>Try fuzzy matching mis-transcribed names. <br/><br/>Maybe also try to handle this better for a new contact - e.g. search google/wikipedia to find who I *meant* rather than who Alexa *heard* me say.</td>
      <td>36</td>
      <td>4</td>
      <td>2022-06-08 20:31:55</td>
      <td>104.28 kb</td>
    </tr>
    <tr>
      <td>nb019-load-past-conversation.ipynb</td>
      <td>Experiment with parsing an old conv file. Eventually could allow resuming an old conv, although that may be out of scope for this round.</td>
      <td>20</td>
      <td>2</td>
      <td>2022-06-09 20:33:22</td>
      <td>65.84 kb</td>
    </tr>
    <tr>
      <td>nb02-youtube-api-playground.ipynb</td>
      <td>Get a feel for the youtube transcript API. Some questions to address:<br/><br/>- Do all/most videos have transcripts available?<br/>- How good is transcript quality?<br/>- How long are time chunks? Do they correspond to sentence start/ends?<br/>- For videos with multiple people, do we know who's talking when?</td>
      <td>144</td>
      <td>6</td>
      <td>2021-05-08 14:16:31</td>
      <td>348.91 kb</td>
    </tr>
    <tr>
      <td>nb020-speedup-lshdict-creation.ipynb</td>
      <td>See if we can speed up lshdict instantiation.</td>
      <td>36</td>
      <td>6</td>
      <td>2022-06-19 16:59:18</td>
      <td>53.99 kb</td>
    </tr>
    <tr>
      <td>nb021-price-monitor.ipynb</td>
      <td>Monitor api usage over time so we can shut things down before things get out of hand if someone figures out a way to start calling the api and using my api key.</td>
      <td>24</td>
      <td>3</td>
      <td>2022-06-30 20:41:29</td>
      <td>129.55 kb</td>
    </tr>
    <tr>
      <td>nb03-transcript-manager.ipynb</td>
      <td>Start tying openai and youtube functionality together to manage the punctuation process.</td>
      <td>138</td>
      <td>3</td>
      <td>2021-05-11 20:56:59</td>
      <td>182.56 kb</td>
    </tr>
    <tr>
      <td>nb04-multi-video-session.ipynb</td>
      <td>Start chaining together punctuation and another task.</td>
      <td>20</td>
      <td>1</td>
      <td>2021-05-18 21:19:27</td>
      <td>28.45 kb</td>
    </tr>
    <tr>
      <td>nb05-gpt-neo-api.ipynb</td>
      <td>`Try getting GPT Neo predictions using the Huggingface API.`</td>
      <td>14</td>
      <td>1</td>
      <td>2021-06-07 21:07:19</td>
      <td>18.28 kb</td>
    </tr>
    <tr>
      <td>nb06-interruptable-decorator.ipynb</td>
      <td>Make decorator to make function handle keyboard interrupt more easily. Hoping to use this on Speaker.speak() in GUI.<br/><br/>UPDATE: realized I already wrote a serviceable version of this. Made a few htools tweaks, no need to use the rest of this notebook.</td>
      <td>18</td>
      <td>2</td>
      <td>2021-05-29 18:36:28</td>
      <td>12.41 kb</td>
    </tr>
    <tr>
      <td>nb07-wiki-summary.ipynb</td>
      <td>Make wiki_summary function more capable of handling missing or ambiguous cases. Also see if image retrieval is easy/possible.</td>
      <td>50</td>
      <td>3</td>
      <td>2021-07-04 13:51:04</td>
      <td>48.10 kb</td>
    </tr>
    <tr>
      <td>nb08-conversation-manager.ipynb</td>
      <td>Trying to figure out how ConversationManager will work. Having a hard time planning this out so I'm thinking it may be best to try one approach to building it, see what issues arise, and then it will be easier to fix them. Start by trying to subclass PromptManager.</td>
      <td>56</td>
      <td>4</td>
      <td>2021-06-25 20:41:44</td>
      <td>66.47 kb</td>
    </tr>
    <tr>
      <td>nb09-conv-manager-longform-rewrite.ipynb</td>
      <td>ConversationManager refactor attempt. Trying to change its interface so it can more effectively:<br/>1. Support longer conversations via prompting with a subset of past responseses,<br/>2. Support longer conversations via summarizing past conv, and<br/>3. Still work with my GUI.</td>
      <td>64</td>
      <td>3</td>
      <td>2021-08-04 20:02:18</td>
      <td>92.85 kb</td>
    </tr>
    <tr>
      <td>nb10-concurrent-speaker.ipynb</td>
      <td>_</td>
      <td>36</td>
      <td>4</td>
      <td>2021-08-31 20:28:54</td>
      <td>38.01 kb</td>
    </tr>
    <tr>
      <td>nb11-slot-extraction.ipynb</td>
      <td>Trying to extract slots when alexa fails to recognize the desired intent.</td>
      <td>29</td>
      <td>6</td>
      <td>2022-03-26 14:17:49</td>
      <td>142.44 kb</td>
    </tr>
    <tr>
      <td>nb12-test-backend-selector.ipynb</td>
      <td>Make sure updated BackendSelector class is working properly. DO NOT PUSH THIS TO GITHUB (added to gitignore so api keys are not exposed).<br/><br/>UPDATE: renamed to GPTBackend.</td>
      <td>79</td>
      <td>7</td>
      <td>2022-04-08 21:26:53</td>
      <td>121.99 kb</td>
    </tr>
    <tr>
      <td>nb13-multi-prompt-queries.ipynb</td>
      <td>Experiment with ways to support passing a list of prompts to query method. Some backends don't support this natively, others do, but none automatically would return the format I want.</td>
      <td>136</td>
      <td>7</td>
      <td>2022-04-14 20:43:18</td>
      <td>194.95 kb</td>
    </tr>
  </tbody>
</table>
<br/>End of auto-generated file data. Do not add anything below this.
