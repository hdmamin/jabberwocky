
---
Start of auto-generated file data.<br/>Last updated: 2022-06-25 11:03:48

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
      <td>app.py</td>
      <td>Flask web server where each alexa intent has its own endpoint. Responses<br/>that don't match a recognized intent (most of them) are routed to the<br/>`delegate` endpoint which attempts to infer the intended intent. Often this is<br/>just a standard conversational reply, in which case the user text is used to<br/>continue a conversation with a GPT-powered persona. Functions can be enqueued<br/>to handle cases like yes/no questions (in hindsight, a finite state machine<br/>would likely have been a more elegant solution here, but I was not that<br/>familiar with them when I started building this).<br/><br/>Examples<br/>--------<br/># Default mode. This uses the openai backend, loads both custom and<br/># auto-generated personas, and uses AI-generated voices via Amazon Polly.<br/>python alexa/app.py<br/><br/># Run the app in dev mode (use free gpt backend by default) and only load<br/># auto-generated personas (people with wikipedia pages).<br/>python alexa/app.py --dev True --custom False<br/><br/># Disable Amazon Polly voices in favor of default alexa voice for everyone.<br/># Downside is persona voices are no longer differentiated by<br/># gender/nationality; upside is we get some primitive emotional inflections<br/># when appropriate.<br/>python alexa/app.py --voice False</td>
      <td>812</td>
      <td>2022-06-25 10:44:05</td>
      <td>31.60 kb</td>
    </tr>
    <tr>
      <td>config.py</td>
      <td>Constants used in our alexa skill. The email is used to send transcripts<br/>to the user when desired. Note that in addition to the log file specified here,<br/>GPT queries are also logged to files like `2022.06.25.jsonlines`. A new file is<br/>generated for each day (the switch occurs at midnight) and each line in the<br/>file corresponds to kwargs for a single gpt query.</td>
      <td>9</td>
      <td>2022-06-24 19:16:41</td>
      <td>435.00 b</td>
    </tr>
    <tr>
      <td>utils.py</td>
      <td>Helpers used in our alexa app. Particularly important<br/>- CustomAsk class: subclass of one of the core flask-ask classes. Lots of<br/>functionality around callbacks/logging/etc. take place here.<br/>- Settings class: used to maintain gpt settings throughout a session. There are<br/>global-level, conversation-level, and person-level settings (recency is<br/>prioritized when resolving the final gpt query kwargs). You can read more about<br/>this system in its docstrings.<br/>- build_utterance_map function: any time we update the dialogue model in the<br/>alexa UI, we must download the model JSON, save it to<br/>jabberwocky/data/alexa/dialogue_model.json, and pass it to this function. This<br/>will save a new dict-like object that allows us to infer intents and slots when<br/>user utterances are not recognized as invoking an existing intent.</td>
      <td>1171</td>
      <td>2022-06-24 19:16:41</td>
      <td>44.25 kb</td>
    </tr>
  </tbody>
</table>
<br/>End of auto-generated file data. Do not add anything below this.
