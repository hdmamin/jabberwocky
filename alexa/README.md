# Jabberwocky-Alexa

I'm keeping this as a private skill for the time being - it was initially designed for a single user (me) and might require a redesign to handle more. Publishing it publicly would also likely require formal approval from OpenAI. In the meantime, you are free to use this repo to set up your own private skill. 

## Creating your own private skill

### Code Setup

1. Clone the repo.

```
git clone https://github.com/hdmamin/jabberwocky.git
```

2. Enter the project directory and install the necessary packages. I recommend using a virtual environment of some kind (virtualenv, conda, etc). If you're using Mac OS and virtualenv, you can use the command

```
cd jabberwocky
make alexa_env
```


If you're not using Mac OS or prefer to use a different environment manager, I suggest viewing `alexa/make_env.sh` to get an idea of the steps you need to implement. Note: the alexa skill uses a newer version of the jabberwocky library than the GUI does, and it's not backward compatible.

3. Add your openai API key somewhere the program can access it. (Make sure to use your actual key, not the literal text `your_openai_api_key`.)

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


4. Run the skill from inside your virtual environment.

```
source alexa/venv/bin/activate
python alexa/app.py
```

5. Use ngrok to expose your endpoint to Alexa. (In theory you can also use AWS Lambda for this, but that seems to be built for smaller skills with less state.)

```
make ngrok
```

This should print a url ending with "ngrok.io" in your terminal. Copy that url and run `curl {url}/health` (paste your actual url in place of {url}). The response should be:

```
{"status": 200, "source": "jabberwocky-alex"}
```

**Developer note:**

If you plan to push new changes to github, I'd recommend using the command `make hooks` to install a git pre-commit hook to prevent you from accidentally exposing your openai API key. (This shouldn't happen regardless but the hook provides some layer of safety in case you print your key in a notebook or something.) You only need to run this once. You can also use the file `pre-commit.py` in the project root as a reference for creating your own hook.


### AWS Console Setup

You will also need to create your skill in the alexa console. I imagine the exact process for this will change over time, but here's how to get started (as of July 2022):

Go to [alexa console](https://developer.amazon.com/alexa/console/ask) and click `Create Skill`.

Name your skill, then choose the following settings below.

1. Choose a model to add to your skill
> Custom

2. Choose a method to host your skill's backend resources
> Provision your own

(The above assumes you plan to use the ngrok deployment method rather than AWS Lambda.)

Scroll back up and click `Create Skill`. On the next page, select "Start from scratch" when asked to choose a template.

Interaction Model > Intents > JSON Editor

in the menu on the left side of your screen. Use `Drag and drop a .json file` to upload the file `data/alexa/dialog_model.json` from this repo. This will create all the intents and slots used by the skill. Click "Save Model" and then "Build Model" at the top of your screen. This will take a few minutes.

To point alexa towards the Flask app you started running earlier, go to the `endpoint` section of the left-hand-side menu and change your service endpoint type from "AWS Lambda ARN" to "HTTPS". Paste your ngrok URL from earlier (make sure it starts with "https" rather than "http" and excludes the "/health" suffix) into the "Default Region" text box. For SSL certificate type select "My development endpoint is a sub-domain of a domain that has a wildcard certificate from a certificate authority". Finally, click "Save Endpoints" at the top of your screen.

You can enable the skill on your phone's alexa app or on the [web version](https://alexa.amazon.com/spa/index.html). You can use the skill settings to enable access to your name (this lets your conversational partners address you by your first name). Enabling access to your email allows Jabberwocky to send you transcripts of your conversations when requested.

## Quickstart

**This chat log provides a minimal example of the dialogue needed to launch the skill, start a conversation with a new contact, and exit the skill when done. We assume you've already enabled your skill on your echo device using the steps above.**

> **User:** Alexa, start Quick Chat.<br>
> **Lou:** Hi there! Who would you like to speak to?<br>
> **User:** William Shakespeare<br>
> **Lou:** I don't see anyone named William Shakespeare in your contacts. Would you like to add a new contact?<br>
> **User:** Yes please.<br>
> **Lou:** I've connected you with William Shakespeare.<br>
> **User:** Hi William! I've been struggling with how to end my novel and I thought you could provide some input.

[User and William Shakespeare chat for a bit.]

...

> **User:** Thanks William! That really helped.<br>
> **William Shakespeare:** My pleasure.<br>
> **User:** Goodbye.

The above example exits the skill entirely, but you could also choose to end the current conversation and start a new one:

> **User:** Lou, hang up.<br>
> **Lou:** Would you like me to send you a transcript of your conversation?<br>
> **User:** Yes please.<br>
> **Lou:** I've sent you a transcript of your conversation. Who would you like to speak to next?<br>
> **User:** Mozart

When Lou asks who you want to speak to, you can also answer "Nobody" or "Noone" to exit the skill.

# Documentation

## Background

Jabberwocky, like most alexa skills, provides several `intents`. Think of these as actions you want to perform while using the skill. Rather than clicking on a button or menu as you might in an app or GUI, you invoke these using plain English. 

We document the available intents below. Each intent provides a few **Sample Utterances**: these are examples of things you can say that will trigger the intent. In practice, there may be hundreds or thousands of phrases that can trigger an intent. When examining the sample utterances, you may notice that any requests to change settings or in-skill options should be directed to Lou. Lou is your in-app assistant. (See the FAQ for details.)

Some of our intents accept one or more options (e.g. if you want to change the max length of your conversational replies, you need to provide a number). The **Options** subsection of an intent's documentation describes what values are acceptable.

Finally, note that there are a few intents that aren't explicitly documented here because they should already be pretty straightforward. E.g. if Lou asks you a yes or no question, you can respond with "Yes" or "No" (or "yes please" or "no thank you" if appropriate - I'm a proponent of treating human-like interfaces with the same basic respect you'd use when speaking with a living person). Once you've started a conversation, you can simply speak normally and it will be treated as a conversational response as long as it doesn't sound like you're trying to invoke one of the other intents.


## Launching the skill

> Alexa, start Quick Chat.

This causes your Echo to launch the skill.

## Scopes

A few intents can be used to change settings that affect the nature and overall quality of your conversations. Settings can be changed at different "scopes" which determine how long your changes persist. We use a "global" scope if you don't explicitly specify one when making a change. This may make more sense after reading through the rest of the intents section (for example, see the `changeModel` intent's Sample Utterances. `changeMaxLength` and `changeTemperature` also use scopes.).

Scope | Explanation
---|---
global | These changes will affect all people and all conversations.
person | These changes will be attached to the person you're currently speaking with (notice this means you must have started a conversation already). They will persist across future conversations with this same person but will not affect your conversations with other people.
conversation | These changes will only affect the current conversation. They will be un-done as soon as the conversation ends.

## Intents

#### `changeBackend`

Jabberwocky provides models from a number of different sources which we call "backends". Conversational quality, price, and other factors can vary dramatically by backend. You can change your backend at basically any time.

**Sample Utterances**
> Lou, use gooseai backend.<br>
> Lou, change backend to openai.<br>
> Lou, please switch backend to banana.<br>
> Lou, switch to huggingface backend.

**Options**

Backend | Free/Paid | Reliability | Conversational Quality
---|---|---|---
OpenAI | Paid | High | High
GooseAI | Paid | High | Moderate
Banana | Free | Moderate | Moderate
Huggingface | Free | Low | Moderate
Hobby | Free | Low | Moderate
Repeat | Free | High | Low
Mock | Free | High | Low

Backends with low "reliability" scores may often be unavailable or lead to frequent timeouts during a conversation. The "Repeat" and "Mock" backends are mostly useful for developer testing.


#### `changeModel`

Our paid backends provide multiple models (technically, so does Huggingface, but that backend often leads to timeouts and we don't recommend using it for a good conversational experience). We number them from 0 to 3, where higher values indicate more powerful models. For example, model 3 corresponds to the "text-davinci-002" model when using the openai backend or the "gpt-neo-20b" model when using the gooseai backend. (These mappings will likely change in the future as the backends release new models.)

Like most intents that change settings, you can specify a scope that will determine how long your changes persist (refer to the Scopes table at the start of the Intents section).

**Sample Utterances**
> Lou, use model 0.<br>
> Lou, change model to 1.<br>
> Lou, switch to model 2.<br>
> Lou, switch to global model 2.<br>
> Lou, use conversation model 0.<br>
> Lou, change person model to 3.

**Options**

Model | Conversational Quality
---|---
0 | Low
1 | Low
2 | Moderate
3 | High


#### `changeMaxLength`

This lets you change the maximum number of tokens to generate in a response. The max we allow is 900 but conversations will generally use much shorter responses (we set the default to 100, but most responses automatically conclude before that point). There are roughly 1.33 tokens in the average word.

Like most intents that change settings, you can specify a scope that will determine how long your changes persist (refer to the Scopes table at the start of the Intents section).

**Sample Utterances**

> Lou, change max length to 75.<br>
> Lou, set max length to 50.<br>
> Lou, set max tokens to 33.<br>
> Lou, set global max length to 90.<br>
> Lou, set conversation max length to 90.<br>
> Lou, set person max length to 100.

**Options**

0 < max length <= 900

#### `changeTemperature`

This lets you change the model "temperature". Lower values (near 0) are often better for formal or educational contexts, e.g. a science tutor. Higher values (near 100) are better for more creative or whimsical contexts. The default is 70. (A note for experienced language model users: while we ask you to provide an integer, we scale it to lie between 0 and 1 behind the scenes. This is just because it's easier for Alexa to parse integers.)

Like most intents that change settings, you can specify a scope that will determine how long your changes persist (refer to the Scopes table at the start of the Intents section).

**Sample Utterances**
> Lou, change temperature to 1.<br>
> Lou, set temp to 90.<br>
> Lou, set temperature to 20.<br>
> Lou, set global temperature to 45.<br>
> Lou, change persona temperature to 2.<br>
> Lou, change conversation temperature to 85.

**Options**

0 < temperature <= 100

#### `readContacts`

This lets you hear a list of all the people you can chat with. (Note that you can always create new contacts - when Lou asks who you'd like to speak to, simply respond with a name that's not in your contacts, then answer "Yes" when asked if you'd like to add a new contact.)

**Sample Utterances**
> Lou, who are my contacts?<br>
> Lou, please read me my contacts.<br>
> Lou, can you read me my contacts?


#### `readSettings`

This lets you hear all your current settings. This can be useful if you've changed a bunch of settings at different scopes and don't remember what they currently resolve to.

**Sample Utterance**
> Lou, what are my settings?<br>
> Lou, read me my settings.

#### `choosePerson`

This intent is invoked when Lou asks who you would like to speak to and you respond with a name. The person can be real or fictional. If they're not already in your contacts, jabberwocky tries to create the persona for you. 

**Sample Utterances**
> Maya Angelou<br>
> Harry Potter

#### `enableAutoPunctuation`

Use this to enable an experimental feature that uses a model to improve Alexa's transcription of your speech (insert punctuation, fix likely mistranscribed words, etc.). This may lead to higher quality responses and reduce the chance of miscommunications. However, it also makes responses take longer and we've observed frequent timeouts with this enabled. We may provide better support for this in a future release.

**Sample Utterances**
> Lou, please use auto punctuation.<br>
> Lou, enable automatic punctuation.<br>
> Lou, please turn on automatic punctuation.<br>
> Lou, turn on auto punctuation.

#### `disableAutoPunctuation`

Disable the experimental auto punctuation feature (see the `enableAutoPunctuation` intent described above). The feature is disabled by default so you only need to use this if you enable it and then want to disable it again.

**Sample Utterances**
> Lou, disable auto punctuation.<br>
> Lou, please disable automatic punctuation.<br>
> Lou please stop using auto punctuation.<br>
> Lou, turn off automatic punctuation.

#### `endChat`

End the conversation that's currently in progress. If you've given Jabberwocky permission to email you, Lou will ask you if you'd like to receive an emailed transcript of your conversation. This only exits the current conversation, whereas `Goodbye` exits the skill entirely.

**Sample Utterances**
> "Lou, hang up."<br>
> "Lou, end chat."

#### `Goodbye`

You can use this to exit the skill (and any ongoing conversation) immediately. This is the fastest way to exit. If you've given the skill permission to email you, you will automatically receive an emailed transcript of your conversation if one is in progress. (Normally, Lou asks you if you'd like a transcript or not, but we can't do that here since it's more of a hard exit. We err on the side of caution and always send a transcript rather than never sending it.)

**Sample Utterances**
> Goodbye.

## FAQ

**Who is Lou and why are they there?**

You can think of Lou as your personal assistant within Jabberwocky. Their presence is mostly a practical implementation detail: by starting all settings-changing commands with "Lou", it becomes easier to distinguish between requests to change settings and conversational responses. Granted, adding a contact named Lou might raise some problems, but I'll tackle that concern if/when it becomes an issue.

The name "Lou" is selected mostly because it's fast to say, reliably recognized by Alexa (voice transcription is still imperfect), and gender neutral (Lou uses your default Alexa voice, which can be male or female depending on your settings; the name "Lou" allows you to visualize whichever gender matches your settings).

## Dev Notes

`zappa_settings.json` is currently unused - I couldn't get this to work so far. The issue seems to be that my environment is just too big (I think). If you want to try this again, add these files back to `requirements.txt`, though perhaps with whatever the most up-to-date versions are is in the future. 

```
awscli==1.25.28
zappa==0.54.1
```

Then follow this tutorial:

 https://developer.amazon.com/blogs/post/8e8ad73a-99e9-4c0f-a7b3-60f92287b0bf/new-alexa-tutorial-deploy-flask-ask-skills-to-aws-lambda-with-zappa

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
