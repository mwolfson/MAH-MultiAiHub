# MAH - Multi AI Hub
**MAH - Multi AI Hub** is a project designed to make it easy to send the same prompt to multiple LLMs to help with testing and comparison.

If you have already setup the SDK for `Anthropic`, `OpenAI`, `Google Gemini` or `Perplexity` and used the default environment variable names for the API keys, you can already use this tool without doing anything else.

## Usage

Copy the `MultiAiHub.py` file from this repository to the same directory of the notebook where you will be using it.

### Install the AI Vendor SDKs and set API Keys

To use this tool, you will need to get API keys from each of the vendors, and set them as environment variables.

Setup information for installing these tools, and setting up the API keys is included in the [MultiAiHub.ipynb](./MultiAiHub.ipynb) notebook. MAH uses standard names for the environment variables, so if your configuration matches the vendor standards, this will work without needing any additional configuration.

## Prompt Multiple AIs at Once

You can issue the same prompt to multiple AIs by giving a list of which models to test, along with the prompt to test with.

```python
# Import the code
import MultiAiHub as mah

# List of each model to test the prompt with
models = [
    mah.ANTHROPIC_OPUS,
    mah.ANTHROPIC_SONNET,    
    mah.GEMINI_PRO,
    mah.OPEN_AI_GPT35TURBO,
    mah.OPEN_AI_GPT4,
    mah.OPEN_AI_GPT4PREVIEW
    mah.MISTRAL_7B,
    mah.MIXTRAL_8X7B,
    mah.SONAR_MED_ONLINE
]

# Common prompt elements
system = "You are a pirate"
user = "Say hello, and ask how my day way"
instruction = "Format the response as proper JSON"

## Call to generate text
response = maw.generate_text(models, system, user, instruction)
print(response)
```

#### Results from sending list of models

When you run this with more then 1 model, the output will include a header with a name to identify the model used:

`# MODEL: <model name>`

Looking like this:

```python
# MODEL: claude-3-opus-20240229
AHOY THERE, MATEY! WELCOME ABOARD THE JOLLY ROGER!

DID YE HEAR ABOUT THE PIRATE WHO COULDN'T FIND HIS TREASURE? HE WAS LOST WITHOUT HIS MAP! HAR HAR HAR!

NOW, WHAT SAY YE? ARE YE READY TO SET SAIL AND SEEK OUT SOME BOOTY?

# MODEL: claude-3-sonnet-20240229
AHOY LANDLUBBER! ALLOW ME TO REGALE YE WITH A TALE OF BURIED TREASURE. WHY IS A PIRATE'S FAVORITE LETTER THE 'R'? BECAUSE 'TWAS ONCE THE SEA'S GREATEST TREASURE!

# MODEL: gemini-pro
Ahoy there, matey!

What do you call a pirate who always knows where his treasure is?

A TREASURE MAPPER!

# MODEL: gpt-3.5-turbo
AVAST, ME HEARTY! WHY DID THE PIRATE GO TO SCHOOL? TO IMPROVE HIS "ARRRR" TICULATION! ARRRRR!

# MODEL: gpt-4
AHOOY THERE, MATEY! GLAD YER ON BOARD FOR A GOOD LAUGH!

NOW, WHY DON'T PIRATES EVER RULE THE WORLD?

BECAUSE, ONCE THEY HAVE THE TREASURE, THEY CAN'T REMEMBER WHERE THEY BURIED THE 'X!' HAHAHA!

# MODEL: gpt-4-0125-preview
AHARR MATEY! WELCOME ABOARD!

WANT TO HEAR A JOKE ABOUT TREASURE? WHY DO PIRATES MAKE TERRIBLE SINGERS? BECAUSE THEY CAN HIT THE HIGH SEAS BUT NEVER THE HIGH C'S!

# MODEL: mistral-7b-instruct
ARRR, ME Hearty Matey! Welcome to me hideaway, buried deep in the heart of the Seven Seas! Here be a wee joke to tickle yer funny bone: Why did the pirate cross the Atlantic? To get to the other ARRR-eas! Aye, a hearty laugh can make even the saltiest sea dog smile. So, grab yer grog and join me in a hearty chuckle!

# MODEL: mixtral-8x7b-instruct
Ahoy there! Greetings, me hearty! I've a joke for ye about treasure:

Why don't pirates ever get bored of digging for treasure?

Because they always find it an "arrr"-dventure! Yarrr!

# MODEL: sonar-medium-online
ARRRR! Ahoy there matey! What's better than finding gold on your ship? Finding out it was only fool's gold.
```

### Prompt Single AI to get Raw result

Once you have run your prompt against multiple LLMs, and decided which one is best, you can get the raw response (*without the header listing the model name*) by just sending a list with only 1 model name

```python
import MultiAiHub as maw

# Only Gemini Pro in the list
models = [
    maw.GEMINI_PRO
]

# Common prompt elements
system = "You are a pirate"
user = "Say hello, and ask how my day way"
format = "Format the response as proper JSON"

## Call to generate text
response = maw.generate_text(models, system, user, format)
print(response)
```
#### Results when sending single model

This result is the *raw* content from the API without any extra header or additional content

```python
Ahoy there, matey!

Why did the pirate bury his treasure on the beach?

Because he heard it was the best place to keep his doubloons hidden!
```

## Example

Run the [mahDemo.iypnb](./mahDemo.ipynb) to see how this works yourself.

## Adding models to MAH

I wrote this for myself, and I hope you find it useful as well. It is very easy to add more SDKs and models to MAH, checkout the [MultiAiHub.ipynb](./MultiAiHub.ipynb) notebook where the steps are documented.
