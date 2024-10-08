- [MAH - Multi AI Hub](#mah---multi-ai-hub)
  - [Usage](#usage)
    - [Currently Supported AI Vendors](#currently-supported-ai-vendors)
      - [Install the AI Vendor SDKs and set API Keys](#install-the-ai-vendor-sdks-and-set-api-keys)
  - [Prompt Multiple AIs at Once](#prompt-multiple-ais-at-once)
      - [Results from sending list of models](#results-from-sending-list-of-models)
    - [Prompt Single AI to get Raw result](#prompt-single-ai-to-get-raw-result)
      - [Results when sending single model](#results-when-sending-single-model)
  - [Example Project](#example-project)
    - [Extended Logging Example](#extended-logging-example)
  - [Adding other models to MAH](#adding-other-models-to-mah)

# MAH - Multi AI Hub

**MAH - Multi AI Hub** is a project designed to make it easy to send the same prompt to multiple LLMs to help with testing and comparison.

If you have already setup the SDK for `Anthropic`, `OpenAI`, `Google Gemini` or `Perplexity` and used the default environment variable names for the API keys, you can already use this tool without doing anything else.

## Usage

Copy the `multi_ai_hub.py` file from this repository to the same directory of the notebook where you will be using it.

### Currently Supported AI Vendors
- **Anthropic** | [API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- **AWS Bedrock** | [API Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)
- **Azure** | [API Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal)
- **Google** | [API Docs](https://ai.google.dev/)
- **OpenAI** | [API Docs](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
- **Perplexity** | [API Docs](https://docs.perplexity.ai/)

#### Install the AI Vendor SDKs and set API Keys

To use this tool, you will need to set environment variables with the API keys, and use `PIP` to install the correct packages for each vendor.

Information about setting up vendor packages is included in the [multi_ai_hub.ipynb](./multi_ai_hub.ipynb) notebook. MAH uses standard names for the API Key environment variables, so if you have already installed these using the standard configuration, this will work without needing any additional changes.

---

## Prompt Multiple AIs at Once

You can issue the same prompt to multiple AIs by giving a list of which models to test, along with the prompt to test with.

```python
# Import the code
import multi_ai_hub as mah

# Possible Models to use:
# ANTHROPIC_OPUS = "claude-3-opus-20240229"
# ANTHROPIC_SONNET = "claude-3-5-sonnet-20240620"
# AZURE_GPT4 = "gpt-4"
# AWS_JURASSIC2_MID = "ai21.j2-mid-v1"
# AWS_LLAMA2_70B = "meta.llama2-70b-chat-v1"
# GEMINI_PRO = "gemini-pro"
# GEMINI_FLASH = "gemini-1.5-flash-latest"
# OPEN_AI_GPT35TURBO = "gpt-3.5-turbo"
# OPEN_AI_GPT4 = "gpt-4"
# OPEN_AI_GPT4O = "gpt-4o"
# OPEN_AI_GPT4PREVIEW = "gpt-4-0125-preview"
# PPLX_LLAMA3_8B = "llama-3-8b-instruct"
# PPLX_LLAMA3_70B = "llama-3-70b-instruct"
# PPLX_MISTRAL_7B = "mistral-7b-instruct"
# PPLX_MIXTRAL_8X7B = "mixtral-8x7b-instruct"
# SONAR_MED_ONLINE = "sonar-medium-online"

# List each model to test your prompt with
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
output_style = "Format the response as proper JSON"

## Call to generate text
response = mah.generate_text(models, system, user, output_style)
print(response)
```

#### Results from sending list of models

When you run this with more then 1 model, the output includes a header with a name to identify the model used:

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

--- 

### Prompt Single AI to get Raw result

Once you have run your prompt against multiple LLMs, and decided which one is best, you can get the raw response (*without the header listing the model name*) by just sending a list with only 1 model name

```python
import multi_ai_hub as mah

# Only Gemini Pro in the list
models = [
    mah.GEMINI_PRO
]

# Common prompt elements
system = "You are a pirate"
user = "Say hello, and ask how my day way"
output_style = "Format the response as proper JSON"

## Call to generate text
response = mah.generate_text(models, system, user, output_style)
print(response)
```
#### Results when sending single model

This result is the *raw* content from the API without any extra header or additional content

```python
Ahoy there, matey!

Why did the pirate bury his treasure on the beach?

Because he heard it was the best place to keep his doubloons hidden!
```

## Example Project

Run the [mah_demo.ipynb](./mah_demo.ipynb) to see how this works yourself.

### Extended Logging Example

Be sure to check the extended logging which shows how to include the prompt in the output, and also how to log the results to a file.

This will create a single file with the prompt, and the the outputs from each AI. Then it displays that output as HTML and saves it to the filesystem 

Example output file: [pirate_20240305_121024.md](./pirate_20240305_121024.md)

## Adding other models to MAH

I wrote this for myself, and I hope you find it useful as well. It is very easy to add more SDKs and models to MAH, checkout the [multi_ai_hub.ipynb](./multi_ai_hub.ipynb) notebook where the steps are documented.
