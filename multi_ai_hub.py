#!/usr/bin/env python
# coding: utf-8

# 

# ## MAH - Multi AI Hub
# 
# This project is designed to make it easy to send the same prompt to multiple LLMs which is useful for testing and comparison.
# 
# ### API Access Required
# 
# You must have access to the services (Currently Anthropic, Google, OpenAI, and Perplexity) in order to use them in this script.

# ### Working with API keys
# 
# Set the API keys as a system variables.
# 
# - [Setting an Environment Variable on Mac/Linux](https://phoenixnap.com/kb/set-environment-variable-mac)
# - [Setting an Environment Variable on Windows](https://phoenixnap.com/kb/windows-set-environment-variable)

# ## Tools to Get Environment Variables from OS
# 
# PIP Install:
# 
# `pip install python-dotenv`

# ## Adding other models
# 
# ### Check for Provider *Helper* Function
# 
# This is organized into API providers, there are helper functions for:
# - [**Anthropic**](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
# - [**Google**](https://ai.google.dev/)
# - [**Hugging Face**](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client)
# - [**OpenAI**](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
# - [**NVidia**](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
# - [**Perplexity**](https://docs.perplexity.ai/)
# 
# Create a new helper function if necessary, then skip to the bottom, and add your calls to the Action dictionary, where these are mapped (pretty simple)
# 
# Happy Model Comparing!
#   

# ## Setup Google GenAI
# 
# ### Import Google Generative GenerativeAI library and set API Key
# 
# PIP Install: 
# 
# `pip install -q google.generativeai`
# 
# You will need to set the Gemini API key as a system variable named: `GOOGLE_API_KEY`.
# 
# #### Then load the key from OS and set it 

# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as googleai

_ = load_dotenv(find_dotenv()) # read local .env file
apiKey = os.getenv('GOOGLE_API_KEY')

googleai.configure(api_key=apiKey,
               transport="rest",
    )


# ## Setup Gemini configuration
# 
# This is where you can configure temperature, safety settings, max tokens, etc

# In[ ]:


from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig

generation_config = GenerationConfig(
    temperature=0.1,
    max_output_tokens=8192,
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# ## Explore the Available Models
# 
# Learn which models are currently available
# 

# In[ ]:


# for m in googleai.list_models():
#     print(f"name: {m.name}")
#     print(f"description: {m.description}")
#     print(f"generation methods:{m.supported_generation_methods}\n")


# ### Filter models to ensure model we want is supported
# - `generateContent` is the value we are looking for

# In[ ]:


# for m in googleai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)


# ### Google AI Helper Function
# 
# - The `@retry` decorator helps you to retry the API call if it fails.

# In[ ]:


from google.api_core import retry

@retry.Retry()
def generate_text_google(prompt, model):
    model = googleai.GenerativeModel(model_name=model,
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    response = model.generate_content(prompt)
    return response.text


# ### Test **Google AI Helper** function

# In[ ]:


# print(generate_text_google("Thursday evenings are perfect for", "gemini-1.5-flash-latest"))


# ## Setup Open AI APIs
# 
# ```
# OpenAI's APIs offer developers the ability to integrate advanced artificial intelligence capabilities into their applications, enabling a wide range of tasks from text generation to complex problem-solving.
# ```
# Documentation: [https://beta.openai.com/docs/](https://beta.openai.com/docs/)
# 
# ### Obtaining API Keys:
# - **OpenAI Platform**: [https://platform.openai.com/](https://platform.openai.com/)
#   - After signing up or logging in, navigate to the API section to manage and obtain your API keys.
# - You will need to set the OpenAI API key as a system variable named: `OPENAI_API_KEY`.  
# 
# Note: do NOT check your API key into a public Github repo, or it will get revoked 
#   
#   
# 

# In[ ]:


import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')


# ### Open AI Helper Function
# 
# PIP Dependencies:
# 
# `pip install --upgrade openai`

# In[ ]:


from openai import OpenAI
client = OpenAI()

def generate_text_openai(pre, prompt, model):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": pre},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content


# ## Test **Open AI Helper** Function

# In[ ]:


#print(generate_text_openai("You are a pirate", "Thursday evenings are perfect for", "gpt-4o"))


# ## Setup Perplexity API
# 
# You will need a key set to `PERPLEXITY_API_KEY`

# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

YOUR_API_KEY = os.getenv('PERPLEXITY_API_KEY')


# ## Perplexity Helper function
# 
# No PIP dependency, you **must** have the **OpenAI SDK Installed**.

# In[ ]:


from openai import OpenAI

perplexityClient = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

def generate_text_perplexity(system, user, model):
    response = perplexityClient.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    )

    content = response.choices[0].message.content
    return content


# ## Test **Perplexity Helper** Function

# In[ ]:


#print(generate_text_perplexity("you are a pirate", "say hello and return the message in uppercase", "mistral-7b-instruct"))


# ## Setup Anthropic
# 
# Check the [docs](https://github.com/anthropics/anthropic-sdk-python), and get an [API Key](https://console.anthropic.com/dashboard)
# 
# ### Import SDK 
# 
# PIP Install:
# 
# `pip install anthropic`
# 

# In[ ]:


from anthropic import Anthropic

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def generate_text_anthropic(user, model="claude-3-opus-20240229"):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": user,
            }
        ],
        model=model,
        )
    content = response.content[0].text 
    return content


# ### Test the Anthropic API directly

# In[ ]:


#print(generate_text_anthropic("you are a pirate" + "say hello and return the message in uppercase", "claude-3-5-sonnet-20240620"))


# ## Setup Hugging Face
# 
# ### PIP Install Hugging Face Hub
# 
# `pip install --upgrade huggingface_hub`
# 
# ### Get API Key
# 
# Head to HuggingFace [Settings Page](https://huggingface.co/settings/tokens) and create an API token.
# 
# and set it as an environment variable named: `HUGGING_FACE_HUB_TOKEN`

# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

HF_API_KEY = os.getenv('HUGGING_FACE_HUB_TOKEN')


# ## Hugging Face Helper function using the InferenceClient

# ### Use the InferenceClient to check if a model is available

# In[ ]:


# bigscience/bloom | bigcode/starcoder

from huggingface_hub import InferenceClient
client = InferenceClient()
client.get_model_status("bigscience/bloom")


# ### Must Enable Models In Hugging Face to use them 
# 
# **Note** - to use HF models (which can be an URL to a private model, or a `model_id`) you will need to load that [model](https://huggingface.co/models) into your Hugging Face profile first.
# 
# Some models are available without enabling them. The first models includes `bloom` which is already enabling, and `gemma7b` which is one that requires to enable it first before using.

# In[ ]:


from huggingface_hub import InferenceClient, InferenceTimeoutError

def generate_text_huggingface(user, model=""):
    try:
        huggingface_client = InferenceClient(model=model, token=HF_API_KEY, timeout=60)
        response = huggingface_client.text_generation(user, max_new_tokens=1024)
    except InferenceTimeoutError:
        print("Inference timed out after 60s.")
    return response




# ### Test the Hugging Face Helper function directly

# In[ ]:


#print(generate_text_huggingface("you are a pirate tell me your favorite color", "meta-llama/Llama-2-7b"))


# ## Setup NVidia NGC
# 
# You must go to (NVidia Builder Portal)[https://build.nvidia.com/] to setup an API key to use their models.
# 
# For this function you **must** have the **Open AI SDK Installed** since there is no SDK for NVidia.

# In[ ]:


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')


# ### Helper function for using NVidia Inference APIs

# In[ ]:


nvidia_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

def generate_text_ngc(system, user, model):
  completion = nvidia_client.chat.completions.create(
    model=model,
    messages=[
      {"role":"system","content":system},
      {"role":"user","content":user},
      
    ],
    max_tokens=1024
  )

  content = completion.choices[0].message.content
  return content


# ### Test for the NVidia function

# 

# In[ ]:


#print(generate_text_ngc("you are a pirate", "you are a pirate tell me your favorite color", "meta/llama3-8b"))


# ## Add Actions to map to different models and AI providers

# 1. Define a function for each model you want to test
# 2. Create a constant to reference that model
# 3. Add both to the dictionary

# In[ ]:


# This is the common interface for all the models
# It takes the **system** message, **user** message and the **output style** instructions and calls
# the model specific function with those inputs (matching the API signature)
# Constants for the models - this name is arbitrary, should be unique
ANTHROPIC_OPUS = "claude-3-opus-20240229"
ANTHROPIC_SONNET = "claude-3-5-sonnet-20240620"
GEMINI_PRO = "gemini-1.0-pro-latest"
GEMINI_FLASH = "gemini-1.5-flash-latest"
HUGGINGFACE_BLOOM = "bigscience/bloom"
HUGGINGFACE_GEMMA7B = "google/gemma-7b"
HUGGINGFACE_LLAMA2_7B = "meta-llama/Llama-2-7b"
OPEN_AI_GPT35TURBO = "gpt-3.5-turbo"
OPEN_AI_GPT4 = "gpt-4"
OPEN_AI_GPT4O = "gpt-4o"
OPEN_AI_GPT4PREVIEW = "gpt-4-0125-preview"
MISTRAL_7B = "mistral-7b-instruct"
MIXTRAL_8X7B = "mixtral-8x7b-instruct"
NVIDIA_LLAMA3_8B = "meta/llama3-8b"
NVIDIA_LLAMA3_70B = "meta/llama3-70b"
SONAR_MED_ONLINE = "sonar-medium-online"

def action_anthropic_opus(system, user, output_style):
    response = generate_text_anthropic(system + user + output_style, ANTHROPIC_OPUS)
    return response

def action_anthropic_sonnet(system, user, output_style):
    response = generate_text_anthropic(system + user + output_style, ANTHROPIC_SONNET)
    return response

def action_gemini_pro(system, user, output_style,):
    response = generate_text_google(system + user + output_style, GEMINI_PRO)
    return response

def action_gemini_flash(system, user, output_style,):
    response = generate_text_google(system + user + output_style, GEMINI_FLASH)
    return response

def action_huggingface_bloom(system, user, output_style,):
    response = generate_text_huggingface(system + user + output_style, HUGGINGFACE_BLOOM)
    return response

def action_huggingface_gemma7b(system, user, output_style,):
    response = generate_text_huggingface(system + user + output_style, HUGGINGFACE_GEMMA7B)
    return response

def action_huggingface_llama2_7b(system, user, output_style,):
    response = generate_text_huggingface(system + user + output_style, HUGGINGFACE_LLAMA2_7B)
    return response

def action_openai_35turbo(system, user, output_style):
    response = generate_text_openai(system, user + output_style, OPEN_AI_GPT35TURBO)
    return response

def action_openai_gpt4(system, user, output_style):
    response = generate_text_openai(system, user + output_style, OPEN_AI_GPT4)
    return response

def action_openai_gpt4o(system, user, output_style):
    response = generate_text_openai(system, user + output_style, OPEN_AI_GPT4O)
    return response

def action_openai_gpt4_preview(system, user, output_style):
    response = generate_text_openai(system, user + output_style, OPEN_AI_GPT4PREVIEW)
    return response

def action_mistral_7b(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, MISTRAL_7B)
    return response

def action_mixtral_8x7b(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, MIXTRAL_8X7B)
    return response

def action_nvidia_llama3_8b(system, user, output_style):
    response = generate_text_ngc(system, user + output_style, NVIDIA_LLAMA3_8B)
    return response

def action_nvidia_llama3_70b(system, user, output_style):
    response = generate_text_ngc(system, user + output_style, NVIDIA_LLAMA3_70B)
    return response

def action_sonar_medium_online(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, SONAR_MED_ONLINE)
    return response

# Dictionary mapping models to their respective functions
action_dict = {
    ANTHROPIC_OPUS: action_anthropic_opus,
    ANTHROPIC_SONNET: action_anthropic_sonnet,
    GEMINI_PRO: action_gemini_pro,
    GEMINI_FLASH: action_gemini_flash,
    HUGGINGFACE_BLOOM: action_huggingface_bloom,
    HUGGINGFACE_GEMMA7B: action_huggingface_gemma7b,
    HUGGINGFACE_LLAMA2_7B: action_huggingface_llama2_7b,
    OPEN_AI_GPT35TURBO: action_openai_35turbo,
    OPEN_AI_GPT4: action_openai_gpt4,
    OPEN_AI_GPT4O: action_openai_gpt4o,
    OPEN_AI_GPT4PREVIEW: action_openai_gpt4_preview,
    MISTRAL_7B: action_mistral_7b,
    MIXTRAL_8X7B: action_mixtral_8x7b,
    NVIDIA_LLAMA3_8B: action_nvidia_llama3_8b,
    NVIDIA_LLAMA3_70B: action_nvidia_llama3_70b,
    SONAR_MED_ONLINE: action_sonar_medium_online
}

        


# ## Main Entry Point to call appropriate functions based which are requested in `models` list

# In[ ]:


def generate_text(models, system, user, output_style):
    """
    Generate text responses from multiple AIs based on **models** in list.

    If there is only 1 models in the list, the response will not include the model name.
    Otherwise, the response will include the model name as a header of the text generated from each model.

    Args:
        models (list): A list of model names indicating which ones to run.
        system (str): The prompt *system* information to define context.
        user (str): The prompt *user* information to describe the question to ask.
        output_style (str): The prompt desired *output_style* of the generated text.

    Returns:
        str: the generated text for all of the models in the input list
    """
    output = ""
    is_single_model = len(models) == 1

    for model in models:
        action = action_dict.get(model)
        if action:
            try:
                response = action(system=system, user=user, output_style=output_style)
                if not is_single_model:
                    output += "\n\n# MODEL: " + model + "\n"
                output += response
            except Exception as e:
                if not is_single_model:
                    output += "\n\n# MODEL: " + model + "\n"
                output += "Exception" + str(e)
        else:
            print("No action defined for model: ", model)

    return output


# ## Final Step
# 
# After making changes to this notebook, run the following on the command-line to create the python script to use:
# 
# ```
# jupyter nbconvert --to script .\multi_ai_hub.ipynb
# ```

# In[ ]:




