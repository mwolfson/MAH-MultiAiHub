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

# In[1]:


import os
from dotenv import load_dotenv, find_dotenv


# ## Setup Google GenAI
# 
# ### Import Google Generative GenerativeAI library and set API Key
# 
# PIP Install: 
# 
# `pip install -q google.generativeai`
# 
# You will need to set the Gemini API key as a system variable named: `GOOGLE_API_KEY`.

# In[2]:


import google.generativeai as googleai

_ = load_dotenv(find_dotenv()) # read local .env file
apiKey = os.getenv('GOOGLE_API_KEY')

googleai.configure(api_key=apiKey,
               transport="rest",
    )


# ## Explore the Available Models
# 
# Learn which models are currently available
# 

# In[3]:


# for m in googleai.list_models():
#     print(f"name: {m.name}")
#     print(f"description: {m.description}")
#     print(f"generation methods:{m.supported_generation_methods}\n")


# ### Filter models to ensure model we want is supported
# - `generateContent` is the value we are looking for

# In[4]:


# for m in googleai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)


# ### Google AI Helper Function
# 
# - The `@retry` decorator helps you to retry the API call if it fails.

# In[5]:


from google.api_core import retry
@retry.Retry()
def generate_text_google(prompt, model):
    model = googleai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text


# ### Test **Google AI Helper** function

# In[6]:


# print(generate_text_google("Thursday evenings are perfect for", "gemini-pro"))


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

# In[7]:


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

# In[8]:


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

# In[9]:


# print(generate_text_openai("You are a pirate", "Thursday evenings are perfect for", "gpt-3.5-turbo"))


# ## Setup Perplexity API
# 
# You will need a key set to `PERPLEXITY_API_KEY`

# In[10]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

YOUR_API_KEY = os.getenv('PERPLEXITY_API_KEY')


# ## Perplexity Helper function
# 
# No PIP dependency

# In[11]:


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

# In[17]:


# print(generate_text_perplexity("you are a pirate", "say hello and return the message in uppercase", "mistral-7b-instruct"))


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

# In[13]:


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

# In[14]:


# print(generate_text_anthropic("you are a pirate" + "say hello and return the message in uppercase", "claude-3-opus-20240229"))


# ## Add Actions to map to different models and AI providers

# 1. Define a function for each model you want to test
# 2. Create a constant to reference that model
# 3. Add both to the dictionary

# In[15]:


# Each model has a function that translates the standard input to match the model's expected input format
def action_anthropic_opus(system, user, format):
    response = generate_text_anthropic(system + user + format, "claude-3-opus-20240229")
    return response

def action_anthropic_sonnet(system, user, format):
    response = generate_text_anthropic(system + user + format, "claude-3-sonnet-20240229")
    return response

def action_gemini_pro(system, user, format,):
    response = generate_text_google(system + user + format, "gemini-pro")
    return response

def action_openai_35turbo(system, user, format):
    response = generate_text_openai(system, user + format, "gpt-3.5-turbo")
    return response

def action_openai_gpt4(system, user, format):
    response = generate_text_openai(system, user + format, "gpt-4")
    return response

def action_openai_gpt4_preview(system, user, format):
    response = generate_text_openai(system, user + format, "gpt-4-0125-preview")
    return response

def action_mistral_7b(system, user, format):
    response = generate_text_perplexity(system, user + format, "mistral-7b-instruct")
    return response

def action_mixtral_8x7b(system, user, format):
    response = generate_text_perplexity(system, user + format, "mixtral-8x7b-instruct")
    return response

def action_sonar_medium_online(system, user, format):
    response = generate_text_perplexity(system, user + format, "sonar-medium-online")
    return response

# Constants for the models
ANTHROPIC_OPUS = "claude-3-opus-20240229"
ANTHROPIC_SONNET = "claude-3-sonnet-20240229"
GEMINI_PRO = "gemini-pro"
OPEN_AI_GPT35TURBO = "gpt-3.5-turbo"
OPEN_AI_GPT4 = "gpt-4"
OPEN_AI_GPT4PREVIEW = "gpt-4-0125-preview"
MISTRAL_7B = "mistral-7b-instruct"
MIXTRAL_8X7B = "mixtral-8x7b-instruct"
SONAR_MED_ONLINE = "sonar-medium-online"

# Dictionary mapping models to their respective functions
action_dict = {
    ANTHROPIC_OPUS: action_anthropic_opus,
    ANTHROPIC_SONNET: action_anthropic_sonnet,
    GEMINI_PRO: action_gemini_pro,
    OPEN_AI_GPT35TURBO: action_openai_35turbo,
    OPEN_AI_GPT4: action_openai_gpt4,
    OPEN_AI_GPT4PREVIEW: action_openai_gpt4_preview,
    MISTRAL_7B: action_mistral_7b,
    MIXTRAL_8X7B: action_mixtral_8x7b,
    SONAR_MED_ONLINE: action_sonar_medium_online
}

        


# ## Main Entry Point to call appropriate functions based which are requested in `models` list

# In[16]:


def generate_text(models, system, user, format):
    """
    Generate text using the specified models.

    If there is only 1 model in the list, the response will not include the model name.
    Otherwise, the response will include the model name as a header of the text generated from each model.

    Args:
        models (list): A list of model names where to run the prompt.
        system (str): The prompt *system* information to define context.
        user (str): The prompt *user* information to describe the question to ask.
        format (str): The prompt desired *format* of the generated text.

    Returns:
        str: the generated text for all of the models in the input list
    """
    output = ""
    is_single_model = len(models) == 1

    for model in models:
        action = action_dict.get(model)
        if action:
            try:
                response = action(system=system, user=user, format=format)
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


# In[ ]:





# In[ ]:




