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

# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv


# ## Adding other models
# 
# ### Check for Provider *Helper* Function
# 
# This is organized into API providers, there are helper functions for:
# - [**Anthropic**](#anthropic_api) | [API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
# - [**AWS Bedrock**](#aws_api) | [API Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)
# - [**Azure**](#azure_api) | [API Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal)
# - [**Google**](#google_api) | [API Docs](https://ai.google.dev/)
# - [**OpenAI**](#openai_api) | [API Docs](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
# - [**Perplexity**](#pplx_api) | [API Docs](https://docs.perplexity.ai/)
# 
# Create a new helper function if necessary, then skip to the bottom, and add your calls to the Action dictionary, where these are mapped (pretty simple)
# 
# Happy Model Comparing!
#   

# ## <a name="google_api"></a>Setup Google GenAI
# 
# ### Import Google Generative GenerativeAI library and set API Key
# 
# PIP Install: 
# 
# `pip install -q google.generativeai`
# 
# You will need to set the Gemini API key as a system variable named: `GOOGLE_API_KEY`.

# In[ ]:


import google.generativeai as googleai

_ = load_dotenv(find_dotenv()) # read local .env file
apiKey = os.getenv('GOOGLE_API_KEY')

googleai.configure(api_key=apiKey,
               transport="rest",
    )


# ## Customize Gemini Settings
# 
# Use `generation_config` to specify various things (Ex. `temperature`, and `max_output_tokens`)
# 
# Use `safety_settings` to check the output to ensure it is free of harmful language.

# In[ ]:


from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig

generation_config = GenerationConfig(
    temperature=0.1,
    max_output_tokens=4096
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
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
    model = googleai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings
        )
    response = model.generate_content(prompt)
    return response.text


# ### Test **Google AI Helper** function

# In[ ]:


#print(generate_text_google("Thursday evenings are perfect for", "gemini-1.5-flash-latest"))


# ## <a name="openai_api"></a>Setup Open AI APIs
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


# ## <a name="pplx_api"></a>Setup Perplexity API
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


# ## <a name="anthropic_api"></a>Setup Anthropic
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


# print(generate_text_anthropic("you are a pirate" + "say hello and return the message in uppercase", "claude-3-opus-20240229"))


# ## <a name="azure_api"></a>Setup Azure
# 
# Check the [docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal), and get a project setup.
# 
# You will need an Project URI and an API_KEY and you should create environment variables for these, with the following names:
# 
# - AZURE_ENDPOINT_URL
# - AZURE_OPENAI_API_KEY
# 
# ### Import SDK 
# 
# There is no additional dependencies, because this uses the OpenAI SDK.

# In[ ]:


import os
from openai import AzureOpenAI

endpoint = os.getenv('AZURE_ENDPOINT_URL')
apiKey = os.getenv('AZURE_OPENAI_API_KEY')
      
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=apiKey,
    api_version="2024-05-01-preview",
)

def generate_text_azure(pre, prompt, model="gpt-4"):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": pre},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content


# ### Test the Azure Endpoint directly

# In[ ]:


# print(generate_text_azure("you are a pirate", "say hello and return the message in uppercase", "gpt-4"))


# ## <a name="aws_api"></a>Setup AWS Bedrock
# 
# Check the [docs](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html), and get a project setup. You will need to setup a project, and request access to the models you wish to use.
# 
# You will need 2 values with environment variables having the following names:
# 
# - AWS_ACCESS_KEY_ID,
# - AWS_SECRET_ACCESS_KEY
# 
# ### Import SDK 
# 
# `pip install boto3 requests`

# In[ ]:


import json
import boto3

# Fetch AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')  # Default to us-east-1 if not set

# # Ensure credentials are set
# if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
#     raise ValueError("AWS credentials not found in environment variables")

# Create a Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def generate_text_aws(pre, prompt, model="ai21.j2-mid-v1"):
    body = json.dumps({
            "prompt": pre + prompt,
            "maxTokens": 2048,
            "temperature": 0.1,
            "topP": 1,
            "stopSequences": [],
            "countPenalty": {"scale": 0},
            "presencePenalty": {"scale": 0},
            "frequencyPenalty": {"scale": 0}
        })
    
    response = bedrock_client.invoke_model(
            modelId='ai21.j2-mid-v1',
            body=body
        )
    
    response_body = json.loads(response['body'].read())
    return response_body['completions'][0]['data']['text']


# ### Test the AWS Endpoint directly

# In[ ]:


# print(generate_text_aws("you are a pirate", "say hello and return the message in uppercase", "ai21.j2-mid-v1"))


# ## Add Actions to map to different models and AI providers

# 1. Define a function for each model you want to test
# 2. Create a constant to reference that model
# 3. Add both to the dictionary

# In[ ]:


# Constants for the models - use the unique name of the model as defined in the SDK
ANTHROPIC_OPUS = "claude-3-opus-20240229"
ANTHROPIC_SONNET = "claude-3-5-sonnet-20240620"
AZURE_GPT4 = "gpt-4"
AWS_JURASSIC2_MID = "ai21.j2-mid-v1"
AWS_LLAMA2_70B = "meta.llama2-70b-chat-v1"
GEMINI_PRO = "gemini-pro"
GEMINI_FLASH = "gemini-1.5-flash-latest"
OPEN_AI_GPT35TURBO = "gpt-3.5-turbo"
OPEN_AI_GPT4 = "gpt-4"
OPEN_AI_GPT4O = "gpt-4o"
OPEN_AI_GPT4PREVIEW = "gpt-4-0125-preview"
PPLX_LLAMA3_8B = "llama-3-8b-instruct"
PPLX_LLAMA3_70B = "llama-3-70b-instruct"
PPLX_MISTRAL_7B = "mistral-7b-instruct"
PPLX_MIXTRAL_8X7B = "mixtral-8x7b-instruct"
SONAR_MED_ONLINE = "sonar-medium-online"

# This is the common interface for all the models
# It takes the **system** message, **user** message and the **output style** instructions and calls
# the model specific function with those inputs (matching the API signature)
def action_anthropic_opus(system, user, output_style):
    response = generate_text_anthropic(system + user + output_style, ANTHROPIC_OPUS)
    return response

def action_anthropic_sonnet(system, user, output_style):
    response = generate_text_anthropic(system + user + output_style, ANTHROPIC_SONNET)
    return response

def action_azure_gpt4(system, user, output_style):
    response = generate_text_azure(system, user + output_style, AZURE_GPT4)
    return response

def action_aws_jurassic2mid(system, user, output_style):
    response = generate_text_aws(system, user + output_style, AWS_JURASSIC2_MID)
    return response

def action_aws_llama270b(system, user, output_style):
    response = generate_text_aws(system, user + output_style, AWS_LLAMA2_70B)
    return response

def action_gemini_pro(system, user, output_style,):
    response = generate_text_google(system + user + output_style, GEMINI_PRO)
    return response

def action_gemini_flash(system, user, output_style,):
    response = generate_text_google(system + user + output_style, GEMINI_FLASH)
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

def action_pplxllama_8b(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, PPLX_LLAMA3_8B)
    return response

def action_pplxllama_70b(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, PPLX_LLAMA3_70B)
    return response

def action_pplxmistral_7b(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, PPLX_MISTRAL_7B)
    return response

def action_pplxmixtral_8x7b(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, PPLX_MIXTRAL_8X7B)
    return response

def action_sonar_medium_online(system, user, output_style):
    response = generate_text_perplexity(system, user + output_style, SONAR_MED_ONLINE)
    return response

# Dictionary mapping models to their respective functions will be used by the client
action_dict = {
    ANTHROPIC_OPUS: action_anthropic_opus,
    ANTHROPIC_SONNET: action_anthropic_sonnet,
    AZURE_GPT4: action_azure_gpt4,
    AWS_JURASSIC2_MID: action_aws_jurassic2mid,
    AWS_LLAMA2_70B: action_aws_llama270b,
    GEMINI_PRO: action_gemini_pro,
    GEMINI_FLASH: action_gemini_flash,
    OPEN_AI_GPT35TURBO: action_openai_35turbo,
    OPEN_AI_GPT4: action_openai_gpt4,
    OPEN_AI_GPT4O: action_openai_gpt4o,
    OPEN_AI_GPT4PREVIEW: action_openai_gpt4_preview,
    PPLX_LLAMA3_8B: action_pplxllama_8b,
    PPLX_LLAMA3_70B: action_pplxllama_70b,
    PPLX_MISTRAL_7B: action_pplxmistral_7b,
    PPLX_MIXTRAL_8X7B: action_pplxmixtral_8x7b,
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


# ## Test Method
# 
# This method will run the MAH with all the models you want to test...this is convenient to check if all the inference calls work as expected.

# In[ ]:


# models = [    
#     AWS_JURASSIC2_MID,
#     AWS_LLAMA2_70B,
#     AZURE_GPT4
# ]

# system = "You are a pirate"
# user = "Make a greeting and tell me a joke about treasure"
# output_style = "Output the response in all capital letters"

# response = generate_text(models, system, user, output_style)

# # Output the response to the console
# print(response)


# ## Final Step
# 
# After making changes to this notebook, run the following on the command-line to create the python script to use:
# 
# ```
# jupyter nbconvert --to script ./multi_ai_hub.ipynb
# ```

# In[ ]:





# In[ ]:




