from functools import partial
from models.gpt import gpt_completion_fn, gpt_nll_fn
from models.gpt import tokenize_fn as gpt_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn

from models.mistral import mistral_completion_fn, mistral_nll_fn
from models.mistral import tokenize_fn as mistral_tokenize_fn

from models.mistral_api import mistral_api_completion_fn, mistral_api_nll_fn
from models.mistral_api import tokenize_fn as mistral_api_tokenize_fn


# Required: Text completion function for each model
# -----------------------------------------------
completion_fns = {
    # GPT Models
    'gpt-4o-mini': partial(gpt_completion_fn, model='gpt-4o-mini'),
    'text-davinci-003': partial(gpt_completion_fn, model='text-davinci-003'),
    'gpt-4': partial(gpt_completion_fn, model='gpt-4'),
    'gpt-4-1106-preview':partial(gpt_completion_fn, model='gpt-4-1106-preview'),
    'gpt-3.5-turbo-instruct': partial(gpt_completion_fn, model='gpt-3.5-turbo-instruct'),
    
    # Mistral Models
    'mistral': partial(mistral_completion_fn, model='mistral'),
    'mistral-api-tiny': partial(mistral_api_completion_fn, model='mistral-tiny'),
    'mistral-api-small': partial(mistral_api_completion_fn, model='mistral-small'),
    'mistral-api-medium': partial(mistral_api_completion_fn, model='mistral-medium'),
    
    # Llama 2 Models
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-13b': partial(llama_completion_fn, model='13b'),
    'llama-70b': partial(llama_completion_fn, model='70b'),
    'llama-7b-chat': partial(llama_completion_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_completion_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_completion_fn, model='70b-chat'),

    # Llama 3 Models (New)
    'llama-3.1-8b': partial(llama_completion_fn, model='llama-3.1-8b'),
    'llama-3.2-3b': partial(llama_completion_fn, model='llama-3.2-3b'),
}

# Optional: NLL/D functions for each model
# -----------------------------------------------
nll_fns = {
    # GPT Models
    'gpt-4o-mini': partial(gpt_nll_fn, model='gpt-4o-mini'),
    'text-davinci-003': partial(gpt_nll_fn, model='text-davinci-003'),
    
    # Mistral Models
    'mistral': partial(mistral_nll_fn, model='mistral'),
    'mistral-api-tiny': partial(mistral_api_nll_fn, model='mistral-tiny'),
    'mistral-api-small': partial(mistral_api_nll_fn, model='mistral-small'),
    'mistral-api-medium': partial(mistral_api_nll_fn, model='mistral-medium'),
    
    # Llama 2 Models
    'llama-7b': partial(llama_nll_fn, model='7b'),
    'llama-13b': partial(llama_nll_fn, model='13b'),
    'llama-70b': partial(llama_nll_fn, model='70b'),
    'llama-7b-chat': partial(llama_nll_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_nll_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_nll_fn, model='70b-chat'),

    # Llama 3 Models (New)
    'llama-3.1-8b': partial(llama_nll_fn, model='llama-3.1-8b'),
    'llama-3.2-3b': partial(llama_nll_fn, model='llama-3.2-3b'),
}

# Optional: Tokenization function for each model
# -----------------------------------------------
tokenization_fns = {
    # GPT Models
    'gpt-4o-mini': partial(gpt_tokenize_fn, model='gpt-4o-mini'),
    'text-davinci-003': partial(gpt_tokenize_fn, model='text-davinci-003'),
    'gpt-3.5-turbo-instruct': partial(gpt_tokenize_fn, model='gpt-3.5-turbo-instruct'),
    
    # Mistral Models
    'mistral': partial(mistral_tokenize_fn, model='mistral'),
    'mistral-api-tiny': partial(mistral_api_tokenize_fn, model='mistral-tiny'),
    'mistral-api-small': partial(mistral_api_tokenize_fn, model='mistral-small'),
    'mistral-api-medium': partial(mistral_api_tokenize_fn, model='mistral-medium'),
    
    # Llama 2 Models
    'llama-7b': partial(llama_tokenize_fn, model='7b'),
    'llama-13b': partial(llama_tokenize_fn, model='13b'),
    'llama-70b': partial(llama_tokenize_fn, model='70b'),
    'llama-7b-chat': partial(llama_tokenize_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_tokenize_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_tokenize_fn, model='70b-chat'),

    # Llama 3 Models (New)
    'llama-3.1-8b': partial(llama_tokenize_fn, model='llama-3.1-8b'),
    'llama-3.2-3b': partial(llama_tokenize_fn, model='llama-3.2-3b'),
}

# Optional: Context lengths for each model
# -----------------------------------------------
context_lengths = {
    # GPT Models
    'gpt-4o-mini': 128000,
    'text-davinci-003': 4097,
    'gpt-3.5-turbo-instruct': 4097,
    
    # Mistral Models
    'mistral-api-tiny': 4097,
    'mistral-api-small': 4097,
    'mistral-api-medium': 4097,
    'mistral': 4096,
    
    # Llama 2 Models
    'llama-7b': 4096,
    'llama-13b': 4096,
    'llama-70b': 4096,
    'llama-7b-chat': 4096,
    'llama-13b-chat': 4096,

    # Llama 3 Models (New)
    # Lưu ý: Sửa Key ở đây cho khớp với Key trong completion_fns
    'llama-3.1-8b': 128000,
    'llama-3.2-3b': 128000,
}