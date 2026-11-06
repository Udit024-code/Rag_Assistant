"""Groq chat calls for the agent, with model fallback.

Groq's free tier limits each model separately (8K tokens/minute for the
gpt-oss models), so when one is rate-limited the next model is tried
instead of failing the request.
"""
import json

from groq import Groq, RateLimitError

from config import get_secret

client = Groq(api_key=get_secret('GROQ_API_KEY'), max_retries=1)

# Planning and evidence checks are short structured calls; answers get the
# bigger model.
FAST_MODELS = ['openai/gpt-oss-20b', 'openai/gpt-oss-120b']
ANSWER_MODELS = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']


def _create(models, **kwargs):
    error = None
    for model in models:
        try:
            # gpt-oss reasons before answering; low effort keeps latency and
            # token use down without hurting these grounded tasks.
            return client.chat.completions.create(model=model, reasoning_effort='low', **kwargs)
        except RateLimitError as e:
            error = e
    raise error


def complete_json(messages, models=FAST_MODELS, max_tokens=1024):
    # temperature 0: the same question should get the same search plan.
    response = _create(models, messages=messages, response_format={'type': 'json_object'},
                       max_completion_tokens=max_tokens, temperature=0)
    return json.loads(response.choices[0].message.content)


def stream_text(messages, models=ANSWER_MODELS, max_tokens=2048):
    """Yield the answer as it is generated."""
    stream = _create(models, messages=messages, stream=True, max_completion_tokens=max_tokens)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
