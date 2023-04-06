import logging

import requests


class Actor(object):
  """An agent that interacts with the OpenAI Chat Completions API."""

  def __init__(self,
               system_prompt,
               max_tokens=256,
               temp=0.7,
               top_p=0.1,
               top_k=40,
               repeat_penalty=1.18,
               stop=["\n###"]):
    self.system_prompt = system_prompt
    self.max_tokens = max_tokens
    self.temp = temp
    self.top_p = top_p
    self.top_k = top_k
    self.repeat_penalty = repeat_penalty
    self.stop = stop

  def respond(self, user_input, custom_system_prompt=None, custom_lead=None):
    """Generate a response to a user input.

    Args:
      user_input: A string containing the user's input.

    Returns:
      A string containing the AI's response.
    """

    messages = [
      custom_system_prompt or self.system_prompt,
      f"### Human: {user_input}",
      "### Assistant:" if custom_lead is None else custom_lead,
    ]

    logging.info("Sending the following prompt to the API:")
    logging.info(messages[1:])

    prompt = "\n\n".join(messages)

    params = {
      "prompt": prompt,
      "max_length": self.max_tokens,
      "temperature": self.temp,
      "top_p": self.top_p,
      "top_k": self.top_k,
      "rep_pen": self.repeat_penalty,
      "stopping_strings": self.stop,
    }

    req = requests.post(
      "http://127.0.0.1:5000/api/v1/generate",
      json=params
    )

    response = req.json()
    response = response["results"][0]["text"]

    # HACK: Account for the fact that the API doesn't always stop at the
    # stopping string
    for stop in self.stop:
      if stop in response:
        response = response.split(stop)[0]

    response = response.strip()

    return response
