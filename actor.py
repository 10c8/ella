import openai


class Actor(object):
  """An agent that interacts with the OpenAI Chat Completions API."""

  def __init__(self,
               system_prompt,
               temp=0.7,
               max_tokens=256,
               top_p=1,
               frequency_penalty=0,
               presence_penalty=0,
               stop=["\n"]):
    self.system_prompt = system_prompt
    self.temp = temp
    self.max_tokens = max_tokens
    self.top_p = top_p
    self.frequency_penalty = frequency_penalty
    self.presence_penalty = presence_penalty
    self.stop = stop

  def respond(self, user_input, custom_system_prompt=None):
    """Generate a response to a user input.

    Args:
      user_input: A string containing the user's input.

    Returns:
      A string containing the AI's response.
    """

    messages = [
      {
        "role": "system",
        "content": custom_system_prompt or self.system_prompt,
      },
    ]

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages + user_input,
      temperature=self.temp,
      max_tokens=self.max_tokens,
      top_p=self.top_p,
      frequency_penalty=self.frequency_penalty,
      presence_penalty=self.presence_penalty,
      stop=self.stop,
    )

    return response
