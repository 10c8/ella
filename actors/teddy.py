import datetime
import json
import logging

from actor import Actor

SYSTEM_PROMPT = """
You receive the contents of a web page and the query that found it, and you generate a summary of the page, relevant to the query.

You always follow your rules:
1. You never answer a message directly. Instead, generate a summary for it.
2. Your answers are always written in JSON.
3. Your answers follow this JSON schema: `{"summary":String}`
4. Writing anything in your answers other than JSON is strictly prohibited.
5. You never provide opinions, or explanations on your answers.
6. If you cannot generate a summary for the page, you return an empty summary in JSON.
"""

class Teddy(Actor):
  def __init__(self):
    super().__init__(
      system_prompt=SYSTEM_PROMPT,
      temp=0.7,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["###"],
    )

  def respond(self, query, user_input):
    logging.info("Teddy starts generating a summary for the following:")
    logging.info(f'Query: "{query}"')
    logging.info(f'Content: "{user_input}"')

    messages = [
      {
        "role": "user",
        "content": f"Query: {query}\nContent: {user_input}",
      },
    ]

    # Include the current date and time in the information sent to Teddy
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    answer = super().respond(messages,
                             custom_system_prompt=SYSTEM_PROMPT + f"\n\nCurrent date and time: {date}.")
    answer = answer.choices[0].message.content

    try:
      answer = json.loads(answer)
    except json.decoder.JSONDecodeError:
      answer = {
        "summary": ""
      }

    if answer["summary"] == "":
      logging.info("Teddy couldn't do it.")

      return {
        "relevant": False,
        "summary": "",
      }
    else:
      logging.info(f'Teddy generated the following: "{answer["summary"]}"')

      return {
        "relevant": True,
        "summary": answer["summary"],
      }
