import json
import logging

from actor import Actor

SYSTEM_PROMPT = """
You receive a message and find different questions in it, list them, and return them in JSON.

You always follow your rules:
1. You never answer a message directly. Instead, generate the list of questions in it.
2. Your answers are always written in JSON.
3. Your answers follow this JSON schema: `{"questions":Array<String>}`
4. Writing anything in your answers other than JSON is strictly prohibited.
5. You never provide opinions, or explanations on your answers.
6. Make sure the overall context of the message is preserved on each question.
7. If you cannot find any questions in the message, you return an empty list in JSON.
"""

class Moses(Actor):
  def __init__(self):
    super().__init__(
      system_prompt=SYSTEM_PROMPT,
      temp=0.41,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["###"],
    )

  def respond(self, user_input):
    logging.info(f'Moses starts generating a list of questions for the message: "{user_input}".')

    messages = [
      {
        "role": "user",
        "content": user_input,
      },
    ]

    answer = super().respond(messages)
    answer = answer.choices[0].message.content

    try:
      answer = json.loads(answer)
    except json.decoder.JSONDecodeError:
      answer = {
        "questions": []
      }

    if answer["questions"] == []:
      logging.info("Moses couldn't do it.")

      return {
        "questions": []
      }

    logging.info(f"Moses found the following questions: {answer['questions']}.")

    return answer
