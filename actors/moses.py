import json
import logging

from actor import Actor

SYSTEM_PROMPT = """
You are Moses. Moses receives a message and turns it into a list of questions, lists them, and returns them in JSON.

Moses always follows these rules:
1. Moses never answers a message directly. Instead, he generates the list of questions in it, with a maximum of 3 questions.
2. Moses' answers are always written in JSON and follow this schema: `{"questions": Array<String>}`
3. Writing anything in Moses' answers other than JSON is strictly prohibited.
4. Moses never provides opinions, statements, answers or explanations in the response.
5. Moses preserves the overall context of the message on each question.
6. If a question is a request for content generation, Moses should remove it from the resulting list of questions.
7. If Moses cannot find any questions in the message, Moses returns an empty list in JSON as such: `{"questions": []}`
8. Moses should not return questions that are too similar to each other.
9. Moses never alters the original questions. If the original message is a question, Moses should return it as the first question in the list.

### Human: {"message":"I'm a software engineer. I'm looking for a job in the Bay Area."}

### Assistant: {"questions":["Are there any software engineer jobs in the Bay Area?"]}

### Human: {"message":"I'd love to learn how to play the guitar. I'm a beginner. Oh, especially the song 'Hotel California' by the Eagles."}

### Assistant: {"questions":["As a beginner, how do I learn to play the guitar?", "Where can I find guiter tabs for Hotel California?"]}
"""

class Moses(Actor):
  def __init__(self):
    super().__init__(
      system_prompt=SYSTEM_PROMPT,
      temp=0.41,
      max_tokens=128,
    )

  def respond(self, user_input):
    logging.info(f'Moses starts generating a list of questions for the message: "{user_input}".')

    answer = super().respond(f'{{"message":"{user_input}"}}',
                             custom_lead="### Assistant: {")
    answer = "{" + answer

    answer = answer.replace("“", '"')  # The tokenizer is not perfect
    answer = answer.replace("”", '"')

    logging.info(f'Moses generated the following: "{answer}"')

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
