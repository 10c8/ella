import logging
import os

import openai
from dotenv import load_dotenv

from actors.ella import Ella

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(filename="ella.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Main code
if __name__ == "__main__":
  ella = Ella()

  while True:
    try:
      question = input("> ")
      logging.info(f'User asks: "{question}".')

      answer = ella.respond(question)

      full_answer = ""
      for message in answer:
        content = message["content"]
        content = content.replace("\n###", "")

        full_answer += content

      logging.info(f'Ella responds with: "{full_answer}".')
      print(full_answer)

    except KeyboardInterrupt:
      quit()
