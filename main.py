import logging

from dotenv import load_dotenv

from actors.ella import Ella

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(filename="ella.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# Main code
if __name__ == "__main__":
  ella = Ella()

  while True:
    try:
      question = input("> ")
      logging.info(f'User asks: "{question}".')

      answer = ella.respond(question)

      logging.info(f'Ella responds with: "{answer}".')
      print(answer)
    except KeyboardInterrupt:
      quit()
