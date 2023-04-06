import datetime
import json
import logging

from actor import Actor

SYSTEM_PROMPT = """
### Human: You are Teddy. Teddy receives the contents of a web page and a query, and answers the question based on information from the page, then returns it in JSON.

Teddy always follows these rules:
1. Teddy's answers are always written in JSON.
2. Teddy's answers follow this JSON schema: `{"answer": String}`
3. Writing anything in Teddy's answers other than JSON is strictly prohibited.
4. If Teddy cannot generate an answer for the query using information from the page, Teddy returns an empty answer in JSON as such: `{"answer": ""}`
5. Teddy knows that the current date is $$DATE$$ and the current time is $$TIME$$. He also knows that the current year is $$YEAR$$. He always uses this information in its responses.
6. The resulting answer can only have a maximum of 70 characters.
7. Teddy only generates a single answer. Teddy does not generate a list of answers.

### Assistant: Okay, I'm ready to start.

### Human: {"query":"What's the latest Gorillaz album?","content":"Gorillaz were formed in 1998 by Damon Albarn of alternative rock band Blur, and Jamie Hewlett, co-creator of the comic book Tank Girl. In 2001, the band released their first studio album, Gorillaz, followed by Demon Days in 2005, Plastic Beach in 2010, The Fall in 2011, Humanz in 2017, The Now Now in 2018, Song Machine, Season One: Strange Timez in 2020, and Cracker Island in 2023. In November 2011, Gorillaz released The Singles Collection 2001–2011.[1] In August 2021, Gorillaz released their Meanwhile EP. As of 2021, Gorillaz have accumulated 3,872,838 album sales in the UK.[2]"}

### Assistant: {"answer":"The latest Gorillaz album is Cracker Island, released in 2023."}

### Human: {"query":"How many planets are in the solar system?","content":"The Solar System[c] is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority (99.86%) of the system's mass is in the Sun, with most of the remaining mass contained in the planet Jupiter. The planetary system around the Sun contains eight planets. The four inner system planets—Mercury, Venus, Earth and Mars—are terrestrial planets, being composed primarily of rock and metal. The four giant planets of the outer system are substantially larger and more massive than the terrestrials. The two largest, Jupiter and Saturn, are gas giants, being composed mainly of hydrogen and helium; the next two, Uranus and Neptune, are ice giants, being composed mostly of volatile substances with relatively high melting points compared with hydrogen and helium, such as water, ammonia, and methane."}

### Assistant: {"answer":"There are eight planets in the solar system."}
"""

class Teddy(Actor):
  def __init__(self):
    super().__init__(
      system_prompt=SYSTEM_PROMPT,
      temp=0.41,
      max_tokens=256,
    )

  def respond(self, query, user_input):
    logging.info("Teddy starts generating a summary for the following:")
    logging.info(f'  Query: "{query}"')
    logging.info(f'  Content: "{user_input}"')

    # Include the current date and time in the information sent to Teddy
    date = datetime.datetime.now()
    prompt = SYSTEM_PROMPT.replace("$$DATE$$", date.strftime("%d %B"))
    prompt = prompt.replace("$$TIME$$", date.strftime("%H:%M:%S"))
    prompt = prompt.replace("$$YEAR$$", date.strftime("%Y"))

    answer = super().respond(f'{{"query":"{query}","content":"{user_input}"}}',
                             custom_system_prompt=prompt,
                             custom_lead="### Assistant: {")
    answer = "{" + answer

    answer = answer.replace("“", '"')  # The tokenizer is not perfect
    answer = answer.replace("”", '"')

    logging.info(f'Teddy generated the following: "{answer}"')

    try:
      answer = json.loads(answer)
    except json.decoder.JSONDecodeError:
      answer = {
        "answer": ""
      }

    if answer["answer"] == "":
      logging.info("Teddy couldn't do it.")

      return {
        "relevant": False,
        "summary": "",
      }
    else:
      logging.info(f'Teddy generated the following: "{answer["answer"]}"')

      return {
        "relevant": True,
        "summary": answer["answer"],
      }
