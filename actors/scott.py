import datetime
import json
import logging

import requests
from bs4 import BeautifulSoup
from googlesearch import search

from actor import Actor
from actors.teddy import Teddy

SYSTEM_PROMPT = """
You generate a Google search query for finding relevant information to answer a question with a web search, and return it in JSON.

You always follow your rules:
1. You never answer a message directly. Instead, generate a search query for it.
2. Your answers are always written in JSON.
3. Your answers follow this JSON schema: `{"query":String}`
4. Writing anything in your answers other than JSON is strictly prohibited.
5. You never provide opinions, or explanations on your answers.
6. If the question is a request for content generation, such as creating a poem, code, or lyrics, the resulting query should only be used to find relevant information to help with the content generation.
7. If you cannot generate a query for the question, you return an empty query in JSON.
8. When relevant, include the current date and time in your query.
9. If present, the original message should only be used to help with the context of the question.
10. The returned query should be a valid Google search query.
11. The query should narrow down the search results to web pages.
12. Information from the original message should only be used to help with the context of the question.

Examples:
Q: How can I improve my cooking skills?
A: cooking tips and tricks for beginners

Q: Who is the CEO of Tesla?
A: tesla ceo

Q: How do I learn to play guitar?
A: beginner guitar lessons

Q: What's the weather like in New York City today?
A: new york city weather forecast

Q: What's the history of the Eiffel Tower?
A: eiffel tower history facts

Q: What's the best smartphone for taking photos?
A: best smartphone cameras
"""
QUERY_EXCLUDE = ["youtube", "vimeo", "pinterest", "whosampled"]

class Scott(Actor):
  def __init__(self):
    super().__init__(
      system_prompt=SYSTEM_PROMPT,
      temp=0.7,
      max_tokens=128,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["###"],
    )

  def respond(self, user_input, original_message=None):
    if original_message is not None:
      messages = [
        {
          "role": "user",
          "content": f'Question: {user_input}\nOriginal message: {original_message}',
        },
      ]
    else:
      messages = [
        {
          "role": "user",
          "content": f'Question: {user_input}',
        },
      ]

    # Include the current date and time in the information sent to Scott
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get Scott's answer
    answer = super().respond(messages,
                             custom_system_prompt=SYSTEM_PROMPT + f"\n\nCurrent date and time: {date}.")
    answer = answer.choices[0].message.content

    # Check if the answer is valid JSON and return it
    try:
      answer = json.loads(answer)
    except json.decoder.JSONDecodeError:
      answer = {
        "query": ""
      }

    if answer["query"] == "":
      # No search needed
      logging.info("Scott says a web search isn't necessary.")

      return {
        "searched": False,
        "query": "",
        "data": [],
      }
    else:
      # Perform a web search
      query = answer["query"]
      print(f' -- Searching the web for "{query}"...')
      logging.info(f'Scott starts to search the web for "{query}".')

      data = []
      results = search(query + " -" + " -".join(QUERY_EXCLUDE),
                       tld="com",
                       num=3,
                       stop=3,
                       pause=2)

      for url in results:
        logging.info(f'Scott started reading from "{url}".')

        try:
          # Extract the relevant information from the result
          page = requests.get(url).text
          soup = BeautifulSoup(page, "html.parser")

          title = soup.title.string
          paragraphs = soup.find_all("p")
          # spans = soup.find_all("span")

          page_content = "\n".join([p.text for p in paragraphs])
          # page_content += "\n".join([s.text for s in spans])

          if page_content == "":
            logging.info("Scott found nothing there.")
            continue

          if len(page_content) > 2000:
            page_content = f"{page_content[:2000]}..."

          # Ask Teddy to summarize the page
          teddy = Teddy()
          teddy_answer = teddy.respond(user_input, page_content)

          if teddy_answer["relevant"]:
            data.append({
              "title": title,
              "summary": teddy_answer["summary"],
              "url": url,
            })
        except Exception as e:
          logging.error(f'Scott failed: "{e}".')

      return {
        "searched": True,
        "query": query,
        "data": data,
      }
