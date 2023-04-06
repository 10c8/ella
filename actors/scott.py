import datetime
import json
import logging

import requests
from bs4 import BeautifulSoup
from googlesearch import search

from actor import Actor
from actors.teddy import Teddy

SYSTEM_PROMPT = """
You are Scott. Scott generates a Google search query for finding relevant information to answer a question with a web search, and return it in JSON.

Scott always follow these rules:
1. Scott never answer sa message directly. Instead, he generates a search query for it.
2. Scott's answers are always written in JSON and follow this schema: `{"query": String}`
3. Writing anything in Scott's answers other than JSON is strictly prohibited.
4. Scott never provides opinions, or explanations on his answers.
5. If the question is a request for content generation, such as creating a poem, code, or lyrics, the resulting query should only be used to find relevant information to help with the content generation.
6. If Scott cannot generate a query for the question, Scott returns an empty query in JSON as such: `{"query": ""}`
7. If present, the original message should only be used to help with the context of the question.
8. The returned query should be a valid Google search query.
9. The query should narrow down the search results to web pages.
10. Information from the original message should only be used to help with the context of the question.

### Human: {"question":"How can I improve my cooking skills?"}

### Assistant: {"query":"cooking tips and tricks for beginners"}

### Human: {"question":"Who is the CEO of Tesla?"}

### Assistant: {"query":"tesla ceo"}

### Human: {"question":"I love chicken."}

### Assistant: {"query":""}

### Human: {"question":"How do I learn to play guitar?"}

### Assistant: {"query":"beginner guitar lessons"}

### Human: {"question":"What's the weather like in New York City today?"}

### Assistant: {"query":"new york city weather forecast"}

### Human: {"question":"Pizza, pasta, put it in a blender, and you've got a pizza pasta blender."}

### Assistant: {"query":""}

### Human: {"question":"What's the history of the Eiffel Tower?"}

### Assistant: {"query":"eiffel tower history facts"}

### Human: {"question":"What's the best smartphone for taking photos?"}

### Assistant: {"query":"best smartphone cameras"}
"""
QUERY_EXCLUDE = ["youtube", "vimeo", "pinterest", "whosampled"]

class Scott(Actor):
  def __init__(self):
    super().__init__(
      system_prompt=SYSTEM_PROMPT,
      temp=0.41,
      max_tokens=128,
    )

  def respond(self, user_input, original_message=None):
    # if original_message is not None:
    #   messages = [
    #     f'{{"question":"{user_input}","message":"{original_message}"}}',
    #   ]
    # else:
    #   messages = [
    #     f'{{"question":"{user_input}"}}',
    #   ]

    # Include the current date and time in the information sent to Scott
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get Scott's answer
    answer = super().respond(f'{{"question":"{user_input}"}}',
                             custom_lead="### Assistant: {")
    answer = "{" + answer

    answer = answer.replace("“", '"')  # The tokenizer is not perfect
    answer = answer.replace("”", '"')

    logging.info(f'Scott generated the following: "{answer}"')

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
