from src.multi_agent_ai.tools.tools import web_search, scrape_url
from rich import print
from src.multi_agent_ai.pipelines.pipelines import run_search_pipeline

print(run_search_pipeline("DEscribe best things about TamilNadu"))
# print(scrape_url.invoke("https://www.linkedin.com/in/sandeep-roychoudhury-0034b2a7/"))