from griptape.structures import Agent
from griptape.tools import WebScraper, FileManager
from examples import utils


agent = Agent(
    prompt_driver=utils.prompt_driver(),
    embedding_driver=utils.embedding_driver(),
    tools=[
        WebScraper(),
        FileManager()
    ]
)

url = "https://www.aboutamazon.com/news/aws/aws-amazon-bedrock-general-availability-generative-ai-innovations"

agent.run(f"summarize {url} and store it to summary.txt")
