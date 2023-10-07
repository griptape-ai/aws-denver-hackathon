from griptape.chunkers import TextChunker
from griptape.drivers import LocalVectorStoreDriver
from griptape.engines import VectorQueryEngine
from griptape.loaders import WebLoader
from griptape.structures import Agent
from griptape.tasks import TextQueryTask
import utils


namespace = "bedrock"
url = "https://www.aboutamazon.com/news/aws/aws-amazon-bedrock-general-availability-generative-ai-innovations"

artifacts = WebLoader(
    chunker=TextChunker(
        max_tokens=200
    )
).load(url)

query_engine = VectorQueryEngine(
    prompt_driver=utils.prompt_driver(),
    vector_store_driver=LocalVectorStoreDriver(
        embedding_driver=utils.embedding_driver()
    )
)

query_engine.upsert_text_artifacts(
    namespace=namespace,
    artifacts=artifacts
)

agent = Agent(
    prompt_driver=utils.prompt_driver(),
    embedding_driver=utils.embedding_driver()
)

agent.add_task(
    TextQueryTask(
        query_engine=query_engine
    )
)

question = "What models does Bedrock support?"

print(
    agent.run("What models does Bedrock support?").output.to_text()
)
