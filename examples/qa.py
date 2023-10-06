import io
import requests
from griptape.drivers import LocalVectorStoreDriver
from griptape.engines import VectorQueryEngine
from griptape.loaders import PdfLoader
from griptape.structures import Agent
from griptape.tools import VectorStoreClient
from griptape.utils import Chat
from examples import utils


namespace = "attention"

response = requests.get("https://arxiv.org/pdf/1706.03762.pdf")
query_engine = VectorQueryEngine(
    prompt_driver=utils.prompt_driver(),
    vector_store_driver=LocalVectorStoreDriver(
        embedding_driver=utils.embedding_driver()
    )
)

query_engine.upsert_text_artifacts(
    namespace=namespace,
    artifacts=PdfLoader().load(
        io.BytesIO(response.content)
    )
)

vector_store_client = VectorStoreClient(
    description="Contains information about the Attention Is All You Need paper.",
    query_engine=query_engine,
    namespace=namespace
)

agent = Agent(
    tool_memory=None,
    prompt_driver=utils.prompt_driver(),
    embedding_driver=utils.embedding_driver(),
    tools=[vector_store_client]
)

Chat(agent).start()
