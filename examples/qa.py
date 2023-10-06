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
engine = VectorQueryEngine(
    vector_store_driver=LocalVectorStoreDriver(
        embedding_driver=utils.embedding_driver()
    )
)

engine.vector_store_driver.upsert_text_artifacts(
    {
        namespace: PdfLoader().load(
            io.BytesIO(response.content)
        )
    }
)

kb_client = VectorStoreClient(
    description="Contains information about the Attention Is All You Need paper.",
    query_engine=engine,
    namespace=namespace
)

agent = Agent(
    tool_memory=None,
    prompt_driver=utils.prompt_driver(),
    embedding_driver=utils.embedding_driver(),
    tools=[kb_client]
)

Chat(agent).start()
