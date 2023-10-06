from griptape.engines import PromptSummaryEngine
from griptape.structures import Agent
from griptape.tasks import TextSummaryTask

from examples import utils

text_to_summarize = """Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon with a single API, along with a broad set of capabilities you need to build generative AI applications, simplifying development while maintaining privacy and security. With the comprehensive capabilities of Amazon Bedrock, you can easily experiment with a variety of top FMs, privately customize them with your data using techniques such as fine-tuning and retrieval augmented generation (RAG), and create managed agents that execute complex business tasks—from booking travel and processing insurance claims to creating ad campaigns and managing inventory—all without writing any code. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with.
"""

agent = Agent(
    prompt_driver=utils.prompt_driver(),
    embedding_driver=utils.embedding_driver()
)

agent.add_task(
    TextSummaryTask(
        summary_engine=PromptSummaryEngine(
            prompt_driver=utils.prompt_driver()
        )
    )
)

print(
    agent.run(text_to_summarize).output.to_text()
)
