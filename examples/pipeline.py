from griptape.structures import Pipeline
from griptape.tasks import ToolkitTask, PromptTask
from griptape.tools import WebScraper
from examples import utils

pipeline = Pipeline(
    prompt_driver=utils.prompt_driver(),
    embedding_driver=utils.embedding_driver()
)

pipeline.add_tasks(
    ToolkitTask(
        input_template="Summarize this website: {{ args[0] }}",
        tools=[
            WebScraper()
        ]
    ),
    PromptTask(
        input_template="Summary: {{ parent_output }}\n\n"
                       "Based on the summary, answer the following question: {{ args[1] }}"
    ),
    PromptTask(
        input_template="Turn the following text into a Twitter thread: {{ parent_output }}"
    )
)

print(
    pipeline.run(
        "https://www.aboutamazon.com/news/aws/aws-amazon-bedrock-general-availability-generative-ai-innovations",
        "what is Bedrock?"
    ).output.to_text()
)
