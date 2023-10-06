from griptape.drivers import BasePromptDriver, OpenAiChatPromptDriver, OpenAiEmbeddingDriver, BaseEmbeddingDriver


def prompt_driver() -> BasePromptDriver:
    return OpenAiChatPromptDriver(
        model="gpt-4"
    )


def embedding_driver() -> BaseEmbeddingDriver:
    return OpenAiEmbeddingDriver()
