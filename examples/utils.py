from griptape.drivers import (
    AmazonBedrockPromptDriver,
    BasePromptDriver,
    BedrockJurassicPromptModelDriver,
    BedrockTitanPromptModelDriver,
    OpenAiChatPromptDriver,
    OpenAiEmbeddingDriver,
    BaseEmbeddingDriver,
    BedrockTitanEmbeddingDriver,
    BedrockClaudePromptModelDriver,
)


def prompt_driver(driver="BEDROCK_CLAUDE") -> BasePromptDriver:
    if driver == "OPENAI":
        return OpenAiChatPromptDriver(model="gpt-4")
    elif driver == "BEDROCK_CLAUDE":
        return AmazonBedrockPromptDriver(
            model="anthropic.claude-v2",
            prompt_model_driver=BedrockClaudePromptModelDriver(),
        )
    elif driver == "BEDROCK_TITAN":
        return AmazonBedrockPromptDriver(
            model="amazon.titan-text-express-v1",
            prompt_model_driver=BedrockTitanPromptModelDriver(),
        )
    elif driver == "BEDROCK_JURASSIC":
        return AmazonBedrockPromptDriver(
            model="ai21.j2-ultra-v1",
            prompt_model_driver=BedrockJurassicPromptModelDriver(),
        )
    else:
        raise ValueError(f"unknown driver {driver}")


def embedding_driver(driver="BEDROCK_TITAN") -> BaseEmbeddingDriver:
    if driver == "OPENAI":
        return OpenAiEmbeddingDriver()
    elif driver == "BEDROCK_TITAN":
        return BedrockTitanEmbeddingDriver()
    else:
        raise ValueError(f"unknown driver {driver}")
