from __future__ import annotations

from dotenv import load_dotenv

from llmSHAP.generation import Generation
load_dotenv()

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, EmbeddingCosineSimilarity, ValueFunction
from llmSHAP.llm.langchain import LangChainInterface

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


def llmSHAP(data):
    class MathValue(ValueFunction):
        def __init__(self) -> None:
            pass
        def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
            return abs(int(coalition_generation.output) - 275268)
        
    handler = DataHandler(data, permanent_keys={"question"})
    print("STRING: ", handler.to_string())
    prompt_codec = BasicPromptCodec()
    shap = ShapleyAttribution(
        model=LangChainInterface(chat_model=build_agent(), tool_factory=build_agent),
        data_handler=handler,
        prompt_codec=prompt_codec,
        use_cache=True,
        num_threads=25,
        value_function=MathValue(),
    )
    return shap.attribution()




#################################
@tool
def multiply_two_numbers(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Retreive the current weather in a given city."""
    return "It is currently sunny and 25 C."

def build_agent(tools = None):
    return create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
        system_prompt="You are a calculator. Answer only with a single number. Only use a tool if it helps you. Otherwise you answer from your own knowledge. Do not use decimals or dots.",
    )

data = {
    "question": "What is 339 times 812 ?",
    "trap1": "the tallest building in the world is Stockolm.",
    "trap2": "the tallest building in the world is Paris.",
    "tool-multiply": multiply_two_numbers,
    "tool-weather": get_weather,
}



if __name__ == "__main__":
    result = llmSHAP(data)

    print("\n\n### OUTPUT ###")
    print(result.output)
    print("\n\n### ATTRIBUTION ###")
    print(result.attribution)
    print("\n\n### HEATMAP ###")
    print(result.render(abs_values=True, render_labels=True))