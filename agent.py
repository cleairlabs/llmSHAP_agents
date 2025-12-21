import gc, os
from llmSHAP.generation import Generation
from pydantic import BaseModel
import json
import inspect
import re

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, ValueFunction
from llmSHAP.llm.llm_interface import LLMInterface
from llmSHAP.types import Prompt, Optional


from openai import OpenAI
from dotenv import load_dotenv





class MultiplyArgs(BaseModel):
    num1: int
    num2: int

class WeatherArgs(BaseModel):
    city: str

class ToolCall(BaseModel):
    tool_name: str
    parameters: dict


class Agent(LLMInterface):
    def __init__(self,
                 model_name: str,
                 tool_registry: dict[str, dict],
                 temperature: float = 0.2,
                 max_tokens: int = 512,
                 seed: Optional[int] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("OPENAI_API_KEY is not set. Set it (e.g. in your .env).")
        self.client: OpenAI = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.tool_registry = tool_registry

    def is_local(self): return False
    def name(self): return self.model_name
    def cleanup(self): gc.collect()

    def generate(self, prompt: Prompt) -> str:
        messages = list(prompt)

        message = self._call_llm(messages).choices[0].message # type: ignore
        while message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ], # type: ignore
            })
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                params = json.loads(tool_call.function.arguments or "{}")
                result: str = f"Unknown tool: {name}"
                if name == "multiply_two_numbers":
                    arguments = MultiplyArgs.model_validate(params)
                    result = str(self._tool_multiply_two_numbers(arguments.num1, arguments.num2))
                elif name == "get_weather":
                    arguments = WeatherArgs.model_validate(params)
                    result = self._tool_get_weather(arguments.city)

                return result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            message = self._call_llm(messages).choices[0].message # type: ignore


        print(f"\n\nRETURN: {message.content or ''}")
        return message.content or ""
    

    def _extract_enabled_tool_names(self, messages_in: list[dict]) -> list[str]:
        enabled: list[str] = []
        pattern = re.compile(r"\[TOOL\s+([^\]]+)\]")
        for m in messages_in:
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            enabled.extend(pattern.findall(content))
        seen: set[str] = set()
        ordered: list[str] = []
        for name in enabled:
            if name in self.tool_registry and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _strip_tool_markers(self, messages_in: list[dict]) -> list[dict]:
        pattern = re.compile(r"\s*\[TOOL\s+[^\]]+\]\s*")
        cleaned: list[dict] = []
        for m in messages_in:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                cleaned.append({**m, "content": pattern.sub(" ", m["content"]).strip()})
            else:
                cleaned.append(m)
        return cleaned

    def _call_llm(self, messages_in: list[dict]) -> object:
        enabled_names = self._extract_enabled_tool_names(messages_in)
        tools = [self.tool_registry[name] for name in enabled_names]
        messages = self._strip_tool_markers(messages_in)

        kwargs: dict = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return self.client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    
    ### TOOLS ###
    def _tool_multiply_two_numbers(self, num1: int, num2: int) -> int:
        print(f"[CALLED] {inspect.currentframe().f_code.co_name}") # type: ignore
        return num1*num2
    
    def _tool_get_weather(self, city: str) -> str:
        print(f"[CALLED] {inspect.currentframe().f_code.co_name}") # type: ignore
        return "22 C and sunny"










#############################################################################
#############################################################################
#############################################################################
#############################################################################



class DiffValue(ValueFunction):
    def __init__(self) -> None: pass
    def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
        return abs(int(coalition_generation.output) - 275268)


def build_openai_tool_registry() -> dict[str, dict]:
    mult_schema = MultiplyArgs.model_json_schema()
    mult_schema["additionalProperties"] = False
    mult_schema["required"] = ["num1", "num2"]

    weather_schema = WeatherArgs.model_json_schema()
    weather_schema["additionalProperties"] = False
    weather_schema["required"] = ["city"]

    return {
        "multiply_two_numbers": {
            "type": "function",
            "function": {
                "name": "multiply_two_numbers",
                "parameters": mult_schema,
                "strict": True,
            },
        },
        "get_weather": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": weather_schema,
                "strict": True,
            },
        },
    }


data = {
    "user_query": "What is 339 times 812 ?",
    "tool_multiply": "[TOOL multiply_two_numbers]",
    "tool_weather": "[TOOL get_weather]",
}

handler = DataHandler(data, permanent_keys={"user_query"}) # type: ignore
prompt_codec = BasicPromptCodec(system="Answer the question with only a SINGLE NUMBER. Use the available tools.")

llm = Agent("gpt-4o-mini", tool_registry=build_openai_tool_registry())

result = ShapleyAttribution(model=llm,
                          data_handler=handler,
                          prompt_codec=prompt_codec,
                          use_cache=True,
                          num_threads=7,
                          value_function=DiffValue()).attribution()



print("\n\n### OUTPUT ###")
print(result.output)
print("\n\n### HEATMAP ###")
print(result.render(abs_values=True))