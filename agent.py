import gc, os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, ValueFunction
from llmSHAP.llm.llm_interface import LLMInterface
from llmSHAP.generation import Generation
from llmSHAP.types import Prompt, Optional
from tools import ToolRunner, build_openai_tool_registry



class Agent(LLMInterface):
    def __init__(
        self,
        model_name: str,
        tool_registry: dict[str, dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        tool_runner: ToolRunner | None = None,
    ):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it (e.g. in your .env).")

        self.client: OpenAI = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.tool_registry = tool_registry
        self.tool_runner = tool_runner or ToolRunner()

    def is_local(self): return False
    def name(self): return self.model_name
    def cleanup(self): gc.collect()


    def generate(self, prompt: Prompt) -> str:
        messages = list(prompt)
        message = self._call_llm(messages).choices[0].message  # type: ignore
        while message.tool_calls:
            messages = self._append_tool_call_message(messages, message)
            for tool_call in message.tool_calls:
                name, params = self._extract_tool_call(tool_call)
                result = self._call_tool(name, params)
                messages = self._append_tool_result_message(messages, tool_call, result)
            message = self._call_llm(messages).choices[0].message  # type: ignore
        return message.content or ""


    def _append_tool_result_message(self, messages, tool_call, result):
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
        return messages


    def _call_tool(self, name: str, params: dict) -> str:
        return self.tool_runner.call(name, params)


    def _extract_tool_call(self, tool_call):
        name = tool_call.function.name
        params = json.loads(tool_call.function.arguments or "{}")
        return name, params


    def _append_tool_call_message(self, messages, message):
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                }
                for tool_call in message.tool_calls
            ],
        })
        return messages


    def _extract_enabled_tool_names(self, messages_in: list[dict]) -> list[str]:
        enabled: list[str] = []
        pattern = re.compile(r"\[TOOL\s+([^\]]+)\]")
        for message in messages_in:
            if message.get("role") != "user": continue
            content = message.get("content", "")
            if not isinstance(content, str): continue
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
        for message in messages_in:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                cleaned.append({**message, "content": pattern.sub(" ", message["content"]).strip()})
            else: cleaned.append(message)
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










#############################################################################
#############################################################################
#############################################################################
#############################################################################



class DiffValue(ValueFunction):
    def __init__(self) -> None: pass
    def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
        return abs(int(coalition_generation.output) - 275268)


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