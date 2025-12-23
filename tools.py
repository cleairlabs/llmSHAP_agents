import inspect
from pydantic import BaseModel


class MultiplyArgs(BaseModel):
    num1: int
    num2: int

class WeatherArgs(BaseModel):
    city: str


class ToolRunner:
    def __init__(self) -> None:
        self._dispatch: dict[str, tuple[type[BaseModel], object]] = {
            "multiply_two_numbers": (MultiplyArgs, multiply_two_numbers),
            "get_weather": (WeatherArgs, get_weather),
        }

    def call(self, name: str, params: dict) -> str:
        entry = self._dispatch.get(name)
        if entry is None: return f"Unknown tool: {name}"
        args_model, func_obj = entry
        args = args_model.model_validate(params).model_dump()
        func = func_obj
        result = func(**args)  # type: ignore[misc]
        return str(result)
    

def multiply_two_numbers(num1: int, num2: int) -> int:
    print(f"[CALLED] {inspect.currentframe().f_code.co_name}")  # type: ignore
    return num1 * num2


def get_weather(city: str) -> str:
    print(f"[CALLED] {inspect.currentframe().f_code.co_name}")  # type: ignore
    return "22 C and sunny"


def _openai_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    required = schema.get("required")
    if not required: schema["required"] = list(schema.get("properties", {}).keys())
    return schema


def build_openai_tool_registry() -> dict[str, dict]:
    return {
        "multiply_two_numbers": {
            "type": "function",
            "function": {
                "name": "multiply_two_numbers",
                "parameters": _openai_schema(MultiplyArgs),
                "strict": True,
            },
        },
        "get_weather": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": _openai_schema(WeatherArgs),
                "strict": True,
            },
        },
    }