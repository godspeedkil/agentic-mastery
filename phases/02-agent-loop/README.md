# Phase 2: Tool use and the agent loop

An agent is a model that can act: it requests actions, your code carries them out, and the
results shape what it does next. This phase builds that loop from scratch, with no framework,
on top of the Phase 1 client.

## 2.1 Function calling

### How it works

The model never runs anything. It can only ask:

1. **You describe tools** in the request: a name, a description, and a JSON Schema for the
   arguments.
2. **The model asks for a call** instead of answering: a tool name, arguments that follow the
   schema, and a call ID.
3. **Your code runs the tool** and sends the result back, tagged with that call ID.
4. **The model continues**: it asks for more calls or gives its answer.

Steps 2–4 repeat; that repetition is the agent loop (2.2).

### On the wire

| | Anthropic Messages | OpenAI Responses |
| --- | --- | --- |
| Tool definition | `name`, `description`, `input_schema` | `type: "function"`, `name`, `description`, `parameters`, `strict` |
| The model's call | `tool_use` content block: `id`, `name`, `input` (an object) | `function_call` output item: `call_id`, `name`, `arguments` (a JSON string) |
| "Waiting on tools" | stop reason `tool_use` | `function_call` items in the output |
| Your result | `tool_result` block in the next user message: `tool_use_id`, `content`, `is_error` | `function_call_output` input item: `call_id`, `output` |
| Forcing a choice | `tool_choice`: `auto`, `any`, `tool`, `none` | `tool_choice`: `auto`, `required`, a named function, `none` |

The concepts match; the shapes don't. As with messages in Phase 1, the client defines one
shape and each adapter translates.

### What to get right

- **One response can request several calls.** Run them all and return every result in the
  next request, each with its own call ID. Parallel calls can be turned off
  (`disable_parallel_tool_use`, `parallel_tool_calls: false`) when order matters.
- **Arguments are model output.** Strict mode constrains them to the schema, as structured
  outputs do (1.5), but not their values: an ID can be well-formed and still not exist.
  Validate before acting.
- **Errors are results.** A failing tool returns an error message to the model (with
  `is_error` on Anthropic) instead of crashing the program. Most models recover: they fix the
  argument or try another tool.
- **Reasoning travels with tool calls.** Reasoning models think before and between calls
  (1.6). Anthropic requires the assistant turn's thinking blocks to come back complete and
  unmodified alongside the tool results. OpenAI, when used without server-side storage
  (`store: false`), returns reasoning as encrypted items that must be requested
  (`include: ["reasoning.encrypted_content"]`) and sent back in their original order.
- **Tool definitions are input tokens.** They are sent with every call and sit at the start of
  the prompt, so keep them stable: they are part of the cacheable prefix (1.7).
- **Forcing a tool can conflict with thinking.** Anthropic models with extended thinking
  accept only `auto` or `none`.

### In the code

A tool is described by a `ToolSpec`. Its arguments are a Pydantic model, as the answer's shape
is with `output_type` (1.5): one class gives the schema sent to the model and later validates
what comes back (`core/src/core/llm/types.py`):

```python
class ToolSpec(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str
    input_type: type[BaseModel]
```

`Request`, the client's description of one call, carries them in its `tools` field:

```python
tools: list[ToolSpec] = Field(default_factory=list)
```

Each provider adapter, the module that translates between these types and one SDK, turns a
spec into its provider's definition, with the schema in strict mode. `to_strict_json_schema`
is the OpenAI SDK's converter from a Pydantic model (`core/src/core/llm/openai_provider.py`;
the Anthropic one differs only in field names):

```python
def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": to_strict_json_schema(spec.input_type),
        "strict": True,
    }
```

The model's request and your answer are two more kinds of content block, the pieces a
message is made of:

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON, exactly as the model wrote it


class ToolResult(BaseModel):
    call_id: str  # the ToolCall's id
    content: str
    is_error: bool = False
```

Arguments stay a JSON string because OpenAI sends one, and the Anthropic adapter serializes
its object to match. Either way they are validated like structured output, where `spec` is
the called tool's `ToolSpec` and `call` the `ToolCall`:
`spec.input_type.model_validate_json(call.arguments)`.

A message refuses blocks its role can't carry, so a result put in the wrong turn fails when
the message is built, not at the provider:

```python
ROLE_BLOCKS: dict[Role, set[str]] = {
    "user": {"text", "tool_result"},
    "assistant": {"text", "tool_call", "reasoning"},
}
```

`Response`, the normalized reply, lists the calls the model made:

```python
@property
def tool_calls(self) -> list[ToolCall]:
    return [block for block in self.content if isinstance(block, ToolCall)]
```

OpenAI reports a response that requests calls as plain `completed`, so its adapter derives
the `tool_use` stop reason itself, matching Anthropic's. `called` is true when the output
contains a function call, and `raw` is the provider's own status, kept for debugging:

```python
return ("tool_use" if called else "end_turn"), raw
```

Reasoning is a third kind of block, holding the provider's data untouched so it can be sent
back as received. Each adapter sends back only its own provider's reasoning:

```python
class Reasoning(BaseModel):
    provider: str
    data: dict[str, Any]
```

For OpenAI, which keeps nothing between calls when `store` is false, the encrypted reasoning
has to be requested whenever tools are sent:

```python
params["include"] = ["reasoning.encrypted_content"]
```