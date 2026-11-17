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

## 2.2 The agent loop

### The loop

An agent is a model, a set of tools, instructions, and a loop that connects them:

```
messages = [task]
repeat:
    response = model(instructions, messages, tools)
    append the whole response to messages     # text, reasoning, and tool calls
    if the response has no tool calls: done   # its text is the answer
    run every tool call
    append all the results to messages, as one user turn
```

The model decides what happens next at every step; the code only executes. That is the
difference from a **workflow**, where code fixes the sequence of steps and the model fills in
each one. Workflows are cheaper and more predictable; agents handle tasks whose steps can't be
known in advance. Use the simplest one that works
([Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)).

### Reasoning and acting

The pattern of alternating thought and action was named **ReAct**
([Yao et al., 2022](https://arxiv.org/abs/2210.03629)): the model reasons about what it knows,
acts to learn more, observes the result, and repeats. Early implementations prompted for it
with "Thought: / Action: / Observation:" text formats and parsed the output. Current models
do it natively: reasoning happens in thinking tokens, actions are tool calls, and
observations are tool results. Text formats are no longer needed.

### Running the tools

For each tool call, the loop:

1. **Finds the tool by name.** An unknown name is an error result, not a crash.
2. **Validates the arguments** against the tool's schema. Invalid arguments are an error
   result that says what was wrong, so the model can retry.
3. **Runs it** and turns the output into text. An exception inside the tool is an error
   result too.

Every call gets exactly one result, with its call ID, and all results from one response go
back together. Calls in one response are independent by construction (the model issued them
without seeing each other's results), so they can run concurrently.

### What the loop costs

Every step resends the whole history: instructions, tool definitions, the task, and every
earlier call and result. Input grows with each step, and total input grows roughly with the
square of the number of steps (1.1). Keeping the history append-only makes each request
extend the previous one, so prompt caching can serve most of it (1.7). Large tool outputs are
the usual culprit when a loop gets expensive; 2.4 covers keeping them small.

### Getting a usable answer

The loop ends when the model answers without calling a tool. For a program to use that
answer, it needs a defined shape. Two ways:

- **Structured output on the answer**, as in 1.5. "Done" stays implicit: the model ends by
  not calling a tool, and any plain-text reply ends the loop too.
- **An answer tool**: a tool whose arguments are the answer's schema, such as
  `submit_decision`. Calling it ends the loop. "Done" becomes an explicit action whose
  arguments are validated like any other call, so a malformed answer goes back to the model
  as an error result instead of ending the run.

### In the code

The loop lives in `core/src/core/agent.py`. A `Tool` pairs the `ToolSpec` the model sees with
the Python function that carries it out. `A` is the tool's argument class, so the function
receives arguments that have already been validated:

```python
@dataclass(frozen=True)
class Tool[A: BaseModel]:
    spec: ToolSpec
    run: Callable[[A], str]
```

`run_tool` handles one call. Each of its three steps either moves on or returns an error
result, a `ToolResult` with `is_error` set. `error_result` builds one, and every message
starts with `Error:` because OpenAI has no error flag (2.1):

```python
def run_tool(call: ToolCall, tools: Mapping[str, Tool]) -> ToolResult:
    tool = tools.get(call.name)
    if tool is None:
        available = ", ".join(sorted(tools))
        return error_result(call, f"Error: unknown tool {call.name!r}. Available: {available}.")
    try:
        arguments = tool.spec.input_type.model_validate_json(call.arguments)
    except ValidationError as exc:
        return error_result(call, f"Error: invalid arguments for {call.name}.\n{exc}")
    try:
        return ToolResult(call_id=call.id, content=tool.run(arguments))
    except Exception as exc:  # any failure inside a tool is reported, never raised
        return error_result(call, f"Error: {call.name} failed: {type(exc).__name__}: {exc}")
```

`run_agent` is the loop. `client.complete` is the Phase 1 client's call, so every step is
retried, priced, and recorded like any other call; it returns a `Completion`, the response
plus its record. `run` is an `AgentRun`, which collects the history in `run.messages`;
`specs` are the tools' specs, and `by_name` maps each tool's name to the tool. Each call's
record is labeled with the caller's `labels` plus its step number:

```python
while True:
    step += 1
    step_request = request.model_copy(update={"messages": list(run.messages), "tools": specs})
    completion = client.complete(
        provider, step_request, labels={**(labels or {}), "step": str(step)}
    )
    response = completion.response
    if response.content:
        run.messages.append(Message(role="assistant", content=response.content))
    if not response.tool_calls:
        return run
    results = []
    for call in response.tool_calls:
        ...  # the answer tool, below
        results.append(run_tool(call, by_name))
    run.messages.append(Message(role="user", content=results))
```

The whole response goes into the history, so reasoning and calls go back to the model on the
next step. The history is only ever appended to, so each request extends the one before,
which is what prompt caching needs. `list(run.messages)` gives each request its own copy, so
later steps can't change a request that was already sent.

The answer tool is a `ToolSpec` passed as `answer` and offered with the others. Inside the
`for` loop, a call to it is handled before `run_tool`: valid arguments end the run with the
answer, and invalid ones go back as an error result:

```python
if answer is not None and call.name == answer.name:
    try:
        run.answer = answer.input_type.model_validate_json(call.arguments)
        return run
    except ValidationError as exc:
        results.append(error_result(call, f"Error: invalid answer.\n{exc}"))
        continue
```