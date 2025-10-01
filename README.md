# mchat_core

A collection of convenience functions for using LLM models and autogen agents driven by configuration files. Primarily used in [MChat](https://github.com/jspv/mchat) but written to be useful in a variety of use cases.

---

## Installation and Usage

Dependencies are declared in `pyproject.toml`. Development and dependency management are primarily done using `uv`.

The `[tools]` dependency group includes additional requirements for bundled LLM Module tools. You only need these if you plan on using the provided tools.

### Optional Tools Dependencies

- With uv (recommended):
  - uv sync --group tools
- With other package managers: see `pyproject.toml` for the `tools` dependency group

### Running Tests

- Run all tests:
  - pytest
- Show skip reasons:
  - pytest -rs
- Run only the tools tests:
  - pytest -m tools -rs
- Exclude tools tests:
  - pytest -m "not tools"

Live LLM tests (opt-in; real API calls):
- By default, tests marked `live_llm` are excluded via pytest.ini (`addopts = -m "not live_llm"`).
- To run them, set an API key and select the marker:
  - OPENAI_API_KEY=your_key pytest -m live_llm -rs

Note:
- Tool tests are marked with the "tools" marker and auto-skip if optional packages are not installed.
- Live LLM tests are marked with the `live_llm` marker and require an OpenAI API key.

---

## Configuration

**Note:** This code is actively developed. Instructions and sample configs may become outdated.

**See the examples.ipynb notebook for more details**

Configuration is managed in three files:

- `settings.toml`
- `.secrets.toml` *(optional but recommended)*
- `agents.yaml`

Edit `settings.toml` to configure your application. Here’s a guide to the available options:

---

### Models

Sections should start with `models.` (with a period) and contain no other periods in the section name.  
Format: `models.type.model_id` — `model_id` will appear in list of available models

    [models.chat.gpt-4o]
    api_key = "@format {this.openai_api_key}"
    model = "gpt-4o"
    api_type = "open_ai"
    base_url = "https://api.openai.com/v1"

> NOTE: Image models and settings here are only for explicitly calling image models from prompts.
> The `generate_image` tool uses only the API key.

---

#### Required Fields

**Chat Models**
- `api_type`: "open_ai" or "azure"
- `model_type`: "chat"
- `model`: Name of the model
- `api_key`: Your key or Dynaconf lookup
- `base_url`: (if required by API)

**Azure Chat Models (additional)**
- `azure_endpoint`: URL for your endpoint
- `azure_deployment`: Deployment name for the model
- `api_version`: API version

**Image Models**
- `api_type`: "open_ai" or "azure"
- `model_type`: "image"
- `model`: Name of the model
- `size`: Size of images to generate
- `num_images`: Number of images to generate
- `api_key`: Your key or Dynaconf lookup

---

### Default Settings

- `default_model`: Default model to use
- `default_temperature`: Default temperature for generation
- `default_persona`: Default persona for generation

---

### Mini Model Configuration

mchat has internal utilites that will use an LLM for simple tasks such as deciding when a conversation is complete or generating short summary labels for a conversation.  If mini_model is specified it will use this model, otherwise it will use the default.

Configurable properties:
- `mini_model`: Model ID used for for internal utilities

---

### Secrets Configuration

Some sensitive config settings (like API keys) should be in `.secrets.toml`:

    # .secrets.toml
    # dynaconf_merge = true

    # Replace the following with your actual API keys
    # openai_models_api_key = "oai_ai_api_key_goes_here"
    # ms_models_api_key = "ms_openai_api_key_goes_here"

---

## Agents & Teams

mchat_core provides a session-based agent management system where each conversation is encapsulated in an `AgentSession` object. This allows for:

- Multiple concurrent conversations with different agents
- Per-session state management (memory, streaming settings, etc.)
- Clean separation between agent definitions and active conversations

### Basic Usage

```python
from mchat_core.agent_manager import AgentManager

# Initialize with agent definitions (path or inline YAML)
manager = AgentManager(agent_paths=["agents.yaml"])  

# Create a new conversation session (returns AgentSession)
session = await manager.new_conversation("my_agent")

# Use the session for conversation
result = await session.ask("Hello, how can you help me?")

# Each session maintains its own state
await session.clear_memory()  # Clear this session's memory
session.cancel()              # Cancel ongoing operations for this session
```

### Agent Configuration

mchat_core provides:
- A default persona
- Example agents: *linux computer* & *financial manager*
- Example teams: round-robin and selector

You can add more agents and teams at the top level in `agents.yaml` (same directory as this README), following the structure in `mchat/default_personas.yaml`.  
When configuring personas, the `extra_context` list lets you define multi-shot prompts—see the `linux computer` persona in `mchat/default_personas.json` for an example.

#### Context options (conversation memory window)

You can control how much prior conversation the agent sees using the `context` block on each agent. Supported types:

- Unbounded (default): keeps all messages
  context:
    type: unbounded

- Buffered: keep only the last N messages
  context:
    type: buffered
    buffer_size: 20

- Token-limited: keep messages up to a token limit (or use the model's remaining tokens if not set)
  context:
    type: token
    token_limit: 4000  # optional

- Head and Tail: keep the first head_size and last tail_size messages
  context:
    type: head_tail
    head_size: 3
    tail_size: 20

If `type` is unknown or the configuration is invalid, it falls back to unbounded.

**Warning**: if the LLM model does not support sytem messages, the prompt will be injected as the first message and will be treated just like any other user message, which means 'buffered' and 'token' can cause the prompt to be removed.  A good option here is to use 'head_tail' which will keep the prompt in the head.  

#### Tool configuration (built-in + MCP)

Agents can be equipped with tools to extend capabilities. You can mix built-in tools with Model Context Protocol (MCP) tools.

Built-in tools example:

```yaml
my_agent:
  type: agent
  description: An agent with web search capabilities
  prompt: You can help users by searching the web for information.
  tools:
    - web_search          # built-in
    - generate_image      # built-in
```

MCP via STDIO (command-line server):

```yaml
searcher:
  type: agent
  description: AI assistant with Google search via MCP
  prompt: You can search Google for current information and cite sources.
  tools:
    - mcp: uvx --from . google-search-mcp
      cwd: /path/to/google_search_mcp
      # Optional extras
      env:
        API_KEY: "${API_KEY}"
      timeout: 30
```

MCP via HTTP (web server):

```yaml
web_searcher:
  type: agent
  description: AI assistant using a web-based MCP server
  prompt: Use the MCP search tool to find and cite current information.
  tools:
    - mcp:http://localhost:8000/mcp
      timeout: 30
```

Notes:
- MCP tools are registered as placeholders at startup and resolved to real tools when a conversation begins.
- STDIO servers support `cwd`, `env`, `timeout`, and extra `args`.
- HTTP servers support `url` and `timeout`.
- See `pyproject.toml` for optional tool dependencies (group: `tools`).

Mixed tools:

```yaml
hybrid_agent:
  type: agent
  description: Agent with both built-in and MCP tools
  tools:
    - web_search
    - generate_image
    - mcp: uvx --from . custom-mcp-server
      cwd: /path/to/server
```

### Session Management

**Important**: Always use `manager.new_conversation()` to create sessions. Direct instantiation of `AgentSession` is not supported and will raise a `RuntimeError` with guidance on proper usage.

### Key Features

- **Concurrent Conversations**: Multiple sessions can run simultaneously with different agents
- **Per-Session State**: Each session maintains its own memory, streaming settings, and context
- **Session-Specific Overrides**: Override model, temperature, streaming, and callbacks per session
- **Independent Control**: Cancel, terminate, or clear memory for individual sessions
- **Property Access**: Access agent properties (prompt, description, model) from the session

### API Reference

```python
# Create session
session = await manager.new_conversation(
    agent="agent_name",
    model_id="gpt-4o",           # Optional: override agent's default model
    temperature=0.7,             # Optional: override agent's default temperature  
    stream_tokens=True,          # Optional: override manager default
    message_callback=my_callback # Optional: override manager default
)

# Session methods
result = await session.ask("Your question here")
session.cancel()                    # Cancel ongoing operations
session.terminate()                 # Terminate conversation
await session.clear_memory()             # Clear conversation memory
memory = await session.get_memory() # Get current memory state

# Session properties (read-only)
session.agent_name    # Name of the agent
session.description   # Agent description
session.prompt        # Agent system prompt
session.model         # Current model ID
session.stream_tokens # Current streaming state (can be modified)
```

## Contributing

Thank you for considering contributing to the project! To contribute, please follow these guidelines:

1. Fork the repository and clone it to your local machine.

2. Create a new branch for your feature or bug fix:

   ```shell
   git checkout -b feature/your-feature-name
   ```

   Replace `your-feature-name` with a descriptive name for your contribution.

3. Make the necessary changes and ensure that your code follows the project's coding conventions and style guidelines - which currently are using PEP 8 for style and *black* for formatting 

4. Commit your changes with a clear and descriptive commit message:

   ```shell
   git commit -m "Add your commit message here"
   ```

5. Push your branch to your forked repository:

   ```shell
   git push origin feature/your-feature-name
   ```

6. Open a pull request from your forked repository to the main repository's `main` branch.

7. Provide a clear and detailed description of your changes in the pull request. Include any relevant information that would help reviewers understand your contribution.



## License
This project is licensed under the [MIT License](LICENSE).

## Contact
Feel free to reach out to me at @jspv on GitHub
