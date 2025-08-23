# mchat_core

A collection of convenience functions for using LLM models and autogen agents driven by configuration files. Primarily used in [MChat](https://github.com/jspv/mchat) but written to be useful in a variety of use cases.

---

## Installation and Usage

Dependencies are declared in `pyproject.toml`. Development and dependency management are primarily done using `uv`.

The `[tools]` dependency group includes additional requirements for bundled LLM Module tools. You only need these if you plan on using the provided tools.

### Optional Tools Dependencies

- With uv (recommended):
  - uv sync --group tools
- With pip (install only what you need - see pyproject.toml)
  - pip install tzlocal fredapi chromadb beautifulsoup4

### Running Tests

- Run all tests:
  - pytest
- Show skip reasons:
  - pytest -rs
- Run only the tools tests:
  - pytest -m tools -rs
- Exclude tools tests:
  - pytest -m "not tools"

Note: Tool tests are marked with the “tools” marker and will auto-skip if optional packages are not installed.

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

### Memory Model Configuration (Currently Disabled)

mchat can maintain conversational memory for long chats. When memory size exceeds model limits, conversations are summarized using a designated model (ideally, a cost-effective one).

Configurable properties:
- `memory_model`: Model ID used for memory (should match one in `models`)
- `memory_model_temperature`: Temperature for memory summarization
- `memory_model_max_tokens`: Token limit for memory model

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

mchat provides:
- A default persona
- Example agents: *linux computer* & *financial manager*
- Example teams: round-robin and selector

You can add more agents and teams at the top level in `agents.yaml` (same directory as this README), following the structure in `mchat/default_personas.yaml`.  
When configuring personas, the `extra_context` list lets you define multi-shot prompts—see the `linux computer` persona in `mchat/default_personas.json` for an example.

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
