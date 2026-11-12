# General LLM pipeline

This project runs **one configured LLM task** over rows in a CSV. The current example summarizes each input in one sentence using OpenRouter and Llama 3.3 70B Instruct.

## Quick start

Run all commands from the repository root.

1. Create and activate the Conda environment:

	```sh
	conda env create -f environment.yml
	conda activate language-model
	```

	If you already created this environment, just activate it.

2. Add your OpenRouter API key to a root-level `.env` file:

	```text
	OPENROUTER_API_KEY=your_openrouter_api_key
	```

	Keep this key private and do not commit `.env`.

3. Check the example input at [data/input.csv](data/input.csv). Its headers must match `dataset.text_column` and `dataset.id_column` in [project.yml](project.yml). The included sample has three rows.

4. Configure the task:
	- In [project.yml](project.yml), set the project/task names and the model. `llm.name` is a display label; `llm.model` is the OpenRouter model ID.
	- Write the task instructions in [prompts/task.txt](prompts/task.txt). `${statement_text}` is replaced with the text from each CSV row. `${statement_id}` and CSV column names are also available.
	- Define the expected structured response in [schemas/response.json](schemas/response.json). The current schema expects a JSON object with one string field, `result`.

5. Run a small test or a full run:

	```sh
	python src/app.py --test
	python src/app.py
	```

	`--test` makes real API calls for up to three rows, once per row. It can incur API usage. It writes results under `results/test/`.

## Configuration

All relative paths are resolved from the directory where you run the command, so run from the repository root. The `sampling` section controls the row count (`null` means all rows) and random seed. `run.repetitions` controls calls per row. CLI options `--n-statements N` and `--repetitions N` override those values. Use `--config path/to/other.yml` to run a different project configuration; its relative paths still resolve from the current working directory.

The prompt template and schema are separate files so task instructions and output structure can be changed without editing Python. With `response_format: json_schema`, the model is asked to return data matching the configured schema.

## Results

Each completed call is stored as a record containing `metadata` (input, model, timing, and run details) and `payload` (the model response). The runner always writes both formats to the configured storage directory:

- CSV: metadata fields are columns by default; `payload` is stored as JSON text in one column. Use `storage.csv_columns` in [project.yml](project.yml) to choose columns and rename them. Values can come from call metadata, the original input row, or the model response—for example `summary: payload.result` or `category: input.category`.
- JSON: the complete `{metadata, input, payload}` record is preserved, including all columns from the original CSV row.

`input_group_id` is shared by repetitions for one input row, so it links those calls together. `replicate_number` is the 1-based call number for that input, and `session_id` uniquely identifies one individual call. `started_at` and `received_at` are UTC timestamps. The same timestamped base name is used for both files. `csv_columns` affects only CSV; JSON remains a full record archive. Set `storage.include_prompt: true` if you also want the rendered prompt saved in metadata; it is sent to the model either way.

## Pipeline overview

`src/app.py` parses options → `src/project_config.py` loads YAML and renders the prompt/loads the schema → `src/pipeline.py` reads rows and coordinates calls → `src/llm_client.py` calls OpenRouter → `src/storage.py` writes CSV and JSON.

Dependencies and the Python version are listed in [environment.yml](environment.yml) and [requirements.txt](requirements.txt).
