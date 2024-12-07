# How to run BenchPRESS
1. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Copy the example environment file to create a new `.env` file:
    ```bash
    cp .example.env .env
    ```

4.  Add your API keys to the `.env` file:
    ```env
    OPENAI_API_KEY="your_api_key_here"
    ANTHROPIC_API_KEY="your_api_key_here"
    GROQ_API_KEY="your_api_key_here"
    ```
5. Adjust the number of examples you wish to generate and evaluate on. 50 examples takes about 30 minutes to run on a M1 Mac.

6. Run the benchmark script:
    ```bash
    python3 benchmark.py
    ```
- Please note that running `benchmark.py` can result in a different result than presented in the poster since the dataset is generated randomly
- Please note that more examples in the dataset will result in longer evaluation times, and API usage costs.  You may adjust the number of iterations by changing the `num_tests` variable and modifying the list of models in the `LIST_MODELS` dictionary.
- This is the same code as the notebook `benchpress.iypnb` ported to a Python script.