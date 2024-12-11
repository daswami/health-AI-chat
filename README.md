**Healthcare AI Chat**

This tool enables healthcare professionals to retrieve patient and medication information effortlessly using natural language, powered by LLMs.

Key Features

- Natural Language Interface: Query patient and medication data using plain English prompts.
  - Example: "Show me all medications prescribed this month."
- SQL Database Backend: Data is stored in a relational database for structured, reliable querying.
- Secure Integration: Built with OpenAI's API for robust natural language processing.

 How It Works

1. Data Storage: Patient and medication data are stored in an SQL database.
2. Query Processing: Users input natural language queries via the CLI.
3. LLM Interpretation: The OpenAI API translates queries into SQL commands.
4. Results Output: The system executes the SQL query and displays results in the CLI.

**LLM Techniques Utilized**

- Few-shot Learning: Providing the LLM with relevant example queries and answers
- Meta Prompting: Providing the LLM with query structure and syntax, reducing token usage
- Feedback Loops: Incorporating positive answers from the LLM as future context to tailor the model to the user's preferences
- Cost and Scalability in Mind: The lightweight gpt-4o-mini model with basic capabilities is utilized to bring costs down while keeping accuracy.

Future Improvements:
1. Contextual Querying: Having the generated queries include more related terms based on the language in the prompt.
2. Scalability: Generalizing the application to work with multiple DB schemas
3. Few-shot Learning Adapations: Provide the ability for users to tailor the model during runtime for fast deployment and improved accuracy at a low cost.

Setup 

1. Create a Virtual Environment
2. Install Dependencies with pip install requirements.txt
3. Set Environment Variables:
   Set the shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`)
   with the OPENAI_API_KEY and DATABASE_URL variables and then run source ~/.bashrc


