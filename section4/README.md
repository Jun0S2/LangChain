## Environment Setup

Install the necessary libraries:

```bash
pip install langchain openai python-dotenv
```

Ensure that you have a `.env` file with your API keys for OpenAI.

---

## Tool Decorator

A tool is a Python function that has been decorated with the @tool decorator from langchain.agents.

```
section4 git:(main) $ black .
reformatted /Users/june/Workspace/LangChain/section4/react-langchain/main.py

All done! ✨ 🍰 ✨
```

take function -> convert it into langchain tool

```
from langchain.agent import tool

# tool decorator : convert into langchain tool
@tool
def ...

```

## Prompt

We can see prompts in LangSmith Prompt
Example : [hwchase17/react](https://smith.langchain.com/hub/hwchase17/react)

---

## Key Concepts

### 1. **LangChain Agents**

Agents in LangChain are components that can autonomously decide which actions to take. They leverage tools and reasoning processes to solve tasks step-by-step.

**ReAct Framework**: The agent follows the ReAct (Reasoning + Acting) framework:

1. **Thought**: Reason about what to do next.
2. **Action**: Choose a tool to execute.
3. **Action Input**: Provide input to the tool.
4. **Observation**: Observe the result of the tool's execution.
5. **Final Answer**: Conclude with the final result.

### 2. **Tools**

Tools are functions that the agent can call to perform specific tasks. In this example, we define a tool to calculate the length of a string.

### 3. **Callbacks**

Callbacks allow monitoring of the agent's behavior at different stages of execution. The custom callback handler logs when the LLM starts and ends.

### 4. **LangChain Expression Language (LCEL)**

LCEL enables chaining multiple operations together using the `|` operator. This modular approach helps build workflows where data flows seamlessly from one component to the next.

---

## Project Structure

```
.
├── main.py          # Main script to run the LangChain agent
├── callbacks.py     # Custom callback handler to log LLM events
├── .env             # Environment file with API keys
└── README.md        # Documentation for the project
```

---

## Code Overview

### **1. Tool Definition**

In `main.py`, we define a tool to calculate the length of a string:

```python
from langchain.agents import tool

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a given text."""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)
```

### **2. Callback Handler**

In `callbacks.py`, we create a custom callback handler to log when the LLM starts and ends:

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM is starting with prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        print(f"LLM output: {response.text}")
```

### **3. Agent Setup**

In `main.py`, we set up the LangChain agent with the defined tool and callback handler:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from callbacks import AgentCallbackHandler

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define tools
tools = [get_text_length]

# Define prompt template
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Initialize LLM with callback handler
llm = ChatOpenAI(
    temperature=0,
    model_kwargs={"stop": ["\nObservation", "Observation"]},
    callbacks=[AgentCallbackHandler()],
)

# Configure the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)

# Run the agent
intermediate_steps = []
agent_step = None

while not isinstance(agent_step, AgentFinish):
    agent_step = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

if isinstance(agent_step, AgentFinish):
    print("### AgentFinish ###")
    print(agent_step.return_values)
```

---

## Expected Output

When you run `main.py`, you should see the following output:

```
Hello ReAct LangChain!
LLM is starting with prompts: ['What is the length of the word: DOG']
AgentAction(tool='get_text_length', tool_input='DOG')
get_text_length enter with text='DOG'
observation=3
LLM is starting with prompts: ['What is the length of the word: DOG']
LLM output: The length of the word DOG is 3.
### AgentFinish ###
{'output': 'The length of the word DOG is 3.'}
```

---

## Conclusion

This project demonstrates how to:

- Define custom tools and integrate them with LangChain agents.
- Use the ReAct framework to guide an agent's reasoning process.
- Implement callback handlers to monitor and debug the agent's execution.

Feel free to expand this project by adding more tools or modifying the prompt template to handle different types of queries.
