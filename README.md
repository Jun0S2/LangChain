# Section 1

## What is LangChain ?

Use models as black box - so without using Machine Learning theories, developers can use them!

> Chain == Chain of Actions

## Environment Setup

```
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchainhub
```

## Connecting Prompt

- Prompt : Receive input
- Chat Models : Interact with LLM

```
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
```

### What are Chains ?

Allow us to `combine` multiple chains together and make one application.

## Open Source LLMs

It depends on the task, big models perform better on agent / brain ones but for lighter tasks - usually it performs pretty well

### Ollama

**`Ollama`** is a tool that runs **LLM (Large Language Model) on local environement**

💡 **In Short:** 👉 "Open Source tool allows us to run models like Chat GPT **in local environemnt** "
After installing ollama in OS, install with pip install and play around

```
pip install langchain-ollama
```

#### 🔍 Ollama Model Response

<details>
  <summary>Click to see the response</summary>
Here are a short summary and two interesting facts about Elon Musk:

**Summary:**
Elon Musk is a businessman, entrepreneur, and inventor who has made significant contributions to various industries, including technology, space exploration, and sustainable energy. He is the CEO of SpaceX, Tesla, Inc., Neuralink, and The Boring Company, among other ventures.

**Interesting Facts:**

1. **From Poor to Billionaire:** Elon Musk was born into a wealthy family in South Africa, but he grew up in Canada after his family immigrated due to financial difficulties. He eventually moved back to the United States to attend Stanford University, where he dropped out to pursue his entrepreneurial ventures.

2. **A Self-Taught Genius:** Despite dropping out of college, Elon Musk has credited himself with learning most of what he knows through self-study and online courses. He has also stated that he believes in taking risks and trying new things, which has contributed to his success in various fields.

Let me know if you'd like more information!

</details>

## LinkedIn

```bash
linkedin_runner.py -- main
third_parties
ㄴ __init__.py
ㄴ linkedin.py
```

[x] ProxyURL API
[x] Data Clean Up
[x] Testing

# Agent

Basically, Engine of Agent is LLM.

- Uses LLM to find what the client wants
- performs the tasks
- returns results

## What is Agent ?

**`Agent`** is an AI that utilizes tools to accomplish objectives.

- **`agent_executor`** → An object that actually executes the agent.
- **`create_react_agent`** → A function that creates a ReAct-based agent.
