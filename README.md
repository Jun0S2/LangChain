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
