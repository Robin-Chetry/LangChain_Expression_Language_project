




''' 
**LangChain + LangServe API Overview**

---

### 1. What is an API?

**API** stands for **Application Programming Interface**. It defines a way for different software applications to communicate with each other by sending requests and receiving responses.

Example:

* Your app sends a request: "Give me the weather for Delhi"
* The server sends back a response: "32°C and sunny"

---

### 2. What is a Web API?

A **Web API** is an API that works over the internet using HTTP. It allows systems (like a website or app) to talk to your server.

Example request to a web API:

```
POST /translate
Body: { "text": "hello", "language": "French" }
Response: "bonjour"
```

---

### 3. What is LangServe?

**LangServe** is a library from LangChain that makes it easy to expose your LangChain chains as a Web API.

Normally, a LangChain `chain` only works inside Python. LangServe lets you serve it so anyone can access it over HTTP.

It handles:

* Creating an API endpoint (like `/chain/invoke`)
* Parsing inputs from users
* Running the chain
* Returning the output

---

### 4. Why are we using LangServe?

You created a chain:

```python
chain = prompt_template | model | parser
```

Now you want others (or yourself) to use this chain over the web. Instead of writing manual FastAPI logic, LangServe simplifies this:

```python
add_routes(app, chain, path="/chain")
```

This automatically creates:

* A POST endpoint: `/chain/invoke`
* API docs: `/chain/docs`
* Playground UI: `/chain/playground`

---

### 5. How LangServe Works

1. You send an HTTP POST request with input JSON:

```json
{
  "language": "Hindi",
  "text": "My name is Robin"
}
```

2. LangServe feeds this input into the chain:

   * Prompt is filled
   * Model runs
   * Output is parsed

3. The response is returned as plain text:

```json
"मेरा नाम रॉबिन है"
```

---

### 6. Summary Table

| Term       | Description                                                        |
| ---------- | ------------------------------------------------------------------ |
| API        | A way for two software systems to communicate                      |
| Web API    | An API that uses HTTP to work over the web                         |
| LangChain  | A library for building LLM-based applications                      |
| LangServe  | A tool that makes LangChain chains accessible as web APIs          |
| FastAPI    | A Python framework used to build and run the server                |
| Why use it | To easily expose your LLM logic over HTTP without boilerplate code |

---

This setup allows anyone to send language + text and get translated results over the internet using your model chain.

'''




from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser=StrOutputParser()

##create chain
chain=prompt_template|model|parser



## App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces",
    # openapi_url=None
)


## Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)

