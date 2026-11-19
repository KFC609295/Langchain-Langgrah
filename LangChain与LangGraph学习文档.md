# Langchain&graph说明文档

> 基于官方文档（docs.langchain.com）与 GitHub 组织 langchain-ai 的内容整理，面向学习与快速上手。
> 整理时间：2026-11-19

---

## 目录

1. [LangChain 生态全景](#一langchain-生态全景)
2. [LangChain 快速入门](#二langchain-快速入门)
3. [LangChain 核心概念](#三langchain-核心概念)
4. [LangGraph 快速入门](#四langgraph-快速入门)
5. [LangGraph 核心概念](#五langgraph-核心概念)
6. [LangGraph 关键能力](#六langgraph-关键能力)
7. [GitHub 生态仓库](#七github-生态仓库)
8. [产品对比与选型](#八产品对比与选型)
9. [历史演进时间线](#九历史演进时间线)
10. [参考资源](#十参考资源)

---

## 一、LangChain 生态全景

LangChain 不是一个单一框架，而是一个覆盖 AI Agent 全生命周期的产品矩阵。四个核心产品分层协作：

| 产品 | 定位 | 关键作用 |
|------|------|----------|
| **LangChain** | Agent 框架 | 模型、工具、Agent 循环的抽象与集成 |
| **LangGraph** | 编排运行时 | 持久执行、流式、人机交互、持久化 |
| **Deep Agents** | Agent 外壳（harness） | 内置规划、子 Agent、文件系统、上下文管理 |
| **LangSmith** | 平台 | 追踪、评估、提示词、部署 |

**产品分层关系（自底向上）：**

```
LangGraph          ← 最底层：编排运行时（图、状态、节点、边）
   ↑
LangChain (create_agent)  ← Agent 框架：最小化的 Agent 循环
   ↑
Deep Agents        ← 更高层：开箱即用的 Agent 外壳
LangSmith          ← 横跨所有框架：可观测性 + 评估 + 部署
```

- **Deep Agents**：在 LangGraph 之上的 agent harness，内置规划、子 Agent、文件系统工具和上下文管理。适合想要"开箱即用"跑生产级 Agent、又不想自己管基础设施的场景。
- **LangChain**（`create_agent`）：高度可定制的最小 harness，容易根据你的用例和数据做裁剪。
- **LangGraph**：低级编排框架，用于需要确定性工作流与 Agent 工作流混合的高级需求。
- **LangSmith**：追踪、调试、评估任意框架构建的 Agent。

> **核心公式：Agent = Model + Harness**
> Agent 是"模型在循环中调用工具直到任务完成"；harness 是循环之外的一切——提示词、工具、以及塑造模型行为的中间件（middleware）。

---

## 二、LangChain 快速入门

### 2.1 定位

LangChain 提供 `create_agent`：一个**极简、高度可配置**的 Agent 外壳。你可以用模型、工具、提示词和中间件组合出恰好符合用例的 Agent。

支持 OpenAI、Anthropic、Google 等多家模型提供商。

### 2.2 安装

LangChain 包与各模型/集成的 provider 包是独立的。

```bash
# 安装 LangChain 核心（需 Python 3.10+）
pip install -U langchain

# 或使用 uv
uv add langchain
```

安装具体集成的 provider 包：

```bash
# OpenAI 集成
pip install -U langchain-openai

# Anthropic 集成
pip install -U langchain-anthropic
```

> 注意：LangChain 提供对数百个 LLM 和数千个其他集成的支持，这些都在独立的 provider 包中。完整列表见官方 Integrations 页面。

### 2.3 第一个 Agent

```python
# pip install -qU langchain "langchain[openai]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-6.1-sol",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
```

### 2.4 配置 API Key

获取任意受支持模型提供商的 API key，在 shell 或 `.env` 文件中设置：

```bash
export OPENAI_API_KEY="your-api-key"
# 或
export ANTHROPIC_API_KEY="your-api-key"
# 或
export GOOGLE_API_KEY="your-api-key"
```

加载 `.env` 文件可用 `python-dotenv` 的 `load_dotenv()`。

### 2.5 构建真实世界的 Agent

Quickstart 中构建了一个"文学数据助手"，关键步骤：

1. **详细系统提示词**——定义 Agent 角色与行为
2. **创建工具**——用 `@tool` 装饰器定义函数，让模型能调用外部系统
3. **模型配置**——用 `init_chat_model` 设置温度、超时、最大 token 等
4. **对话记忆**——用 checkpointer 维护跨轮次状态
5. **Deep Agents**——对比开箱即用的能力
6. **测试** Agent

```python
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")

model = init_chat_model(
    "gemini-3.1-pro-preview",
    model_provider="google-genai",
    temperature=0.5,
    timeout=600,
    max_tokens=25000,
    streaming=True,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt="You are a literary data assistant.",
    checkpointer=checkpointer,
)
```

> **LangChain Agent vs Deep Agent 的关键差异**：Deep Agent 内置了规划（`write_todos`）、虚拟文件系统（`grep`/`read_file`）、子 Agent 派生等能力，能真正回答"文件里有多少行包含 Gatsby"这类需要代码执行的问题；而普通 LangChain Agent 缺少这些工具时只能返回估算值或 `null`。

### 2.6 追踪 Agent 调用（LangSmith）

```bash
export LANGSMITH_AGENT_ID="<your-agent-id>"
export LANGSMITH_AGENT_ENVIRONMENT="local"
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

---

## 三、LangChain 核心概念

### 3.1 Agent = Model + Harness

- **Agent**：模型在循环中调用工具直到任务完成。
- **Harness**：循环之外的一切——提示词、工具、塑造模型行为的中间件。
- Harness 的职责：**在正确的时间把正确的上下文给模型**。

`create_agent` 是高度可配置的 harness，最简单形式：

```python
from langchain.agents import create_agent

agent = create_agent(model="openai:gpt-6.1-sol", tools=tools)
```

### 3.2 核心组件

#### Model（模型）

传模型标识字符串（`"provider:model"`）或已初始化的模型实例：

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-6.1-sol", temperature=0.5, timeout=300, max_tokens=25000)
agent = create_agent(model=model, tools=tools)
```

#### Tools（工具）

传任意 Python callable、LangChain tool 或 tool dict：

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_agent(model="openai:gpt-6.1-sol", tools=[search])
```

工具的 name、description、参数名会成为模型 prompt 的一部分，所以要写清楚。

#### System prompt（系统提示词）

塑造 Agent 处理任务的方式，接受字符串或 `SystemMessage`。动态 prompt 用 middleware。

```python
agent = create_agent(
    model="openai:gpt-6.1-sol",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

#### Structured output（结构化输出）

用 `response_format=` 返回校验过的 schema：

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class Answer(BaseModel):
    summary: str
    confidence: float

agent = create_agent(model="openai:gpt-6.1-sol", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

#### Agent state（Agent 状态）

每个 Agent 通过 `AgentState` 管理执行上下文，这是一个类型化字典，保存当前对话历史和自定义字段。

内置字段：

| 字段 | 类型 | 描述 |
|------|------|------|
| `messages` | `list[BaseMessage]` | 当前线程的完整对话历史，只增不改 |

添加自定义字段（如 `user_id`、计数器）：

```python
from langchain.agents import AgentState, create_agent

class MyState(AgentState):
    user_id: str
    call_count: int

agent = create_agent(model="openai:gpt-6.1-sol", tools=[], state_schema=MyState)
```

### 3.3 Invocation（调用）

用 `thread_id` 持久化并恢复对话历史：

```python
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-6.1-sol",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# 后续轮次复用同一 thread_id 以保留历史
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

### 3.4 配置 Harness（中间件 Middleware）

`create_agent` 通过中间件逐步增加能力，从 guardrails、重试到路由、自定义工具策略，只组合用例需要的部分。可配置项包括：

- **执行环境**（execution environment）
- **上下文管理**（context management）
- **规划与委派**（planning and delegation）
- **命名 Agent**（name your agent）
- **容错**（fault tolerance）
- **护栏**（guardrails）
- **转向**（steering）
- **中间件资源**（middleware resources）

### 3.5 核心优势

1. **标准模型接口**——用一套接口对接聊天模型、嵌入等，跨提供商切换只需最小代码改动，保持应用可移植。
2. **高度可配置的 harness**——从 `create_agent` 最小 harness 出发，通过中间件逐步加能力。
3. **构建于 LangGraph 之上**——LangChain 的 Agent 建立在 LangGraph 上，因此获得持久执行、人机交互、持久化等能力。
4. **用 LangSmith 调试**——在一个地方查看 trace、工具调用、状态转换和延迟。

---

## 四、LangGraph 快速入门

### 4.1 定位

LangGraph 是面向**长时间运行、有状态 Agent** 的低级编排框架和运行时。你把工作流定义为**共享状态、节点、边**组成的图，可以在同一张图中混合确定性步骤与 LLM 驱动步骤。

LangGraph 专注于编排能力，如持久执行、流式、人机交互。它**不规定** prompt 或 Agent 架构。

> 学习前先了解 Agent 的构建块，尤其是 **models** 和 **tools**。LangGraph 不强制要求使用 LangChain（虽然文档示例常配合使用）。

### 4.2 安装

```bash
pip install -U langgraph
# 或
uv add langgraph
```

创建最小图：

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

### 4.3 完整示例：计算器 Agent

一个可调用工具的完整 Agent，含 8 个步骤。

```python
# 1. 定义工具和模型
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-sonnet-4-6", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 2. 定义状态
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # add_messages 追加而非替换
    llm_calls: int

# 3. 定义模型节点
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM 决定是否调用工具"""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# 4. 定义工具节点
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """执行工具调用"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# 5. 定义结束逻辑
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否调用工具决定是继续还是停止"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# 6. 构建并编译 Agent
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()

# 7. （可选）设置 LangSmith 追踪
# export LANGSMITH_TRACING=true
# export LANGSMITH_API_KEY="your-langsmith-api-key"

# 8. 运行 Agent
from langchain.messages import HumanMessage

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

---

## 五、LangGraph 核心概念

### 5.1 Graph API：状态、节点、边

Graph API 把 Agent 工作流定义为 **state（状态）、nodes（节点）、edges（边）** 组成的图。`Send` 和 `Command` 控制图如何运行：`Send` 动态调用节点做 fan-out，`Command` 在更新状态的同时路由到下一个节点。

- **State（状态）**：应用的共享快照，通常是类型化 schema。
- **Nodes（节点）**：读取状态、做工作、返回更新的函数。
- **Edges（边）**：固定或条件转换，决定下一个节点。

一句话：**节点做工作，边决定接下来运行什么。**

#### Super-steps（超级步）

LangGraph 用消息传递（类似 Google Pregel）运行图，执行以离散的 super-step 推进：

- 并行运行的节点共享一个 super-step
- 串行运行的节点用不同的 super-step
- 节点在入边上收到输入时被激活
- super-step 后没有待处理消息的节点变为非激活
- 当所有节点都非激活且没有传输中的消息时，运行结束

#### StateGraph

`StateGraph` 是主要的图构建器，用你的状态 schema 参数化。

#### 编译图

```python
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    text: str

def node_a(state: State):
    return {"text": state["text"] + "a"}

graph = (
    StateGraph(State)
    .add_node("node_a", node_a)
    .add_edge(START, "node_a")
    .add_edge("node_a", END)
    .compile()
)

graph.invoke({"text": ""})  # {'text': 'a'}
```

编译会检查图结构（如孤立节点），也是挂载 checkpointers、breakpoints 等运行时选项的地方。**调用/流式前必须先编译。**

### 5.2 State（状态）与 Reducers（归约器）

状态是节点读写、更新的共享数据，存储在 **channels** 中（状态对象的键）。每个 channel 持有值并定义更新如何合并进来：

- 大多数 channel 用 last-write-wins（后写覆盖）
- 带 reducer 的 channel 用组合方式合并值（如追加到列表）

节点返回部分更新，LangGraph 通过每个 channel 的 reducer 应用这些更新。

用 `TypedDict`、`dataclass`（用于默认值）或 Pydantic `BaseModel`（需要校验时）定义状态。性能优先用 `TypedDict` 或 `dataclass`。

#### 默认 reducer

默认 reducer 用右值覆盖左值：

```python
# 从 {"foo": 1, "bar": ["hi"]} 开始
# 更新 {"foo": 2} → {"foo": 2, "bar": ["hi"]}
# 再更新 {"bar": ["bye"]} → {"foo": 2, "bar": ["bye"]}
```

#### 自定义 reducer

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], operator.add]  # 追加合并
```

#### 消息状态（MessagesState）

```python
from typing import Annotated
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

> 用 `add_messages` 而非 `operator.add`：后者只能追加，无法在人机交互编辑时按 ID 就地更新已有消息。

`MessagesState` 是预构建的状态（单 `messages` 键 + `add_messages`），可子类化加字段：

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
```

### 5.3 Nodes（节点）

节点是实现图中某一步的同步或异步函数，可接受：

1. `state`：图状态
2. `config`：`RunnableConfig`（含 `thread_id`、`tags` 等）
3. `runtime`：`Runtime` 对象（含运行时上下文、store、stream_writer 等）

```python
from dataclasses import dataclass
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    results: str

@dataclass
class Context:
    user_id: str

builder = StateGraph(State)

def node_with_runtime(state: State, runtime: Runtime[Context]):
    return {"results": f"Hello, {state['input']}!"}

builder.add_node("node_with_runtime", node_with_runtime)
```

#### 重执行与幂等性

使用 checkpointer 时，LangGraph 在 super-step 边界保存 checkpoint，而非节点函数内部。中断或重试后，受影响的节点**从头开始**重新运行。因此要设计节点逻辑使重执行安全，优先用幂等键、upsert、先读后写检查。

#### 选择工作单元

| 原语 | 有独立 checkpoint？ | 恢复/重试时 | 何时选用 |
|------|---------------------|-------------|----------|
| 节点内的普通函数 | 否 | 随节点一起重跑 | 步骤后无路由分叉，重复可接受 |
| Task | 否（结果随父 checkpoint 存） | 已完成任务返回存储结果 | 有副作用或昂贵，恢复时不能重复 |
| Subgraph | 是（有状态模式） | 在子图内恢复 | 有独立路由/状态 schema，或多父图复用 |
| Graph node | 是 | 从该节点开始恢复 | 步骤后可能有路由分叉，或需中断/时间旅行边界 |

### 5.4 Edges（边）

- **普通边（Normal edges）**：总是从节点 A 到 B
- **条件边（Conditional edges）**：调用函数决定下一个节点
- **入口（Entry point）**：用户输入后的第一个节点（`START` → 节点）
- **条件入口**：从 `START` 用函数路由

```python
graph.add_edge("node_a", "node_b")  # 普通边

graph.add_conditional_edges("node_a", routing_function)  # 条件边

graph.add_edge(START, "node_a")  # 入口
graph.add_conditional_edges(START, routing_function)  # 条件入口
```

> 每个节点要么用静态边，要么用动态路由（条件边/`Command`），不要混用，否则两条路径都可能运行。

### 5.5 Send（扇出 / map-reduce）

当需要"每个 item 运行一次同一节点、各自带独立输入、item 数量运行时才知道"时用 `Send`（map-reduce / orchestrator-worker 模式）：

```python
from langgraph.types import Send

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

### 5.6 Command（状态更新 + 路由合一）

当单一步骤既要更新状态、又要选择下一步时用 `Command`：

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: CommandState) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="my_other_node",
    )
```

`Command` 接收：
- `update`：状态更新（类似普通节点返回）
- `goto`：下一个节点（类似条件边）
- `graph`：从子图导航时的父图目标
- `resume`：中断后恢复用的值

三个使用位置：从节点返回、作为 `invoke`/`stream` 输入、从工具返回。

> 注意：`Command(resume=...)` 是唯一作为 `invoke()`/`stream()` 输入的 Command 模式。继续普通对话要传普通 dict，不要传 `Command(update=...)`。

---

## 六、LangGraph 关键能力

### 6.1 持久化（Persistence）

持久化层通过 **checkpointers**（短期记忆）和 **stores**（长期记忆）给 Agent 提供记忆能力。

- **Checkpointers**：把线程的图状态持久化为 checkpoint，用于短期、线程作用域的记忆（对话连续性、人机交互、时间旅行、容错）。
- **Stores**：跨线程的长期记忆。

```python
# 短期记忆：内存 checkpoint（生产环境用持久化 checkpointer）
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke({"messages": [...]}, config=config)
```

常见 checkpointers：`PostgresSaver`、`SqliteSaver`、`MemorySaver`（仅内存，重启不持久）。

### 6.2 人机交互（Human-in-the-loop / Interrupts）

中断允许你在图执行的特定点暂停，等待外部输入后再继续。触发中断时，LangGraph 用持久化层保存图状态，无限等待直到你恢复执行。

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # 暂停并请求审批
    approved = interrupt("Do you approve this action?")
    return {"approved": approved}
```

**恢复执行：**

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}
graph.stream_events({"input": "data"}, config=config, version="v3")

# 中断后，用 Command(resume=...) 恢复
resumed = graph.stream_events(Command(resume=True), config=config, version="v3")
```

**中断响应 schema（类型化输入校验）：**

```python
from pydantic import BaseModel

class Decision(BaseModel):
    approved: bool
    note: str | None = None

def node(state: State):
    answer = interrupt({"question": "approve?"}, response_schema=Decision)
    return {"answer": answer}
```

**常见模式：**
- 审批工作流（批准/拒绝）
- 处理多个中断（用中断 ID 映射 resume 值）
- 审阅与编辑状态
- 工具内中断（发送邮件前审批）
- 校验人工输入

**中断规则（重要）：**
1. 不要用 try/except 包裹 `interrupt` 调用（会捕获中断异常）
2. 不要重排节点内 `interrupt` 调用顺序（严格按索引匹配）
3. 不要返回复杂值（函数、类实例等不可序列化）
4. 中断前的副作用必须是幂等的

### 6.3 流式（Streaming）

LangGraph 图暴露 `stream`（同步）和 `astream`（异步）方法，以迭代器产出流式输出。传一个或多个 stream mode 控制收到的数据：

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
```

Stream modes：`updates`（图状态）、`values`（每步完整状态）、`messages`（LLM token）、`custom`（自定义数据）、`checkpoints`、`tasks`、`debug`。

> 新应用推荐用 **event streaming**（`graph.stream_events(..., version="v3")`），它为每个投影提供独立迭代器（messages、values、subgraphs、output），可独立消费。

### 6.4 记忆（Memory）

- **短期记忆**：线程作用域，通过 checkpointers 实现（见 6.1）。
- **长期记忆**：跨会话，通过 stores 实现。

### 6.5 部署与可观测

LangGraph 可以独立工作，也可配合其他 LangChain 产品：
- **LangChain**：构建于 LangGraph 之上的集成与 Agent 抽象
- **LangSmith Agent Server**：部署和扩展有状态 Agent，Studio 可视化原型
- **LangSmith Observability**：追踪请求、评估输出、监控部署

---

## 七、GitHub 生态仓库

LangChain 官方 GitHub 组织为 `langchain-ai`，核心仓库如下：

### 7.1 主要仓库

| 仓库 | 语言 | 说明 |
|------|------|------|
| `langchain-ai/langchain` | Python | 核心框架（约 134k stars），模型、工具、Agent 循环 |
| `langchain-ai/langchainjs` | JS/TS | LangChain 的 JavaScript 版本 |
| `langchain-ai/langgraph` | Python | 低级编排框架，构建有状态多角色 Agent |
| `langchain-ai/langgraphjs` | JS/TS | LangGraph 的 JavaScript 版本 |
| `langchain-ai/deepagents` | Python | "batteries-included" Agent 外壳 |
| `langchain-ai/deepagentsjs` | JS/TS | Deep Agents 的 JavaScript 版本 |
| `langchain-ai/langsmith-sdk` | - | LangSmith 平台 SDK |
| `langchain-ai/docs` | - | 官方文档仓库 |
| `langchain-ai/open-swe` | - | 开源异步编码 Agent 框架 |
| `langchain-ai/langchain-skills` | - | 提升 Agent 在 LangChain 生态任务上的表现 |

### 7.2 仓库结构（Monorepo）

**LangChain** 是 monorepo，含多个包：
- 核心包：`langchain`（chains/agents/retrieval 逻辑）、`langchain-core`（基础接口与核心抽象）
- Partner 包：独立版本，位于 `libs/providers/`（如 `langchain-openai`、`langchain-anthropic`、`langchain-google`）
- 支持包：`langchain-text-splitters`、`langchain-standard-tests`、`langchain-community`

**LangGraph** 是 monorepo，含多个 Python 包：
- 核心：`langgraph`（核心框架）、`langgraph-prebuilt`（高层 API）
- Checkpoint：`langgraph-checkpoint`（基础接口）、`langgraph-checkpoint-postgres`、`langgraph-checkpoint-sqlite`
- SDK/CLI：`langgraph-sdk`（Agent Server API 的 Python SDK）、`langgraph-cli`

**Deep Agents** 是 monorepo：
- 核心：`deepagents`、`deepagents-code`（交互式终端）、`deepagents-cli`（部署工具）
- 集成：`deepagents-evals`（评估套件）、`deepagents-acp`（Agent Client Protocol）

### 7.3 Deep Agents CLI

```bash
# 安装
uv tool install deepagents-cli

# 在项目目录启动
deepagents

# 管理多个 Agent
deepagents list
deepagents --agent backend-dev
deepagents reset backend-dev
```

特点：读写文件、带审批执行 shell 命令、联网搜索、发 HTTP 请求、跨会话记忆（存储在 `~/.deepagents/AGENT_NAME/memories/`）、可视化 todo 规划。

---

## 八、产品对比与选型

### 8.1 三者定位差异

| 产品 | 层级 | 最佳场景 |
|------|------|----------|
| **LangGraph** | Agent 运行时 | 需要精细控制、持久化、人机交互的生产系统；混合确定性步骤与 Agent 组件 |
| **LangChain** | Agent 框架 | 用核心 Agent 循环、从零构建所有 prompt/tools；快速交付 + 中间件定制 |
| **Deep Agents** | Agent 外壳 | 长期运行的自主 Agent，内置规划工具、文件系统、子 Agent、详细提示词 |

### 8.2 选型建议

- **快速原型 / 换模型**：选 LangChain（跨提供商、一行换模型）
- **需要精确控制状态和流程**：选 LangGraph
- **长期复杂任务（研究/编码）**：选 Deep Agents
- **可观测性 / 评估 / 部署**：配套 LangSmith（框架无关）

官方 2026 年对比结论：LangChain 适合跨模型快速原型，配 LangGraph 做有状态多 Agent 编排，配 Deep Agents 做长期工作流，配 LangSmith 做企业级可观测与评估。

---

## 九、历史演进时间线

| 时间 | 事件 |
|------|------|
| 2022-10-24 | LangChain 作为 Python 包发布（早于 ChatGPT 一个月），含 LLM 抽象 + Chains |
| 2022-12 | 加入第一个通用 Agent（用 LLM 生成 JSON 表示工具调用） |
| 2023-01 | 发布 JavaScript 版本 |
| 2023-02 | LangChain Inc. 成立 |
| 2023-03 | OpenAI 发布 function calling，LangChain 改用此方式做工具调用 |
| 2023-06 | LangSmith 发布（闭源平台，可观测 + 评估） |
| 2024-01 | LangChain 发布 0.1.0（首个非 0.0.x 版本，聚焦稳定性） |
| 2024-02 | LangGraph 作为开源库发布，补齐低级编排层 |
| 2024-06 | 集成超 700 个，拆分为独立 provider 包或 langchain-community |
| 2024-10 | LangGraph 成为构建多步 AI 应用的首选方式 |
| 2025-04 | 更新 message 格式以支持多模态输入 |
| 2025-10-20 | LangChain 发布 1.0，chains/agents 全面重构为单一 Agent 抽象（基于 LangGraph） |
| 2026-03-15 | Deep Agents 作为开源 Agent 外壳发布（v0.5.3） |

---

## 十、参考资源

### 官方文档
- LangChain 概览：https://docs.langchain.com/oss/python/langchain/overview
- LangChain 安装：https://docs.langchain.com/oss/python/langchain/install
- LangChain Quickstart：https://docs.langchain.com/oss/python/langchain/quickstart
- LangChain Agents：https://docs.langchain.com/oss/python/langchain/agents
- LangGraph 概览：https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph Quickstart：https://docs.langchain.com/oss/python/langgraph/quickstart
- LangGraph Graph API：https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph 持久化：https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph 中断：https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph 流式：https://docs.langchain.com/oss/python/langgraph/streaming

### GitHub
- 组织：https://github.com/langchain-ai
- LangChain：https://github.com/langchain-ai/langchain
- LangGraph：https://github.com/langchain-ai/langgraph
- Deep Agents：https://github.com/langchain-ai/deepagents

### 其他
- LangChain 理念与历史：https://docs.langchain.com/oss/python/langchain/philosophy
- AI Agent 框架对比：https://www.langchain.com/resources/ai-agent-frameworks

---

> **文档说明**：本文档基于官方文档与公开资料整理，代码示例为官方文档原样呈现（含未来版本号的模型名，实际使用请以官方最新文档为准）。建议配合官方文档和 GitHub 仓库同步学习。

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
