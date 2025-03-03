import base64
from dotenv import load_dotenv
import functools
import gradio as gr
from gradio import ChatMessage
import json
import operator
import os
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
import requests
from typing import Annotated, List, Tuple, Union, Literal, Sequence
from typing_extensions import TypedDict
import uuid

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.embeddings.base import Embeddings
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from langfuse.callback import CallbackHandler

from tools import tavily_tool, get_powersizer_api_tool, retrieve_docs_powerstore_tool, retrieve_docs_connectrix_tool
from utils import CustomLogger


load_dotenv()

_printed = set()  # track state msg ids
_current_agent = "supervisor"
_is_first_question = True


HOME_PATH = os.getenv('HOME_PROD')
PROJECT_PATH = os.getenv('PROJ_HOME_PROD')

logger = CustomLogger(f"{HOME_PATH}/location_of_.log")
logger.log_info(f"Status || Start logging!")

try:
    langfuse_handler = CallbackHandler(
        public_key=<PUBLIC KEY>,
        secret_key=<SECRET_KEY>,
        host=HOST_ADDRESS,
        session_id=SESSION_NAME
    )
except:
    langfuse_handler=None
    logger.log_warning(f"Langfuse loading error!")


def create_llm():
    MODEL_ID = MODEL_ID
    VLLM_API_URL = VLLM_API
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url=VLLM_API_URL,
        api_key="EMPTY",
        max_tokens=1024,
        temperature=0.1
    )
    return llm


class AgentState(TypedDict):
    # Add the Annotated type for messages to enable proper message handling
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str


def create_agent_state_prompts():
    members = ["Researcher", "PowerStore_Expert", "Connectrix_Expert", "Sizer_Expert"]
    system_prompt = (
        # "You are a helpful assistant tasked with answering user questions. "
        "You are a supervisor tasked with managing a conversation between the following workers: {members}. "
        "\nGiven the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. "
        "\nFor questions related to the PowerStore, use the 'PowerStore_Expert' to get information from a vector store. "
        "\nFor questions related to the Connectrix switches, use the 'Connectrix_Expert' to get information from a vector store. "
        "\nFor questions related to sizing server hardware for various AI workloads, use the 'Sizer_Expert' to get data from the PowerSizer API. "
        "\nFor any general questions, use the 'Researcher' to get information from the web."
        "\nWhen finished, respond with FINISH." 
    )
    options = ["FINISH"] + members

    class RouteResponse(BaseModel):
        next: Literal[*options]

    global_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    return RouteResponse, global_prompt



def create_agent_graph(RouteResponse, global_prompt, llm):
    """Create the agent graph without memory"""
    # Construct Graph
    def agent_node(state, agent, name):
        """Create the nodes in the graph - it takes care of converting the agent response to a human message.
         Add it the global state of the graph"""
        try:
            result = agent.invoke(state, config={"callbacks": [langfuse_handler]})
        except:
            result = agent.invoke(state)
        logger.log_info(f"agent_node_state || {state}")
        logger.log_info(f"agent_node_result || {result}")
        return result

    def supervisor_agent(state):
        supervisor_chain = global_prompt | llm.with_structured_output(RouteResponse)
        try:
            output = supervisor_chain.invoke(state, config={"callbacks": [langfuse_handler]})
        except:
            output = supervisor_chain.invoke(state)
        return output

    # Create react agents and add tools
    sys_prompt_research=("You are a helpful and conversational assistant for a Pre-sales team, \
answer the user question directly, you may use the web tool to search for more information if necessary.")

    research_agent = create_react_agent(llm, tools=[tavily_tool], state_modifier=sys_prompt_research)
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

    powerstore_agent = create_react_agent(llm, tools=[retrieve_docs_powerstore_tool])
    powerstore_node = functools.partial(agent_node, agent=powerstore_agent, name="PowerStore_Expert")

    connectrix_agent = create_react_agent(llm, tools=[retrieve_docs_connectrix_tool])
    connectrix_node = functools.partial(agent_node, agent=connectrix_agent, name="Connectrix_Expert")

    powersizer_agent = create_react_agent(llm, tools=[get_powersizer_api_tool])
    powersizer_node = functools.partial(agent_node, agent=powersizer_agent, name="Sizer_Expert")

    workflow = StateGraph(AgentState)
    # workflow.add_node("Chat", chat_node)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("PowerStore_Expert", powerstore_node)
    workflow.add_node("Connectrix_Expert", connectrix_node)
    workflow.add_node("Sizer_Expert", powersizer_node)
    workflow.add_node("supervisor", supervisor_agent)

    members = ["Researcher", "PowerStore_Expert", "Connectrix_Expert", "Sizer_Expert"]
    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    workflow.set_entry_point("supervisor")

    return workflow


def initialize_graph_with_memory(workflow):
    """Initialize the graph with a new memory instance"""
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph, memory


def _print_event(event: dict, _printed: set, max_length=5000):
    messages = event.get("messages")
    next_step = event.get("next")
    logger.log_info(f"next agent: {next_step}")
    if messages:
        for msg in messages:
            if msg.id not in _printed:
                msg_repr = msg.pretty_repr(html=True)
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                logger.log_info(f"{msg_repr}")
                _printed.add(msg.id)
                logger.log_info(f"completed msg ids: {_printed}")


def obtaining_chatbot1_msg(user_message, history):
    history.append(ChatMessage(role="user", content=user_message))
    yield "", history


# TODO !
# ref: https://huggingface.co/spaces/gradio/langchain-agent/tree/main
async def interact_with_langchain_agent(messages: list, max_length=5000):
    global _current_agent, _is_first_question
    _last_agent = None
    prompt = messages[-1]['content']

    initial_state = {
        "messages": [HumanMessage(content=prompt, id=f"init_msg_{uuid.uuid4()}")],
        "next": ""
    }

    # Keep track of the last valid diagram
    last_valid_diagram = None
    try:
        last_valid_diagram = Image.open("images/current_architecture.png")
    except:
        last_valid_diagram = Image.open("images/graph.png")

    # Handle the initial diagram state differently for first question
    if _is_first_question:
        # First show start node
        _current_agent = "__start__"
        try:
            current_diagram = update_architecture_diagram("__start__")
            last_valid_diagram = current_diagram
            yield messages, current_diagram

            # Then transition to supervisor before processing begins
            _current_agent = "supervisor"
            current_diagram = update_architecture_diagram("supervisor")
            last_valid_diagram = current_diagram
            messages.append(ChatMessage(role="assistant", content="Next Agent: supervisor"))
            yield messages, current_diagram
        except Exception as e:
            print(f"Error updating initial diagram: {e}")

        _is_first_question = False  # Mark that we're no longer on first question
    else:
        # For subsequent questions, start with supervisor
        _current_agent = "supervisor"
        try:
            current_diagram = update_architecture_diagram("supervisor")
            last_valid_diagram = current_diagram
            yield messages, current_diagram
        except Exception as e:
            print(f"Error updating initial diagram: {e}")

    async for chunk in global_graph.astream(initial_state, config, stream_mode="values"):
        logger.log_info(f"graph stream chunk || {chunk}")

        msgs = chunk.get("messages")
        next_step = chunk.get("next")

        # Update current agent and diagram when agent changes
        if next_step and (next_step != _last_agent):
            if next_step == "FINISH":
                # When finishing, highlight the end node
                _current_agent = "FINISH"
            else:
                _current_agent = next_step

            try:
                # Generate diagram with appropriate highlighting
                current_diagram = update_architecture_diagram(_current_agent)
                last_valid_diagram = current_diagram
            except Exception as e:
                print(f"Error updating diagram: {e}")
                current_diagram = last_valid_diagram

            messages.append(ChatMessage(role="assistant", content=f"Next Agent: {next_step}"))
            yield messages, current_diagram
            _last_agent = next_step

        if msgs:
            current_diagram = last_valid_diagram
            for m in msgs:
                if m.id not in _printed:
                    msg_repr = m.pretty_repr(html=True)
                    if len(msg_repr) > max_length:
                        msg_repr = msg_repr[:max_length] + " ... (truncated)"
                    logger.log_info(f"{msg_repr}")

                    if "Tool Calls" in msg_repr:
                        messages.append(ChatMessage(
                            role="assistant",
                            content=msg_repr,
                            metadata={"title": f"🛠️ Used tool {m.tool_calls[0]['name']}"}
                        ))
                        yield messages, current_diagram
                    elif "Tool Message" in msg_repr:
                        messages.append(ChatMessage(
                            role="assistant",
                            content=msg_repr,
                            metadata={"title": f"🛠️ Tool Message & Response [{m.tool_call_id}]"}
                        ))
                        yield messages, current_diagram
                    elif m.content != prompt:
                        messages.append(ChatMessage(role="assistant", content=m.content))
                        yield messages, current_diagram

                    _printed.add(m.id)
                    logger.log_info(f"completed msg ids: {_printed}")


def clear_all_chat_history():
    """Clear all chat history by recreating the graph with new memory"""
    global global_graph, _printed, workflow, _current_agent, _is_first_question
    _current_agent = "supervisor"
    _is_first_question = True  # Reset the first question flag

    current_state = global_graph.get_state(config)
    if current_state and "messages" in current_state.values:
        messages = current_state.values["messages"]
        global_graph.update_state(config, {
            "messages": [RemoveMessage(id=m.id) for m in messages]
        })

    global_graph, _ = initialize_graph_with_memory(workflow)

    # Reset printed message tracking
    _printed = set()

    logger.log_info("Successfully reset chat memory and history")

    # Generate fresh initial diagram
    try:
        # Generate a new diagram with no highlighting for fresh start
        diagram_path = style_langgraph_pipeline('__start__')
        initial_diagram = Image.open(diagram_path)
    except Exception as e:
        logger.log_info(f"Error generating initial diagram: {e}")
        try:
            initial_diagram = Image.open("images/current_architecture.png")
        except:
            initial_diagram = None

    return [], "", initial_diagram


def style_langgraph_pipeline(node_to_style='supervisor', is_finished=False):
    """Generate a styled Mermaid diagram based on the current active agent and finished state"""
    # Define color mappings
    default_color = '#f2f0ff'
    active_color = '#90EE90'  # Light green
    end_color = '#bfb6fc'     # Purple for end node

    # Valid nodes from the graph structure
    valid_nodes = {
        '__start__',
        '__end__',
        'supervisor',
        'Researcher',
        'PowerStore_Expert',
        'Connectrix_Expert',
        'Sizer_Expert'
    }

    # Handle FINISH state specially
    if node_to_style == 'FINISH':
        normalized_node = 'FINISH'
    # For other nodes, try exact match first
    elif node_to_style in valid_nodes:
        normalized_node = node_to_style
    # If no exact match, try case-insensitive match
    else:
        node_lower = node_to_style.lower()
        for valid_node in valid_nodes:
            if valid_node.lower() == node_lower:
                normalized_node = valid_node
                break
        else:
            normalized_node = node_to_style  # fallback to original if no match

    # Create Mermaid string with dynamic styling
    mermaid_str = """%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([<p>__start__</p>]):::first
    Researcher(Researcher):::researcher
    PowerStore_Expert(PowerStore_Expert):::powerstore_expert
    Connectrix_Expert(Connectrix_Expert):::connectrix_expert
    Sizer_Expert(Sizer_Expert):::sizer_expert
    supervisor(supervisor):::supervisor
    __end__([<p>__end__</p>]):::last
    
    Connectrix_Expert --> supervisor;
    PowerStore_Expert --> supervisor;
    Researcher --> supervisor;
    Sizer_Expert --> supervisor;
    __start__ --> supervisor;
    supervisor -.-> Researcher;
    supervisor -.-> PowerStore_Expert;
    supervisor -.-> Connectrix_Expert;
    supervisor -.-> Sizer_Expert;
    supervisor -. FINISH .-> __end__;"""

    # If conversation is finished, highlight end node and remove highlighting from others
    if is_finished:
        mermaid_str += f"\nclassDef last fill:{active_color},stroke:#333,stroke-width:2px"
        # Set all other nodes to default color
        mermaid_str += f"\nclassDef first fill-opacity:0,stroke:#333"
        agents = ['supervisor', 'Researcher', 'PowerStore_Expert', 'Connectrix_Expert', 'Sizer_Expert']
        for agent in agents:
            mermaid_str += f"\nclassDef {agent.lower()} fill:{default_color},stroke:#333"
    elif normalized_node == '__start__':
        # Special handling for start node
        mermaid_str += f"\nclassDef first fill:{active_color},stroke:#333,stroke-width:2px"
        mermaid_str += f"\nclassDef last fill:{end_color}"
        agents = ['supervisor', 'Researcher', 'PowerStore_Expert', 'Connectrix_Expert', 'Sizer_Expert']
        for agent in agents:
            mermaid_str += f"\nclassDef {agent.lower()} fill:{default_color},stroke:#333"
    else:
        # Normal state - highlight active agent
        mermaid_str += f"\nclassDef first fill-opacity:0"
        mermaid_str += f"\nclassDef last fill:{end_color}"
        agents = {
            'supervisor': 'supervisor',
            'Researcher': 'researcher',
            'PowerStore_Expert': 'powerstore_expert',
            'Connectrix_Expert': 'connectrix_expert',
            'Sizer_Expert': 'sizer_expert'
        }

        for agent_name, class_name in agents.items():
            if agent_name == normalized_node:
                mermaid_str += f"\nclassDef {class_name} fill:{active_color},stroke:#333,stroke-width:2px"
            else:
                mermaid_str += f"\nclassDef {class_name} fill:{default_color},stroke:#333"

    # Generate and save the image
    graphbytes = mermaid_str.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")

    url = f"https://mermaid.ink/img/{base64_string}"
    response = requests.get(url)

    if response.status_code == 200:
        output_path = "images/current_architecture.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    else:
        return "images/graph.png"  # Fallback to static image


def update_architecture_diagram(new_agent=None):
    """Update the architecture diagram based on current agent"""
    global _current_agent

    # If no new agent is specified, use current agent
    agent_to_use = new_agent if new_agent is not None else _current_agent

    # Check if we're in a finished state
    is_finished = (agent_to_use == "FINISH")

    try:
        diagram_path = style_langgraph_pipeline(agent_to_use, is_finished)
        return Image.open(diagram_path)
    except Exception as e:
        print(f"Error updating diagram: {e}")
        # Return the last successfully generated diagram or fallback
        try:
            return Image.open("images/current_architecture.png")
        except:
            return Image.open("images/graph.png")


css = """
footer {visibility: hidden}

"""


def main_demo():
    global global_graph, config, workflow, _current_agent

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": 8}

    llm = create_llm()
    RouteResponse, global_prompt = create_agent_state_prompts()
    workflow = create_agent_graph(RouteResponse, global_prompt, llm)
    global_graph, memory = initialize_graph_with_memory(workflow)

    initial_diagram = update_architecture_diagram("__start__")

    with gr.Blocks(theme = gr.themes.Default(text_size="lg"), css = css) as agent_chat_interface:
        with gr.Row():
            with gr.Column(scale=15):
                gr.Markdown("# Chat with a LangChain Agent 🕵️‍♂️ and see its thoughts 💭")
            with gr.Column(scale=1, min_width=200):
                toggle_dark = gr.Button(value="💡🌙")

        with gr.Tab("Chatbot"):
            with gr.Accordion("Agentic Architecture Summary", open=False):
                with gr.Row():
                    gr.Markdown("""- A supervisor agent orchestrates the workflow by coordinating with four specialized expert agents (Researcher, PowerStore, Connectrix, and Sizer)\n
- The system leverages a LangChain-based graph architecture where each expert agent is equipped with specific tools (like tavily_tool for Research, powersizer_api_tool for Sizing) to perform specialized tasks.\n
- The framework implements state management and memory tracking capabilities using MemorySaver, allowing for persistent conversation history and checkpointing between agent interactions                
                """)
            with gr.Row():
                architecture_diagram = gr.Image(
                    label="Current Agent Architecture",
                    value=initial_diagram,
                )
            with gr.Row():
                chatbot_2 = gr.Chatbot(
                    type="messages",
                    label="LLM Agent",
                    avatar_images=["images/smiling.png", "images/robot_emoji.png"],
                    height=800
                )
            with gr.Row():
                input_2 = gr.Textbox(lines=1, label="Chat Message")
            with gr.Row():
                with gr.Column():
                    chat_btn = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    clear = gr.ClearButton(value="Clear Chat & History", components=[input_2, chatbot_2])
            with gr.Row():
                gr.Examples(label="Example Agent Tasks",
                            examples=[EXAMPLE_QUESTIONS],
                            inputs=input_2)

        ###################################################################################
        ################################ Dark mode toggle #################################
        ###################################################################################
        toggle_dark.click(
            None,
            js="""
            () => {
                document.body.classList.toggle('dark');
                document.body.classList.toggle('vsc-initialized dark');
                document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
            }
            """, )
        ###################################################################################
        ##################################### Chatbot #####################################
        ###################################################################################
        input_2.submit(fn=obtaining_chatbot1_msg, inputs=[input_2, chatbot_2], outputs=[input_2, chatbot_2], queue=False) \
                .then(interact_with_langchain_agent, [chatbot_2], [chatbot_2, architecture_diagram],
                concurrency_limit=8, show_progress="minimal")  # num of worker threads
        chat_btn.click(fn=obtaining_chatbot1_msg, inputs=[input_2, chatbot_2], outputs=[input_2, chatbot_2], queue=False) \
                .then(interact_with_langchain_agent, [chatbot_2], [chatbot_2, architecture_diagram],
                concurrency_limit=8, show_progress="minimal")
        clear.click(fn=clear_all_chat_history, outputs=[chatbot_2, input_2, architecture_diagram])

    return agent_chat_interface


if __name__ == "__main__":
    demo = main_demo()
    demo.launch(server_name="0.0.0.0", share=False)