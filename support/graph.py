from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, List
from .nodes import text_to_sql, rag_query
from langchain_openai import ChatOpenAI
from .rdb_conn import sql_query
from dotenv import load_dotenv
from .logger import Logging
import json
import os

load_dotenv()
Logging.setLevel()

class QueryState(TypedDict):
    """State for the query routing graph."""
    query: str
    conversation_history: List[BaseMessage]
    standalone_query: str
    route: str
    reasoning: str
    # NEW: 'reviews' or 'spec'
    content_type: str  
    content_reasoning: str
    sql_result: str
    rag_result: str
    final_answer: str
    error: str


def get_prompt(name: str) -> str:
    try:
        prompt_path = os.path.join("templates", f"{name}.txt")
        system_prompt = open(prompt_path, "r").read()
        return system_prompt
    except Exception as e:
        Logging.logError(str(e))
        raise e


def preprocessing_node(state: QueryState) -> QueryState:
    try:
        query = state["query"]
        history = state.get("conversation_history", [])
        Logging.logInfo("Executing Preprocessing Node")
        
        # If no history, query is already standalone
        if not history or len(history) == 0:
            Logging.logDebug("No history, query is already standalone")
            return {
                **state,
                "standalone_query": query
            }
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)        
        system_prompt = get_prompt(name="preprocess")

        # Format conversation history
        history_text = ""
        for i, msg in enumerate(history[-6:]):  # Last 3 exchanges (6 messages)
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"
        
        user_prompt = f"""
        Conversation History:
        {history_text}

        Current Query: {query}

        Reformulate the current query into a standalone query:
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        standalone_query = response.content.strip()
        
        Logging.logDebug(f"Original: '{query}'")
        Logging.logDebug(f"Stndalone: '{standalone_query}'\n")
        
        return {
            **state,
            "standalone_query": standalone_query
        }
        
    except Exception as e:
        Logging.logError(str(e))
        return {
            **state,
            "standalone_query": query,
            "error": str(e)
        }


def router_node(state: QueryState) -> QueryState:
    try:
        Logging.logInfo("Executing Router Node")
        query = state.get("standalone_query", state["query"])
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        system_prompt = get_prompt(name="router")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        response = llm.invoke(messages)
        decision = json.loads(response.content)
        
        route = decision.get("route", "text2sql").lower()
        if route not in ["text2sql", "rag"]:
            route = "text2sql"
        
        Logging.logDebug(f"Router Decision: {route.upper()}")
        Logging.logDebug(f"Reasoning: {decision.get('reasoning', '')}\n")
        
        return {
            **state,
            "route": route,
            "reasoning": decision.get("reasoning", "")
        }
        
    except Exception as e:
        Logging.logError(str(e))
        return {
            **state,
            "route": "text2sql",
            "reasoning": f"Error in routing: {str(e)}",
            "error": str(e)
        }


def text2sql_node(state: QueryState) -> QueryState:
    try:
        Logging.logInfo("Executing Text2SQL Node")
        query = state.get("standalone_query", state["query"])

        sql = text_to_sql(query)
        result = sql_query(sql)
        return {
            **state,
            "sql_result": sql,
            "final_answer": result
        }
    except Exception as e:
        Logging.logError(str(e))
        return {
            **state,
            "sql_result": "",
            "final_answer": str(e),
            "error": str(e)
        }


def content_type_node(state: QueryState) -> QueryState:
    """
    Router that determines whether to query reviews or references.
    Output is used as metadata filter for ChromaDB.
    """
    try:
        Logging.logInfo("Executing Content Type Router Node")
        query = state.get("standalone_query", state["query"])
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        system_prompt = get_prompt(name="content_router")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        response = llm.invoke(messages)
        decision = json.loads(response.content)
        
        content_type = decision.get("content_type", "reviews").lower()
        if content_type not in ["reviews", "spec"]:
            content_type = "spec"
        
        Logging.logDebug(f"Content Type Decision: {content_type.upper()}")
        Logging.logDebug(f"Reasoning: {decision.get('reasoning', '')}\n")
        
        return {
            **state,
            "content_type": content_type,
            "content_reasoning": decision.get("reasoning", "")
        }
        
    except Exception as e:
        Logging.logError(str(e))
        return {
            **state,
            "content_type": "spec",
            "content_reasoning": f"Error in content routing: {str(e)}",
            "error": str(e)
        }
    

def rag_node(state: QueryState) -> QueryState:
    try:
        Logging.logInfo("Executing RAG Node")
        query = state.get("standalone_query", state["query"])
        content_type = state.get("content_type", None)
        result = rag_query(query, content_type)
        return {
            **state,
            "rag_result": result,
            "final_answer": result
        }
    except Exception as e:
        Logging.logError(str(e))
        return {
            **state,
            "sql_result": "",
            "final_answer": str(e),
            "error": str(e)
        }


def post_process_node(state: QueryState) -> QueryState:
    try:
        Logging.logInfo("Executing Post-Processing Node")
        original_query = state["query"]
        standalone_query = state.get("standalone_query", original_query)
        route = state["route"]
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )

        # Prepare context based on route
        if route == "text2sql":
            raw_output = state.get("final_answer", "")
            sql_result = state.get("sql_result", "")
            system_prompt = get_prompt(name="postprocess_sql")

            user_prompt = f"""
            Original Query: {standalone_query}
            Query Used: {sql_result}
            Database Results:
            {raw_output}

            Please provide a natural language response to the user's query based on these results.
            """

        else:
            raw_output = state.get("final_answer", "")
            system_prompt = get_prompt(name="postprocess_rag")

            user_prompt = f"""
            Original Query: {standalone_query}

            Retrieved Information:
            {raw_output}

            Please provide a clear, natural language explanation to answer the user's query.
            """
    
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        final_answer = response.content
        
        Logging.logDebug(f"Post-processing output: {final_answer}")

        # Update conversation history
        history = state.get("conversation_history", [])
        updated_history = history + [
            HumanMessage(content=original_query),
            AIMessage(content=final_answer)
        ]
        return {
            **state,
            "final_answer": final_answer,
            "conversation_history": updated_history
        }
        
    except Exception as e:
        Logging.logError(str(e))
        
        # Fallback to raw output
        fallback = state.get("sql_result") or state.get("rag_result") or "No results available"

        # Still update history even on error
        history = state.get("conversation_history", [])
        updated_history = history + [
            HumanMessage(content=original_query),
            AIMessage(content=fallback)
        ]
        
        return {
            **state,
            "final_answer": fallback,
            "conversation_history": updated_history,
            "error": str(e)
        }


def route_query(state: QueryState) -> Literal["text2sql", "rag"]:
    """
    Conditional edge that determines which node to execute next.
    """
    return state["route"]


def create_query_graph() -> StateGraph:
    try:
        Logging.logInfo("Creating the LangGraph")

        workflow = StateGraph(QueryState)
        workflow.add_node("preprocessing", preprocessing_node)
        workflow.add_node("router", router_node)
        workflow.add_node("content_router", content_type_node)
        workflow.add_node("text2sql", text2sql_node)
        workflow.add_node("rag", rag_node)
        workflow.add_node("post_processing", post_process_node)
        workflow.set_entry_point("preprocessing")
        workflow.add_edge("preprocessing", "router")
        workflow.add_conditional_edges(
            "router",
            route_query,
            {
                "text2sql": "text2sql",
                "rag": "content_router"
            }
        )
        workflow.add_edge("content_router", "rag")
        workflow.add_edge("text2sql", "post_processing")
        workflow.add_edge("rag", "post_processing")
        workflow.add_edge("post_processing", END)
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    except Exception as e:
        Logging.logError(str(e))
        raise e


class ProdLensQueryEngine:
    def __init__(self) -> None:
        """Initialize the query engine with compiled graph."""
        self.graph = create_query_graph()
        self.thread_id = "default_conversation"
    

    def query(self, user_query: str, thread_id: str = None) -> dict:
        try:
            if thread_id is None:
                thread_id = self.thread_id
        
            Logging.logInfo("=" * 70)
            Logging.logInfo(f"Processing Query: '{user_query}'")
            Logging.logDebug(f"Thread ID: {thread_id}")
            Logging.logInfo("=" * 70 + "\n")
            
            config = {"configurable": {"thread_id": thread_id}}

            # Retrieve previous conversation history from checkpointer
            try:
                previous_state = self.graph.get_state(config)
                previous_history = previous_state.values.get("conversation_history", [])
            except:
                previous_history = []

            # Initialize state
            initial_state = {
                "query": user_query,
                "conversation_history": previous_history,
                "standalone_query": "",
                "route": "",
                "reasoning": "",
                "content_type": "",
                "content_reasoning": "",
                "sql_result": "",
                "rag_result": "",
                "final_answer": "",
                "error": ""
            }
            
            final_state = self.graph.invoke(initial_state, config)
            
            Logging.logDebug("=" * 70)
            Logging.logDebug("EXECUTION COMPLETE")
            Logging.logDebug("=" * 70 + "\n")
            
            return final_state
        except Exception as e:
                Logging.logError(str(e))
                raise e


    def get_conversation_history(self, thread_id: str = None) -> List[BaseMessage]:
        try:
            if thread_id is None:
                thread_id = self.thread_id
            
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            
            return state.values.get("conversation_history", [])
        except Exception as e:
            Logging.logError(str(e))
            raise e


    def clear_history(self, thread_id: str = None):
        if thread_id is None:
            thread_id = self.thread_id
        
        Logging.logInfo(f"Cleared conversation history for thread: {thread_id}")


    def new_conversation(self, thread_id: str = None) -> str:
        import uuid
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        self.thread_id = thread_id
        Logging.logInfo(f"Started new conversation with thread ID: {thread_id}")
        return thread_id


    def visualize(self, output_path: str = "query_graph.png", save: bool = False):
        try:
            graph_png = self.graph.get_graph().draw_mermaid_png()
            if save:
                with open(output_path, "wb") as f:
                    f.write(graph_png)
            
                Logging.logDebug(f"Graph visualization saved to {output_path}")                
        except Exception as e:
            Logging.logError(str(e))
            raise e


if __name__ == "__main__":
    engine = ProdLensQueryEngine()

    result = engine.query("What kind of monitors should buy for gaming purposes?")
    print(f"Answer:\n{result['final_answer']}")
    print(f"Standalone Query: {result['standalone_query']}\n")

    result = engine.query("What about SteelSeries ones?")
    print(f"Answer:\n{result['final_answer']}\n")
    print(f"Standalone Query: {result['standalone_query']}\n")
    
    result = engine.query("Tell me more about it")
    print(f"Answer:\n{result['final_answer']}\n")
    print(f"Standalone Query: {result['standalone_query']}\n")

    engine.visualize()