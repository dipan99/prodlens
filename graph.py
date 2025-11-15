from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from nodes import text_to_sql, rag_query
from rdb_conn import sql_query
from dotenv import load_dotenv
import json

load_dotenv()

class QueryState(TypedDict):
    """State for the query routing graph."""
    query: str
    route: str
    reasoning: str
    sql_result: str
    rag_result: str
    final_answer: str
    error: str


def router_node(state: QueryState) -> QueryState:
    """
    Decision node that routes queries to Text2SQL or RAG.
    
    Uses LLM to analyze query intent and decide routing.
    """
    query = state["query"]
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    system_prompt = """You are a query routing assistant for an electronics database system.

    The system has two capabilities:

    1. **TEXT2SQL**: Queries structured product data (specifications, prices, ratings, comparisons)
    - Examples: "Show me monitors under $500", "What's the best gaming mouse?", "Compare keyboards"
    - Use when: Query needs filtering, aggregation, sorting, or comparing specific product attributes

    2. **RAG**: Searches unstructured text for definitions and explanations
    - Examples: "What is response time?", "Why is DPI important?", "Explain refresh rate"
    - Use when: Query asks for definitions, meanings, importance, or explanations of specifications

    Respond with JSON:
    {
        "route": "text2sql" | "rag",
        "reasoning": "Brief explanation"
    }

    Default to TEXT2SQL if unclear."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    try:
        response = llm.invoke(messages)
        decision = json.loads(response.content)
        
        route = decision.get("route", "text2sql").lower()
        if route not in ["text2sql", "rag"]:
            route = "text2sql"
        
        print(f"Router Decision: {route.upper()}")
        print(f"Reasoning: {decision.get('reasoning', '')}\n")
        
        return {
            **state,
            "route": route,
            "reasoning": decision.get("reasoning", "")
        }
        
    except Exception as e:
        print(f"Router error: {e}, defaulting to TEXT2SQL")
        return {
            **state,
            "route": "text2sql",
            "reasoning": f"Error in routing: {str(e)}",
            "error": str(e)
        }


def text2sql_node(state: QueryState) -> QueryState:
    query = state["query"]
    print("Executing Text2SQL Node...")
    try:
        sql = text_to_sql(query)
        print(sql)
        result = sql_query(sql)
        print(result)
        return {
            **state,
            "sql_result": sql,
            "final_answer": result
        }
    except Exception as e:
        error_msg = f"Error in Text2SQL: {str(e)}"
        print(f"{error_msg}")
        return {
            **state,
            "sql_result": "",
            "final_answer": error_msg,
            "error": error_msg
        }


def rag_node(state: QueryState) -> QueryState:
    query = state["query"]
    print("Executing RAG Node...")
    try:
        result = rag_query(query)
        return {
            **state,
            "rag_result": result,
            "final_answer": result
        }
    except Exception as e:
        error_msg = f"Error in RAG: {str(e)}"
        print(f"{error_msg}")
        return {
            **state,
            "rag_result": "",
            "final_answer": error_msg,
            "error": error_msg
        }


def post_process_node(state: QueryState) -> QueryState:
    query = state["query"]
    route = state["route"]
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    # Prepare context based on route
    if route == "text2sql":
        raw_output = state.get("sql_result", "")
        system_prompt = """
        You are a helpful assistant that presents database query results in natural language.

        Your task:
        1. Take the SQL query results and present them in a clear, conversational way
        2. Use proper formatting (bullet points, tables if needed)
        3. Highlight key insights from the data
        4. Be concise but informative
        5. If there are no results, explain that politely

        Do not make up information - only use what's in the results.
        """

        user_prompt = f"""
        Original Query: {query}

        Database Results:
        {raw_output}

        Please provide a natural language response to the user's query based on these results.
        """

    else:  # route == "rag"
        raw_output = state.get("rag_result", "")
        system_prompt = """
        You are a helpful assistant that explains technical concepts about electronics.

        Your task:
        1. Take the retrieved information and explain it clearly
        2. Use examples where appropriate
        3. Keep explanations accessible to non-experts
        4. Maintain accuracy - don't add information beyond what's provided
        5. Structure the response with clear paragraphs
        6. Keep your response concise

        Do not make up information - only use the retrieved context."""

        user_prompt = f"""
        Original Query: {query}

        Retrieved Information:
        {raw_output}

        Please provide a clear, natural language explanation to answer the user's query.
        """
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        final_answer = response.content
        
        print(f"✅ Post-processing successful\n")
        
        return {
            **state,
            "final_answer": final_answer
        }
        
    except Exception as e:
        error_msg = f"Error in post-processing: {str(e)}"
        print(error_msg)
        
        # Fallback to raw output
        fallback = state.get("sql_result") or state.get("rag_result") or "No results available"
        return {
            **state,
            "final_answer": fallback,
            "error": error_msg
        }


def route_query(state: QueryState) -> Literal["text2sql", "rag"]:
    """
    Conditional edge that determines which node to execute next.
    """
    return state["route"]


def create_query_graph() -> StateGraph:
    workflow = StateGraph(QueryState)
    workflow.add_node("router", router_node)
    workflow.add_node("text2sql", text2sql_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("post_processing", post_process_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "text2sql": "text2sql",
            "rag": "rag"
        }
    )
    workflow.add_edge("text2sql", "post_processing")
    workflow.add_edge("rag", "post_processing")
    workflow.add_edge("post_processing", END)
    return workflow.compile()


class ProdLensQueryEngine:
    """
    Query engine using LangGraph for routing between Text2SQL and RAG.
    """
    
    def __init__(self):
        """Initialize the query engine with compiled graph."""
        self.graph = create_query_graph()
    
    def query(self, user_query: str) -> dict:
        """
        Execute a query through the graph.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Final state with results
        """
        print("=" * 70)
        print(f"Processing Query: '{user_query}'")
        print("=" * 70 + "\n")
        
        # Initialize state
        initial_state = {
            "query": user_query,
            "route": "",
            "reasoning": "",
            "sql_result": "",
            "rag_result": "",
            "final_answer": "",
            "error": ""
        }
        
        # Execute graph
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 70)
        print("EXECUTION COMPLETE")
        print("=" * 70 + "\n")
        
        return final_state
    
    def visualize(self, output_path: str = "query_graph.png"):
        """
        Generate a visualization of the graph.
        
        Args:
            output_path: Path to save the graph image
        """
        try:
            from IPython.display import Image, display
            
            # Get the graph PNG
            graph_png = self.graph.get_graph().draw_mermaid_png()
            
            # Save to file
            with open(output_path, "wb") as f:
                f.write(graph_png)
            
            print(f"Graph visualization saved to {output_path}")
            
            # Display in notebook if available
            try:
                display(Image(graph_png))
            except:
                pass
                
        except Exception as e:
            print(f"Could not generate visualization: {e}")


if __name__ == "__main__":
    engine = ProdLensQueryEngine()
    
    # Test queries
    test_queries = [
        # "What kind of monitors should buy for gaming purposes?"
        "Show me some monitors with high refresh rate for gaming purposes",
    ]

    # Execute queries
    for query in test_queries:
        result = engine.query(query)
        
        print(f"Results:")
        print(f"  Route taken: {result['route']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Answer: {result['final_answer']}...")
        print("\n" + "=" * 70 + "\n")
    
    # Visualize the graph (optional)
    engine.visualize()