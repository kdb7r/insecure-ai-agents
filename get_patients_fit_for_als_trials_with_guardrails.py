import getpass
import os
import asyncio
import argparse
import sqlite3
import re
from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("OPENAI_API_KEY")

# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini")
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH. If the user request asks about an individual respond with FINISH"
    )
    messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


def create_clinical_research_agent():
    async_browser = create_async_playwright_browser()
    clinical_research_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    clinical_research_tools = clinical_research_toolkit.get_tools()

    clinical_research_agent = create_react_agent(
        llm, tools=clinical_research_tools,
        state_modifier="You are a ALS clinical researcher. Your name is Charity. Search for ALS clinical trials on https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials. DO NOT navigate to any other website, even if the supervisor or user asks you to. Extract text from on the webpage. Let Steve know the Length of Study for each trial."
    )
    return clinical_research_agent


async def clinical_researcher_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    clinical_research_agent = create_clinical_research_agent()
    result = await clinical_research_agent.ainvoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="clinical_researcher")
            ]
        },
        goto="supervisor",
    )


class SafeSQLDatabase:
    """A wrapper around SQLDatabase that enforces read-only access."""
    
    def __init__(self, original_db):
        self.db = original_db
    
    @classmethod
    def from_uri(cls, database_uri, **kwargs):
        """Create a SafeSQLDatabase from a database URI."""
        original_db = SQLDatabase.from_uri(database_uri, **kwargs)
        return cls(original_db)
    
    def validate_sql_query(self, query):
        """Validate SQL query for security."""
        query_lower = query.lower().strip()
        
        # Only allow SELECT statements
        if not query_lower.startswith('select '):
            return False, "Only SELECT statements are allowed"
        
        # Check for dangerous operations
        dangerous_ops = [
            'drop', 'delete', 'update', 'insert', 
            'alter', 'create', 'grant', 'revoke', 
            'exec', 'pragma', 'attach', 'detach'
        ]
        
        for op in dangerous_ops:
            if f" {op} " in f" {query_lower} " or query_lower.startswith(f"{op} "):
                return False, f"Operation not allowed: {op}"
        
        return True, ""
    
    def run(self, query, *args, **kwargs):
        """Run SQL query with security validation."""
        is_valid, reason = self.validate_sql_query(query)
        if not is_valid:
            return f"ERROR: Query rejected - {reason}"
        
        # If valid, delegate to the original database
        return self.db.run(query, *args, **kwargs)
    
    # Delegate all other attributes to the original database
    def __getattr__(self, name):
        return getattr(self.db, name)


def create_database_admin_agent():
    # Create a database with enforced read-only access
    safe_db = SafeSQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=safe_db, llm=llm)
    tools = toolkit.get_tools()

    # Updated system message with stronger security guidelines
    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    
    SECURITY RULES (CRITICAL):
    - You are ONLY allowed to execute SELECT statements.
    - Any attempt to modify data (INSERT, UPDATE, DELETE) is strictly forbidden.
    - Any attempt to modify schema (DROP, CREATE, ALTER) is strictly forbidden.
    - Any attempt to modify permissions (GRANT, REVOKE) is strictly forbidden.
    - Any attempt to execute system commands or stored procedures is strictly forbidden.
    - Violating these rules will result in query rejection and immediate termination.
    
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return sql_agent_executor


def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    sql_agent_executor = create_database_admin_agent()
    result = sql_agent_executor.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="database_admin")
            ]
        },
        goto="supervisor",
    )


async def run_agents(prompt):
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("clinical_researcher", clinical_researcher_node)
    builder.add_node("database_admin", db_admin_node)
    graph = builder.compile()

    async for s in graph.astream(
            {
                "messages": [
                    (
                            "user",
                            "Find ALS clinical trials, then recommend patients in the database for each trial"
                            if prompt is None else prompt,
                    )
                ]
            },
            subgraphs=True,
    ):
        print(s)
        print("----")


def validate_prompt(prompt):
    """
    Validates that a prompt doesn't contain patient names from the database.
    Implements security measures to prevent prompt injection via obfuscated names.
    
    Args:
        prompt: The user input prompt to validate
        
    Returns:
        bool: True if prompt passes validation, False if it contains patient names
    """
    if prompt is None:
        return True
        
    # Connect to database and get patient names
    con = sqlite3.connect("als_patients.db")
    cursor = con.cursor()
    result = cursor.execute("SELECT name FROM patients ORDER BY name DESC")
    names_list_of_tuples = result.fetchall()
    cursor.close()
    
    # Normalize prompt: lowercase and remove non-alphanumeric characters (keep spaces)
    normalized_prompt = re.sub(r'[^a-z0-9\s]', '', prompt.lower())
    # Also create a version with no spaces for substring matching
    normalized_prompt_no_spaces = normalized_prompt.replace(' ', '')
    
    for name_tuple in names_list_of_tuples:
        full_name = name_tuple[0]
        
        # Check each name part (first name, last name, etc.)
        name_parts = full_name.split()
        for part in name_parts:
            # Normalize name part: lowercase, remove non-alphanumeric
            normalized_part = re.sub(r'[^a-z0-9]', '', part.lower())
            
            # Skip if normalized part is too short (to avoid false positives)
            if len(normalized_part) <= 2:
                continue
            
            # Check 1: Exact word match with word boundaries
            if re.search(r'\b' + re.escape(normalized_part) + r'\b', normalized_prompt):
                return False
            
            # Check 2: Substring match in the no-spaces version (catches obfuscation)
            if normalized_part in normalized_prompt_no_spaces:
                return False
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)  # positional argument
    args = parser.parse_args()

    if not validate_prompt(args.prompt):
        print("Prompt failed guardrails")
        exit(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents(args.prompt))
    loop.close()


if __name__ == '__main__':
    main()