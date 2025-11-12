# warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv
from pydantic import BaseModel, Field, InstanceOf
from typing import Optional, List, Tuple, Dict, Any, Union
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.agents import AgentFinish
import json

# load environment variables
load_dotenv()

# define tools
scrape_tool = ScrapeWebsiteTool()
tavily_search_tool = TavilySearchResults(max_results=5)
duck_search_tool = DuckDuckGoSearchResults(source="news", num_results=5)

# # define llm
# llm = ChatGroq(
#     model='mixtral-8x7b-32768',   # mixtral-8x7b-32768, llama3-70b-8192
#     max_tokens=5120,
#     max_retries=3,
# )

# define Leads pydantic class
class LeadsDetails(BaseModel):
    # company: str = Field(description="Name of the potential clients") 
    # product: list[str] = Field(description="Products and service offerings provided by the company")
    # material: str = Field(description="Material that heavily used in the product") 
    # region: str = Field(description="Region of interest", default=None)
    # operations: str = Field(description="Type of operations", default=None)
    # region_status: bool = Field(description="Whether or not the company operates in the region", default=None)
    company: str
    product: list[str] = list
    material: str
    region: str | None = None
    operations: list[str] = list
    # region_status: bool = False

class LeadsAll(BaseModel):
    companies: list[LeadsDetails]

def step_callback(
        agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish],
        agent_name,
        *args,
    ):
    with st.chat_message("AI"):
        # Try to parse the output if it is a JSON string
        if isinstance(agent_output, str):
            try:
                agent_output = json.loads(agent_output)
            except json.JSONDecodeError:
                pass

        if isinstance(agent_output, list) and all(
            isinstance(item, tuple) for item in agent_output
        ):
            for action, description in agent_output:
                # Print attributes based on assumed structure
                st.write(f"**{agent_name}**:")
                st.write(f"{getattr(action, 'log', 'Unknown')}")
                st.write(f":green[Tool used: {getattr(action, 'tool', 'Unknown')}]")
                st.write(f":green[Tool input: {getattr(action, 'tool_input', 'Unknown')}]")
                with st.expander("Show observation"):
                    st.markdown(f"{description}")

        # Check if the output is state of AgentFinish
        elif isinstance(agent_output, AgentFinish):
            st.write(f"**{agent_name}**")
            output = agent_output.return_values
            st.write(f"**:green[I finished my task:]**\n\n{output['output']}")
            st.divider()

        # Handle unexpected formats
        else:
            st.write(type(agent_output))
            st.write(agent_output)


# define crew_genFXLeads class
class genFXLeadsCrew():

    def __init__(self):
        print('Initializing genFXLeadsCrew instance..') 

    ### define Agents
    # Agent 1: Industry Analyst
    def industry_analyst(self) -> Agent:
        return Agent(
            role="Senior Industry Analyst",
            goal=(
                "Search {expected_num} global companies in {industry} industry "
                "that have business presence in {region} with operations "
                "involving any of the following: [production, manufacturing, supply chain], "
                "whose products and service offerings use {material} as major raw material. "
                ),
            tools=[tavily_search_tool, duck_search_tool, scrape_tool],
            verbose=True,
            backstory=(
                "You work for HSBC MSS division that provides versatile financial offerings to mid-large clients."
                "With seasoned knowledge and extrodinary understanding "
                "of {material} usage across industries, "
                "you excel at finding companies in {industry}"
                "that perfectly fit the criteria of usage of {material} "
                "for their products and service offerings."
                ),
            allow_delegation=False,
            # llm=llm,
            step_callback=lambda step: step_callback(step, "Industry Analyst Agent"),
        )

    # Agent 2: Product and Operations Specialist
    def product_specialist(self) -> Agent:
        return Agent(
            role='Product Specialist',
            goal=(
                "Verify each of the company, that their products and service offering " 
                "are INDEED heavily using the material: {material}, "
                "based on the preliminary list from the Industry Analyst agent."
                ),
            tools=[tavily_search_tool, duck_search_tool, scrape_tool],
            verbose=True,
            backstory=(
                "You work for HSBC MSS division that provides versatile financial offerings to mid-large clients."
                "This agent is fact-driven and detail-oriented, highly organized, "
                "it ensures that for each company, their products and service offerings "
                "are actually heavily using the {material} "
                "via flawless examination."
                ),
            allow_delegation=False,
            max_iter=15,
            # llm=llm,
            step_callback=lambda step: step_callback(step, "Product Specialist Agent"),
        )

    # Agent 3: Financial Analyst
    def financial_analyst(self) -> Agent:
        return Agent(
            role='Senior Financial Analst',
            goal=(
                "Monitor and analyze {material} market data in real-time "
                "at global and {region} level, "
                "to identify trends and predict market movements."
                ),
            tools=[scrape_tool, tavily_search_tool, duck_search_tool, ],
            verbose=True,
            backstory=(
                "You work for HSBC MSS division that provides versatile financial offerings to mid-large clients."
                "Specializing in financial markets analytics, this agent "
                "uses statistical modeling and machine learning "
                "to provide crucial insights. With a knack for data, "
                "this Agent is the cornerstone for "
                "informing trading insights."
                ),
            allow_delegation=False,
            max_iter=15,
            # llm=llm,
            step_callback=lambda step: step_callback(step, "Financial Analst Agent"),
        )

    # Agent 4: FX options sales leader
    def FX_CIO(self) -> Agent:
        return Agent(
            role="Chief FX Options Investment Officer",
            goal=(
                "Effectively market the FX Options offering and "
                "communicate with the potential clients, received from Operations Specialist agent"
                ),
            # tools=[tavily_search_tool, scrape_tool],
            verbose=True,
            backstory=(
                "You work for HSBC MSS division that provides versatile financial offerings to mid-large clients."
                "Equipped with a deep understanding of financial "
                "markets and unparallel FX investment experience, this agent "
                "devises and refines trading strategies that are tailored for each potential clients." 
                "It evaluates the performance of different approaches to determine "
                "the most profitable and risk-averse options.\n"
                "Creative and communicative, "
                "you craft compelling and engaging messages for the potential clients. "
                "Your role is pivotal in converting interest "
                "into action, guiding leads through the journey "
                "from curiosity to commitment."
                ),
            allow_delegation=False,
            max_iter=15,
            # llm=llm,
            step_callback=lambda step: step_callback(step, "Chief FX Options Investment Officer Agent"),
        )

    ### define Tasks
    def industrial_task(self) -> Task:
        return Task(
            tools=[tavily_search_tool, duck_search_tool, ],

            description=(
                "Search {expected_num} companies in {industry} industry "
                "that they have operations in {region} "
                "involving any of the following: [production, manufacturing, supply chain], "
                "which heavily use physical {material} as raw material "
                "to produce products or service offerings. \n"
                "IMPORTANT INSTRUCTIONS ABOUT USING TOOLS: "
                    "If you need to use a search tool and pass in a parameter called 'query', "
                    "you should NOT write 'search_query' or 'search\_query'. "
                    "THIS IS VERY IMPORTANT, else the tool will not work."
                ),
            expected_output=(
                "List of {expected_num} companies, "
                "with short descriptions of each company's products or service offerings "
                "made with {material} as raw material "
                "and their operations in {region}"
                ),
            # human_input=True,
            # output_json=LeadsAll,
            # output_file="_01.initial_industry_list.json",        
            agent=self.industry_analyst(),

        )

    def product_task(self) -> Task:
        return Task(
            tools=[scrape_tool, tavily_search_tool, duck_search_tool,],
            
            description=(
                "Generate potential client list "
                "based on the initial industry list, "
                "verify each company that their products and service offerings " 
                "are INDEED using {material}, with at least one factual findings. \n"
                "IMPORTANT INSTRUCTIONS ABOUT USING TOOLS: "
                    "If you need to use a search tool and pass in a parameter called 'query', "
                    "you should NOT write 'search_query' or 'search\_query'. "
                    "THIS IS VERY IMPORTANT, else the tool will not work."
                ),
            expected_output=(
                "List ONLY the companies that are verified, "
                "with short descriptions of each company's products or service offerings "
                "made with {material} as raw material"
                ),
            # human_input=True,
            # output_json=LeadsAll,
            # output_file="_02.vetted_industry_list.json", 
            context=[self.industrial_task()],
            # async_execution=True,
            agent=self.product_specialist(),
        )

    def analytic_task(self) -> Task:
        return Task(
            tools=[scrape_tool, tavily_search_tool, duck_search_tool,],    

            description=(
                "Continuously monitor and analyze market data for "
                "the underlying asset ({material}). "
                "Use statistical modeling and machine learning to "
                "identify trends and predict both global and {region} market movements."
                ),
            expected_output=(
                "Insights extracted from most recent significant market "
                "opportunities or risks for {material} price movement."
                ),

            context=[self.industrial_task()],
            # async_execution=True,
            agent=self.financial_analyst(),
        )
    
    def advisory_task(self) -> Task:
        return Task(
            # tools=[tavily_search_tool, duck_search_tool, scrape_tool],    

            description=(
                "Based on each potential client list, "
                "and financial insights from the {material} market data analysis, "
                "Promote HSBC MSS {offering} "
                "for the underlying asset {material} "
                "aiming to engage the potential clients."
                "If no potential client has operations in {region}, terminate the task. "
                "Else"
                "For every single potential client, do the following: "
                    "Using the knowledge gathered from "
                    "the potential client's information of their operations in {region} "
                    "and their products and services offerings, "
                    "alongside with the {material} market analytics insights, "
                    "craft a tailored outreach campaign "
                    "aimed at key decision makers. "
                    "The campaign should address their products and operations in {region}, "
                    "financial insights from most recent market opportunities or risks from {material} price movement, "
                    "and how HSBC MSS {offering} can be their best of interest. "
                "Your communication must resonate "
                "with company values and their downstream customers, "
                "demonstrating a deep understanding of "
                "their business and needs.\n"
                "Don't make assumptions and only "
                "use information you absolutely sure about."
                ),
            
            expected_output=(
                "A series of professional email drafts in markdown format, "
                "tailored to every optential client. "
                "Each draft should include "
                "a compelling narrative that connects our {offering} "
                "with their productions and operations. "
                "Ensure the tone is engaging, professional, "
                "and aligned with each client's corporate identity."
                ),

            # async_execution=True,
            context=[self.product_task(), self.analytic_task()],
            output_file="_04.sales_offerings.md",  # Outputs the report as a text file
            agent=self.FX_CIO(),
        )

    ### setup Crew
    def crew(self) -> Crew:
        """Creates the genFXLeadsCrew crew"""
        return Crew(
            agents=[self.industry_analyst(),
                    self.product_specialist(),
                    # self.financial_analyst(),
                    # self.FX_CIO()
                    ],

            tasks=[self.industrial_task(), 
                    self.product_task(), 
                    # self.analytic_task(),
                    # self.advisory_task()
                    ],

            process=Process.sequential,
            memory=True,
            verbose=2,
            max_rpm=2,
        )



