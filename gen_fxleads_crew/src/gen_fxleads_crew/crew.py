
# warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools.tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.human import HumanInputRun
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Union
from langchain.pydantic_v1 import BaseModel, Field

import streamlit as st
from gen_fxleads_crew.CustomStreamlitCallbackHandler import CustomStreamlitCallbackHandler, step_callback

# Uncomment the following line to use an example of a custom tool
# from gen_fxleads_crew.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

# load environment variables
load_dotenv()

# define tools
scrape_tool = ScrapeWebsiteTool()
tavily_search_tool = TavilySearchResults(max_results=5)
duck_search_tool = DuckDuckGoSearchResults(source="news", num_results=5)
human_input_tool = HumanInputRun(input_func=st.text_input)

# specify a llm
from langchain_groq import ChatGroq
llm = ChatGroq(
    model='llama3-70b-8192',   # mixtral-8x7b-32768, llama3-70b-8192
	temperature=0,
    max_tokens=5120,
    max_retries=3,
)

# define Leads pydantic class
class LeadsDetails(BaseModel):
    company: str = Field(description="Name of the potential clients") 
    product: List[str] = Field(description="Products and service offerings provided by the company")
    material: str = Field(description="Material that heavily used in the product") 
    region: str = Field(description="Region of interest", default=None)
    operations: str = Field(description="Type of operations", default=None)
    region_status: bool = Field(description="Whether or not the company operates in the region", default=None)
    # company: str
    # product: list[str] = list
    # material: str
    # region: str | None = None
    # operations: list[str] = list
    # region_status: bool = False

class LeadsAll(BaseModel):
    companies: List[LeadsDetails]

@CrewBase
class GenFxleadsCrew():
	"""GenFxleadsCrew crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def industry_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['industry_analyst'],
			tools=[tavily_search_tool, duck_search_tool, scrape_tool],
			verbose=True,
			allow_delegation=False,
			# callbacks=[CustomStreamlitCallbackHandler(color='violet')],
            step_callback=lambda step: step_callback(step, "Industry Analyst Agent"),
		)

	@agent
	def product_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['product_specialist'],
			tools=[tavily_search_tool, duck_search_tool, scrape_tool],
			verbose=True,
			allow_delegation=False,
			max_iter=15,
			# callbacks=[CustomStreamlitCallbackHandler(color='violet')],
			step_callback=lambda step: step_callback(step, "Product Specialist Agent"),
		)
	
	# @agent
	def crew_manager(self) -> Agent:
		print(">>>> Crew Manager <<<<")
		return Agent(
			role="Crew Manager",
			goal=(
				"You are a crew manager. You have to manage the crew "
				"and make sure they do their job well from end to end."
			),
			backstory=(
				"You are a seasoned manager of working crews. "
				"You excel at result driven and fact based decision making. "
			),
			verbose=True,
			allow_delegation=True,
			step_callback=lambda step: step_callback(step, "Crew Manager"),
		)

	# @agent
	# def financial_analyst(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['financial_analyst'],
	# 		tools=[scrape_tool, tavily_search_tool, duck_search_tool],
	# 		verbose=True,
	# 		allow_delegation=False,
	# 		max_iter=15,
	# 		step_callback=lambda step: step_callback(step, "Financial Analst Agent"),
	# 	)
	
	# @agent
	# def FX_CIO(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['FX_CIO'],
	# 		# tools=[tavily_search_tool, duck_search_tool, scrape_tool],
	# 		verbose=True,
	# 		allow_delegation=False,
	# 		max_iter=15,
	# 		step_callback=lambda step: step_callback(step, "Chief FX Options Investment Officer Agent"),
	# 	)

	@task
	def industrial_task(self) -> Task:
		return Task(
			tools=[tavily_search_tool, duck_search_tool,],
			config=self.tasks_config['industrial_task'],
			agent=self.industry_analyst(),
			# callbacks=[CustomStreamlitCallbackHandler(color='violet')],
			# human_input=True,
		)

	@task
	def product_task(self) -> Task:
		return Task(
			tools=[scrape_tool, tavily_search_tool, duck_search_tool,],
			config=self.tasks_config['product_task'],
			agent=self.product_specialist(),
			context=[self.industrial_task()],
		)
	
	# @task
	def manage_crew(self) -> Task:
		return Task(
			tools=[],
			description=(
				"Manage the crew of workers and their tasks "
				"by orchastrate the tasks in the most efficient way to "
				 "maximize the result of the task"
				 "and minimize the risk of failure."
				 "The crew consists of 2 workers:"
				 "- Product Specialist"
				 "- Industry Analyst"
				"IMPORTANT: when delegate to a coworker, "
				"you need to capitalize the first letter of the coworker's title. "
			),
			expected_output=(
				"A list of well vetted potential clients."
			),
			agent=self.crew_manager(),
			out_pydantic=LeadsAll,
			# context=[self.product_task()],
		)
	
	# @task
	# def analytic_task(self) -> Task:
	# 	return Task(
	# 		tools=[scrape_tool, tavily_search_tool, duck_search_tool,],
	# 		config=self.tasks_config['analytic_task'],
	# 		agent=self.financial_analyst(),
	# 		context=[self.industrial_task()],
	# 	)
	
	# @task 
	# def advisory_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['advisory_task'],
	# 		agent=self.FX_CIO(),
	# 		context=[self.product_task(), self.analytic_task()],
	# 		output_file='tests/_04.sales_offerings_3.md',
	# 	)

	@crew
	def crew(self) -> Crew:
		"""Creates the GenFxleadsCrew crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			# manager_llm=llm, 
			# manager_agent=self.crew_manager(),
			# memory=True,
			verbose=2,
			max_rpm=5,
		)
      
