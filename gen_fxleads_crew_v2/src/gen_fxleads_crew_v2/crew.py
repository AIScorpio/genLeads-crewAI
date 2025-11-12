from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import streamlit as st
from gen_fxleads_crew.CustomStreamlitCallbackHandler import CustomStreamlitCallbackHandler, step_callback
from crewai_tools.tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper 
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from datetime import datetime

load_dotenv()

# Uncomment the following line to use an example of a custom tool
# from gen_fxleads_crew_v2.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool
# define tools
scrape_tool = ScrapeWebsiteTool()
tavily_search_tool = TavilySearchResults(max_results=10, api_wrapper=TavilySearchAPIWrapper())
duck_search_tool = DuckDuckGoSearchResults(source="news", num_results=10, api_wrapper=DuckDuckGoSearchAPIWrapper())

# specify a llm
from langchain_groq import ChatGroq
llm1 = ChatGroq(
    # model='mixtral-8x7b-32768',   # mixtral-8x7b-32768, llama3-70b-8192, llama3-8b-8192, gemma-7b-it
	model='llama3-groq-70b-8192-tool-use-preview',
	temperature=0,
    max_tokens=15000,      # TPM: 5000, 6000, 15000, 30000
    max_retries=5,
)

llm2 = ChatGroq(
    # model='mixtral-8x7b-32768',   # mixtral-8x7b-32768, llama3-70b-8192, llama3-8b-8192, gemma-7b-it
	model='llama3-groq-70b-8192-tool-use-preview',
	temperature=0,
    max_tokens=15000,      # TPM: 5000, 6000, 15000, 30000
    max_retries=5,
)

@CrewBase
class GenFxleadsCrewV2():
	"""GenFxleadsCrewV2 crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['research_analyst'],
			tools=[duck_search_tool,],
			verbose=True,
			allow_delegation=False,
            step_callback=lambda step: step_callback(step, "Research Analyst"),
			# llm=llm1,
		)

	@agent
	def reportor(self) -> Agent:
		return Agent(
			config=self.agents_config['report_writer'],
			tools=[duck_search_tool, ],
			verbose=True,
			allow_delegation=False,
            step_callback=lambda step: step_callback(step, "Reporter Writer"),
			# llm=llm1,
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.researcher(),
			output_file=f"/tests/research_{self._time}.md",
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			agent=self.reportor(),
			context=[self.research_task()],
			output_file=f"/tests/report_{self._time}.md",
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the GenFxleadsCrewV2 crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			manager_llm=llm2,
			process=Process.sequential,
			verbose=2,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			max_rpm=30,
		)