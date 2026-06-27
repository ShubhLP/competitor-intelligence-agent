import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import TavilySearchTool

# load environment variables from .env
load_dotenv()

# initializing Gemini LLM using CrewAI's native LLM wrapper
gemini_llm = LLM(
    model = 'gemini/gemini-2.5-flash',
    api_key = os.getenv("GEMINI_API_KEY"),
    temperature = 1.0  # Use the Gemini 3 recommended temperature
)

# Tavily search tool
web_search_tool = TavilySearchTool()

# agent 1: competitor research agent
researcher = Agent(
    role = 'Senior Competitor Intelligence Analyst',
    goal="Discover, analyze, and synthesize the latest updates, feature releases, or strategic pivots for a given company or topic.",
    backstory=(
        "You are an expert market researcher with an eagle eye for detail. "
        "You know exactly where to look for company news, product updates, and industry shifts. "
        "Your analytical reports are concise, fact-backed, and highlight what truly matters."
    ),
    llm=gemini_llm,
    verbose=True,
    memory=False,
    tools=[web_search_tool],
    allow_delegation=False # This agent works independently on research
)

# agent 2: The LinkedIn Content Strategist
writer = Agent(
    role="Expert B2B Content Strategist",
    goal="Transform dense competitive intelligence data into engaging, highly professional, and viral LinkedIn content.",
    backstory=(
        "You are a master copywriter specializing in the tech and business sectors. "
        "You know how to take complex reports and extract compelling hooks, readable bullet points, "
        "and clear takeaways that drive high engagement and shares on professional networks."
    ),
    llm=gemini_llm,
    verbose=True,
    memory=True,
    allow_delegation=False
)

# task 1: Research Assignment
research_task = Task(
    description=(
        "Conduct a deep-dive investigation into the latest major updates or trends regarding: {topic}. "
        "Identify key product launches, strategic announcements, or market impacts that occurred recently."
    ),
    expected_output=(
        "A comprehensive markdown report detailing at least 3 major insights, "
        "including source data, a brief analysis of why it matters, and a summary of market sentiment."
    ),
    agent=researcher,
    markdown=True
)

# task 2: Writing Assignment (This depends on the output of Task 1)
write_task = Task(
    description=(
        "Review the markdown report provided by the Researcher agent. "
        "Extract the most impactful insights and format them into a polished LinkedIn post. "
        "The post must include a hook line, structured bullet points, actionable takeaways, and relevant hashtags."
    ),
    expected_output="A ready-to-publish LinkedIn post formatted in clean markdown, optimized for professional engagement.",
    agent=writer
)

# Combining agents and tasks
crew = Crew(
    agents = [researcher, writer],
    tasks = [research_task, write_task],
    verbose = True,
    markdown=True
)

if __name__ == "__main__":
    print("Initializing run for the Agent Crew...")
    inputs = {"topic" : "Artificial Intelligence trends in enterprise software"}
    result = crew.kickoff(inputs=inputs)
    print("\n================ FINAL AGENT OUTPUT ================\n")
    print(result)