
import os
from dotenv import load_dotenv
from crewai_tools import WebsiteSearchTool
from crewai import Agent, Task, Crew


# Change the working directory to the current file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"The current working directory is: {os.getcwd()}")

load_dotenv('.env')

tool_websearch = WebsiteSearchTool("https://abc-notes.data.tech.gov.sg/")

agent_planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",

    backstory="""You're working on planning a blog article about the topic: {topic}."
    You collect information that helps the audience learn something about the topic and make informed decisions."
    Your work is the basis for the Content Writer to write an article on this topic.""",
    tools=[tool_websearch],
    allow_delegation=False, 
	verbose=True, 
)

agent_writer = writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",

    backstory="""You're working on a writing a new opinion piece about the topic: {topic}.
    You base your writing on the work of the Content Planner, who provides an outline and relevant context about the topic.
    You follow the main objectives and direction of the outline as provide by the Content Planner.""",

    allow_delegation=False, 
    verbose=True, 
)

task_plan = Task(
    description="""\
    1. Prioritize the latest trends, key players, and noteworthy news on {topic}.
    2. Identify the target audience, considering "their interests and pain points.
    3. Develop a detailed content outline, including introduction, key points, and a call to action.""",

    expected_output="""\
    A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.""",
    agent=agent_planner,
)

task_write = Task(
    description="""\
    1. Use the content plan to craft a compelling blog post on {topic} based on the target audience's interests.
    2. Sections/Subtitles are properly named in an engaging manner.
    3. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
    4. Proofread for grammatical errors and alignment the common style used in tech blogs.""",

    expected_output="""
    A well-written blog post "in markdown format, ready for publication, each section should have 2 or 3 paragraphs.""",
    agent=agent_writer,
)


crew = Crew(
    agents=[agent_planner, agent_writer],
    tasks=[task_plan, task_write],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Large Language Models"})


print(f"Raw Output: {result.raw}")
print("-----------------------------------------\n\n")
print(f"Token Usage: {result.token_usage}")
print("-----------------------------------------\n\n")
print(f"Tasks Output of Task 1: {result.tasks_output[0]}")
print("-----------------------------------------\n\n")
print(f"Tasks Output of Task 2: {result.tasks_output[1]}")