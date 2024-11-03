import os
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

if load_dotenv('.env'):
   # for local development
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
   SERPER_API_KEY=os.getenv('SERPER_API_KEY')
else:
   OPENAI_KEY = st.secrets['OPENAI_API_KEY']

# Instantiate tools
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide find the most relevant articles',
    backstory='search websites ending with .gov.sg to find up to date information provided by the government of singapore.',
    tools=[search_tool, web_rag_tool],
    verbose=True
)

# Define tasks
research = Task(
    description='Find the most relevant article based on the provided prompt{prompt} and provide a summary.',
    expected_output='The key points of the article in an format suitable for Rag',
    agent=researcher
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[researcher],
    tasks=[research],
    verbose=True,
    planning=True,  # Enable planning feature
)
inputs={"prompt":"tell me about the closure of Special Account at 55 years old"}
# Execute tasks
crew.kickoff(inputs=inputs)
