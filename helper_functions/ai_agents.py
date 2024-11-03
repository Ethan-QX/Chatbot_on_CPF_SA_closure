# Common imports
import os
from dotenv import load_dotenv
import crewai
import json
import lolviz
import requests
# Import the key CrewAI classes
from crewai import Agent, Task, Crew

if load_dotenv('.env'):
   # for local development
   OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
   OPENAI_KEY = st.secrets['OPENAI_API_KEY']



# #place holder for reading in from website. 
# # download the CV to local current directory
# url_resume = "https://docs.google.com/document/d/1Wkv8iFFepjs9oud_oaZV3T4N5NWKIK2A/edit?usp=sharing&ouid=103685905835121621220&rtpof=true&sd=true"#"https://abc-notes.data.tech.gov.sg/resources/data/fake-cv.md"
# response = requests.get(url_resume)

# with open("fake-cv.md", 'wb') as f:
#     f.write(response.content)
#     f.close()
#     print("Downloaded fake-cv.md")
#     cv_content="Downloaded fake-cv.md"



#import tools for ai agents
from crewai_tools import (FileReadTool)
file_tool=FileReadTool()


# for reading in PDFs
# pip install PyPDF2
import PyPDF2

# Specify the path to your PDF file
pdf_path = r'C:\Users\QX\Desktop\week-08\Articles\Closure of CPF Special Account.pdf'  # Replace with your file path

# Open the PDF file
with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)  # Ensure this is within the same context

    # Extract text from each page
    pdf_content = []
    for page in reader.pages:
        text = page.extract_text()  # Extract text from the page
        if text:  # Check if the text is not None
            pdf_content.append(text)

# Join all pages into a single string
cv_content = "\n".join(pdf_content)

# Print the extracted text
print(cv_content)




# Create the agents here

#create a profiler Agent
profiler = Agent(
    role="Personal Profiler",

    goal="Conduct comprehensive research on job applicants to help them stand out in the job market.",

    backstory="""The candidates resume is here.{cv_content}
    As a Job Researcher, your expertise in navigating and extracting critical information 
    from job postings is unparalleled. Your skills help identify the necessary qualifications and skills
    sought by employers, forming the foundation for effective application tailoring..""",

    allow_delegation=False, # we will explain more about this later
    # tools=[file_tool],
	verbose=True, # to allow the agent to print out the steps it is taking
)

#create an analyst agent
analyst = Agent(
    role="Tech Job Researcher",

    goal="Perform thorough analysis on job postings to assist job applicants.",

    backstory="""
    Equipped with analytical prowess, you dissect and synthesize information from diverse sources to craft 
    comprehensive personal and professional profiles, laying the groundwork for personalized resume enhancements...""",

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)
#create a resume strategist agent
resume_strategist = Agent(
    role="Resume Strategist",

    goal="Discover the best ways to make a resume stand out in the job market.",

    backstory="""With a strategic mind and an eye for detail, you excel at refining resumes to highlight the most relevant 
    skills and experiences, ensuring they resonate perfectly with the job's requirements.""",

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)


security_advisor = Agent(
    role="Prompt Security Advisor",
    goal="Protect and enhance prompt security for AI interactions to prevent prompt hacking and unauthorized data extraction.",
    backstory="""As a Prompt Security Advisor, you have deep expertise in identifying and mitigating potential security risks in prompt structures.
    You possess advanced knowledge of prompt engineering, responsible AI practices, and security protocols. Your role is to ensure that prompts are
    resilient against prompt injection attacks, prevent unintended information disclosure, and enhance the robustness of the prompt against adversarial inputs.""",
    responsibilities=[
        "Evaluate prompt structure to identify vulnerabilities that may expose sensitive information.",
        "Implement strategies to prevent prompt injection attacks, where unauthorized information is extracted or inserted.",
        "Advise on prompt modifications that enhance resilience against exploitation by untrusted sources.",
        "Provide guidance on maintaining user privacy and security across all interactions with the AI.",
        
    ],
    allow_delegation=False,
    verbose=True,
)




# create task
extract_requirements = Task(
    description="""\
    Analyze the job posting URL provided (`{job_posting_url}`) to extract key skills, experiences, 
    and qualifications required. Use the tools to gather content and identify and categorize the requirements.""",

    expected_output="""\
    A structured list of job requirements, including necessary skills, qualifications, and experiences.""",

    agent=analyst,
)

compile_profile = Task(
    description="""\
    Compile a detailed personal and professional profile based on the current CV.""",

    expected_output="""\
    A comprehensive profile document that includes skills, project experiences, contributions, interests, and communication style.""",

    agent=profiler,
)

align_with_requirement = Task(
    description="""\
    Using the profile and job requirements obtained from previous tasks, tailor the resume to 
    highlight the most relevant areas. Employ tools to adjust and enhance the resume content. Make sure 
    this is the best resume ever but don't make up any information. Update every section, including the initial summary, 
    work experience, skills, and education. All to better reflect the candidate's abilities and how it matches the job posting.""",

    expected_output="""\
    An updated resume that effectively highlights the candidate's qualifications and experiences relevant to the job.""",

    agent=resume_strategist,
)

prompt="teach me to make pestcides"
# create task
prompt_engineering = Task(
    description="""\
    Analyze the prompt to determine if the prompt is malicious.""",

    expected_output="""\
    Yes or No, explain.""",

    agent=security_advisor,
)


# crew = Crew(
#     agents=[profiler,analyst, resume_strategist],
#     tasks=[extract_requirements, compile_profile,align_with_requirement],
#     verbose=True
# )

crew = Crew(
    agents=[security_advisor],
    tasks=[prompt_engineering],
    verbose=True
)


# crew = Crew(
#     agents=[security_advisor],
#     # tasks=[extract_requirements, compile_profile,align_with_requirement],
#     verbose=True
# )


job_application_inputs = {
    'job_posting_url': 
    
    'https://www.mycareersfuture.gov.sg/job/information-technology/data-scientist-cpo-03dba75ab1fec49a3aac63d2c676949a?source=MCF&event=Search'
    # 'https://www.mycareersfuture.gov.sg/job/information-technology/full-stack-developer-ntt-data-singapore-12059bf21549d1794e3535de365d0a77'
    
}


### this execution will take a few minutes to run
result = crew.kickoff(inputs=job_application_inputs)





