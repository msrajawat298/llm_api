
# Know more about Msrajawat298 Q&A: Question and Answer System Based on Google Palm LLM and Langchain for E-learning company

This is an end to end LLM project based on Google Palm and Langchain. We are building a Q&A system for an e-learning company called vitabletech (website: vitabletech.in). vitabletech sells data related courses and bootcamps. They have thousands of learners who uses discord server or email to ask questions. This system will provide a streamlit based user interface for students where they can ask questions and get answers. 

![](about-me-demo.gif)

## Project Highlights

- Use a real CSV file of FAQs.
- We will build an LLM based question and answer system that can reduce the workload of their human staff.
- Students should be able to use this system to ask questions directly and get answers within seconds

## You will learn following,
  - Langchain + Google Palm: LLM based Q&A
  - Streamlit: UI
  - Huggingface instructor embeddings: Text embeddings
  - FAISS: Vector databse

## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/msrajawat298/llm_by_Msrajawat298
```
2.Navigate to the project directory:

```bash
  cd llm_by_Msrajawat298
```
3. Install the required dependencies Means install docker:

```bash
  docker compose up -d
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```
## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledgeable of FAQs, click on Create Knowledge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter

## Sample Questions
  - Who is Mayank Singh Kushwah?
  - What certifications does Mayank possess?
  - Could you describe Mayank's role at EagleView?
  - What projects has Mayank worked on in the past?
  - What are Mayank's achievements and awards?
  - What skills does Mayank possess?
  - How has Mayank contributed to his previous roles?
  - What is Mayank's approach to learning and professional development?
  - What are Mayank's future goals?
  - Tell me about Mayank's educational background.
  - What programming languages is Mayank proficient in?
  - Describe Mayank's role as an intern at Techcato Projections Pvt. Ltd.

## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your Google API key.