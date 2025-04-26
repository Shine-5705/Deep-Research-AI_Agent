
# 🧠 Deep Research AI Agentic System
### Overview
This project is a Deep Research AI Agentic System that crawls websites using Tavily for online information gathering. Implemented using a dual-agent system with one agent focused on research and data collection, while the second agent functions as an answer drafter. It allows users to enter a question and receive a summarized, accurate response based on current information gathered from the web.

### 🔧 Tech Stack
- Frontend/UI: Streamlit
- LLM Integration: OpenAI GPT (via langchain-openai)
- Web Search: Tavily API
- Orchestration: LangGraph for multi-agent workflow
- Environment Management: dotenv

---

### 🛠️ Features
- Research Agent: Crawls the web using Tavily to collect relevant and current content.
- Answer Drafter Agent: Uses a GPT model to synthesize a structured and concise answer from the research.
- Agentic Flow: Implemented using LangGraph to model a sequence of interactions between agents.
- Frontend App: Built with Streamlit to provide an easy-to-use interface.
- Auto Reset UI: JavaScript logic clears user input after each query.

---

### 🔁 Application Flow
- User Input: A user submits a research question via the UI.
- Research Agent: Searches web sources using the Tavily API and extracts top 5 relevant snippets.
- Answer Drafter Agent: Formats the content and feeds it to a prompt for GPT, which returns a final structured answer.
- Display: The result is displayed in the frontend.
- Reset Option: Users can reset the input and start a new session.

---

### 🧩 Components
1. app.py (Frontend): 
Renders UI with a title, input form, and buttons.
Triggers run_deep_research_system() upon form submission.
Displays the answer.
Executes a small JavaScript snippet via components.html() to clear the input field on reset.

2. deep_research_system.py (Backend)
Loads API keys from .env or environment variables.
Defines ResearchState, the shared state dictionary.
research_agent(): Uses Tavily to collect data from the web.
answer_drafter_agent(): Builds a prompt with the data and uses GPT to generate an answer.

---

### Uses LangGraph to model the flow:
research_agent → answer_drafter_agent → END

---

### 📦 Environment Setup
```bash
streamlit   
tavily
langgraph
langchain
langchain-openai
python-dotenv
```

---

## 🔐 Environment Variables
Add the following secrets to Hugging Face (or a .env file if running locally):   
OPENAI_API_KEY=your_openai_key   
TAVILY_API_KEY=your_tavily_key   

---

### ▶️ Running the App
```bash
streamlit run app.py
```
Hugging Face Project URL: https://huggingface.co/spaces/shubhamgs/Deep_Research_AI_Agent   
On Hugging Face Spaces, make sure:   
app.py is the main file   
requirements.txt is present   
Secrets key are added properly   

---

## **Author** 
👤 Shine Gupta
🔗 GitHub: https://github.com/Shine-5705  
🔗 Hugging Face Project URL: https://huggingface.co/spaces/Shinegupta/Deep_Research_AIAgent