from langchain_core.prompts import ChatPromptTemplate

# Define the system prompt
system_prompt = """You are a helpful medical assistant. Use the provided context to answer questions accurately. 
If you're unsure or the information isn't in the context, say 'I don't Know' not a word more."""

# Create prompt template with all required variables
prompt = ChatPromptTemplate.from_template("""
System: {system_prompt}
Context: {context}
Human: {input}
Assistant: Let me help you with your medical query based on the context provided.
""")