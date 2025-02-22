import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Health Care Assistant')
input_text = st.text_input("Enter your symptoms or health concern")

# Memory
symptoms_memory = ConversationBufferMemory(input_key='symptoms', memory_key='chat_history')
medications_memory = ConversationBufferMemory(input_key='condition', memory_key='medication_history')
nutrition_memory = ConversationBufferMemory(input_key='condition', memory_key='nutrition_history')

# OpenAI LLM
llm = OpenAI(temperature=0.8)

# First chain: Symptom analysis
symptoms_prompt = PromptTemplate(
    input_variables=['symptoms'],
    template="List 2 possible conditions for these symptoms: {symptoms}. Summarize."
)

symptoms_chain = LLMChain(
    llm=llm, prompt=symptoms_prompt, verbose=True, output_key='condition', memory=symptoms_memory
)

# Second chain: First aid medications
medications_prompt = PromptTemplate(
    input_variables=['condition'],
    template="Provide 2 first aid medications for {condition}. Summarize."
)

medications_chain = LLMChain(
    llm=llm, prompt=medications_prompt, verbose=True, output_key='medications', memory=medications_memory
)

# Third chain: Nutritional food recommendations
nutrition_prompt = PromptTemplate(
    input_variables=['condition'],
    template="Recommend 2 nutritional foods for {condition}. Summarize."
)

nutrition_chain = LLMChain(
    llm=llm, prompt=nutrition_prompt, verbose=True, output_key='nutrition', memory=nutrition_memory
)

# Sequential chain
healthcare_chain = SequentialChain(
    chains=[symptoms_chain, medications_chain, nutrition_chain],
    input_variables=['symptoms'],
    output_variables=['condition', 'medications', 'nutrition'],
    verbose=True
)

if input_text:
    results = healthcare_chain({'symptoms': input_text})

    # Display results
    st.write("### Summary of Results")
    st.write("**Possible Conditions:**", results.get('condition'))
    st.write("**First Aid Medications:**", results.get('medications'))
    st.write("**Nutritional Foods:**", results.get('nutrition'))

    # Expandable sections for detailed memory
    with st.expander('Possible Conditions'):
        st.info(symptoms_memory.buffer)

    with st.expander('First Aid Medications'):
        st.info(medications_memory.buffer)

    with st.expander('Nutritional Foods'):
        st.info(nutrition_memory.buffer)