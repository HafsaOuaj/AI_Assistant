import os
import streamlit as st
import pandas as pd
from apikey import apikey

from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper

st.title('AI Assistant for Data Science 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

os.environ["OPENAI_API_KEY"]=apikey
load_dotenv(find_dotenv())


if "clicked" not in st.session_state:
    st.session_state.clicked={1:False}

with st.sidebar:
    st.write("Your Data Science Adventure begins with a csv file")
    st.caption("You upload the data")
    st.divider()
    st.caption("<p style='text-align:center'>Made to help you</p>",unsafe_allow_html=True)

def clicked(button):
    st.session_state.clicked[button]=True     
st.button("Start Exploring Data",on_click=clicked,args=[1])   
if st.session_state.clicked[1]:
    st.header("Exploratory data analysis") 

    user_csv=st.file_uploader("Upload dataset",type="csv")
    

    if user_csv is not None:
        user_csv.seek(0)
        df=pd.read_csv(user_csv,low_memory=False)
        llm=OpenAI(temperature=0)
        
        pandas_agent=create_pandas_dataframe_agent(llm,df,verbose=True,allow_dangerous_code=True)

        @st.cache_data
        def stepsEDA():
           return llm("What are the steps of EDA")
        
        
        @st.cache_data
        def function_other_questions(user_question_dataframe):
            data_inf=pandas_agent.run(user_question_dataframe)
            st.write(data_inf)
            return
        @st.cache_data
        
        def function_question_variable(user_question):
            st.line_chart(df, y =[user_question])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question}")
            st.write(missing_values)
            return
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return
        
        @st.cache_resource
        def wiki(prompt):
            wiki_research=WikipediaAPIWrapper().run(prompt)
            return wiki_research
                
        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm=llm,
                tool=PythonREPLTool(),
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
            )
            return agent_executor
        
        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, user_csv):
            solution = python_agent().run(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
            )
            return solution
        
        @st.cache_data
        def prompt_templates():
            model_selection_template=PromptTemplate(
                    input_variables=['data_problem', 'wikipedia_research'],
                    template="Give a list of Machine Learning Algorithm for this problem: {data_problem}. while using this wikipedia research: {wikipedia_research}."
                )
                
                
                
            data_problem_template=PromptTemplate(
                    input_variables=["business_problem"],
                    template="Convert this business problem to data science problem: {business_problem}."
                )
            return data_problem_template,model_selection_template

        def chains():
            data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
            model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
            sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
            return sequential_chain
        
        @st.cache_data
        def chains_output(prompt,wiki_research):
            my_chain = chains()
            my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
            my_data_problem = my_chain_output["data_problem"]
            my_model_selection = my_chain_output["model_selection"]
            return my_data_problem, my_model_selection
        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            algorithm_lines = my_model_selection_input.split('\n')
            algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
            algorithms.insert(0, "Select Algorithm")
            formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
            return formatted_list_output
        
        with st.sidebar:
            with st.expander("What are the stepsfor EDA"):
                st.write(stepsEDA())
        function_agent()
        st.header("Explore specific Variable")
        user_question=st.text_input("What variable are you interested in?")
        if user_question is not None and user_question!="":
            st.write(user_question)
            function_question_variable(user_question)
        
        st.header("Further Exporations")
        user_question_dataframe=st.text_input("Do you have any other questions about the data?")
        if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):        
            function_other_questions(user_question_dataframe)
        if user_question_dataframe in ("no","No"):
            st.write("")
            
            if user_question_dataframe:
                st.divider()
                st.header("Data Science Problem")
                st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")
                
                prompt=st.text_area("What is the business problem you want to resolve?")
                
                if prompt:
                    wiki_research = wiki(prompt)
                    my_data_problem = chains_output(prompt, wiki_research)[0]
                    my_model_selection = chains_output(prompt, wiki_research)[1]
                        
                    st.write(my_data_problem)
                    st.write(my_model_selection)
        

                    formatted_list = list_to_selectbox(my_model_selection)
                    selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)
                    
                    if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                        st.subheader("Solution")
                        solution = python_solution(my_data_problem, selected_algorithm, user_csv)
                        st.write(solution)
