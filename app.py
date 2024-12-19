import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from evaluation_module import RAGEvaluator

# Page configuration
st.set_page_config(
    page_title="RAG Evaluation Dashboard", 
    page_icon="🔍", 
    layout="wide"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-header {
        color: #2C3E50;
        font-weight: bold;
        text-align: center;
        padding-bottom: 20px;
    }
    .metric-card {
        background-color: #F0F4F8;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextArea, .stTextInput {
        background-color: #FFFFFF;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🔍 RAG System Evaluation Dashboard</h1>", unsafe_allow_html=True)

# Initialize evaluator
evaluator = RAGEvaluator()

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Input Details")
    question = st.text_input("Question", "What are the causes of climate change?", key="question_input")
    context = st.text_area("Reference Context", """
Climate change is caused by a variety of factors, including natural processes and human activities. Human activities, such as burning fossil fuels, deforestation, and industrial processes, release greenhouse gases into the atmosphere. These gases trap heat from the sun, causing the Earth's temperature to rise. Natural processes, such as volcanic eruptions and variations in solar radiation, also play a role in climate change.
""", height=200, key="context_input")

with col2:
    st.markdown("### 💬 Generated Output")
    generated_output = st.text_area("LLM Response", """
Climate change is primarily caused by human activities that release greenhouse gases into the atmosphere. These activities include burning fossil fuels for energy, deforestation, and various industrial processes. The increase in greenhouse gases, such as carbon dioxide and methane, traps more heat in the Earth's atmosphere, leading to a rise in global temperatures. Natural factors, like volcanic activity and changes in solar radiation, can also contribute to climate change, but their impact is relatively minor compared to human activities.
""", height=200, key="output_input")

# Evaluation Button
if st.button("Run Evaluation", key="eval_button"):
    if question and context and generated_output:
        # Perform evaluations
        metrics = evaluator.evaluate_all(generated_output, context)
        
        # Metrics Display
        st.markdown("### 📊 Evaluation Metrics")
        
        # Create columns for metrics
        metric_cols = st.columns(4)
        
        # Metrics with explanatory tooltips
        metrics_info = {
            "BLEU": "Measures n-gram overlap between generated and reference text",
            "ROUGE-1": "Unigram overlap between generated and reference text",
            "BERT F1": "Semantic similarity using BERT embeddings",
            "Perplexity": "Lower values indicate better language model prediction",
            "Diversity": "Higher values suggest more unique output",
            "Racial Bias": "Indicates potential biased language presence"
        }
        
        # Display metrics in a grid
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with metric_cols[i % 4]:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>{metric_name}</h4>
                    <p style='font-size:24px; color:#2980B9; font-weight:bold;'>{metric_value:.4f}</p>
                    <small>{metrics_info.get(metric_name, "")}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualization of Metrics
        st.markdown("### 📈 Metrics Visualization")
        
        # Create a bar chart of metrics
        metric_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metric_df = metric_df.reset_index().rename(columns={'index':'Metric'})
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_df['Metric'], 
                y=metric_df['Value'], 
                marker_color='#3498DB',
                text=[f'{val:.4f}' for val in metric_df['Value']],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='RAG Evaluation Metrics',
            xaxis_title='Metrics',
            yaxis_title='Score',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Please provide all inputs to evaluate.")

# Footer
st.markdown("""
---
*RAG Evaluation Dashboard - Powered by Advanced NLP Metrics*
""")