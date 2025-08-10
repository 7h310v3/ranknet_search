"""
Streamlit UI connected to RankNet Search API
Run: streamlit run ui/streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import json

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="RankNet Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .search-result {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .search-result:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_api_stats():
    """Get stats from API"""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
    return None

def search_api(query, top_k=10):
    """Search using API"""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": top_k}
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
    return None

def log_click(query_id, doc_id, position, dwell_time=30.0):
    """Log click to API"""
    try:
        requests.post(
            f"{API_URL}/feedback",
            json={
                "query_id": query_id,
                "doc_id": doc_id,
                "position": position,
                "dwell_time": dwell_time
            }
        )
        return True
    except Exception as e:
        st.error(f"API Error: {str(e)}")
    return False

def train_model():
    """Trigger model training"""
    try:
        with st.spinner("Training model... This may take a minute."):
            response = requests.post(
                f"{API_URL}/train",
                json={"epochs": 1000, "batch_size": 32}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
    return None
    # This function is not implemented in our API
    return []

# Main UI
st.title("🔍 RankNet Search Engine")
st.markdown("#### Learning to Rank with RankNet")

# Check API connection
try:
    api_status = "🟢 Connected" if requests.get(f"{API_URL}/health").status_code == 200 else "🔴 Disconnected"
except:
    api_status = "🔴 Disconnected"

st.sidebar.markdown(f"**API Status:** {api_status}")

# Search Section
search_container = st.container()
with search_container:
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Search for articles",
            value=st.session_state.last_query,
            placeholder="Try: machine learning, python, data science...",
            key="search_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Filters in sidebar
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    # Number of results
    num_results = st.slider("Number of results", 5, 50, 10)
    
    # Advanced options
    with st.expander("Advanced Options"):
        show_score = st.checkbox("Show relevance scores", value=True)
        show_preview = st.checkbox("Show content preview", value=True)
        enable_click_tracking = st.checkbox("Track clicks", value=True)
    
    # Train model button
    if st.button("🧠 Train Model", use_container_width=True):
        training_result = train_model()
        if training_result:
            st.success("Model training completed!")
            st.json(training_result)

# Perform search
if (query and search_button) or (query and query != st.session_state.last_query and len(query) > 2):
    st.session_state.last_query = query
    
    # Show loading spinner
    with st.spinner(f"Searching for '{query}'..."):
        start_time = time.time()
        
        # Call API
        search_results = search_api(
            query=query,
            top_k=num_results
        )
        
        search_time = (time.time() - start_time) * 1000
    
    if search_results and search_results.get('results'):
        # Add to search history
        st.session_state.search_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'results': len(search_results['results'])
        })
        
        # Display results header
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Found {search_results['total']} results")
        with col2:
            st.metric("Search Time", f"{search_time:.1f}ms")
        
        # Display results
        for i, result in enumerate(search_results['results']):
            with st.container():
                st.markdown(f"### {i+1}. {result['title']}")
                if show_score:
                    st.caption(f"Score: {result['score']:.4f}")
                
                # Metadata row
                meta_cols = st.columns(4)
                with meta_cols[0]:
                    st.caption(f"👤 {result['author']}")
                with meta_cols[1]:
                    st.caption(f"📅 {result['timestamp'][:10]}")
                with meta_cols[2]:
                    st.caption(f"👁️ {result.get('views', 0):,} views")
                with meta_cols[3]:
                    url_display = result['url']
                    if len(url_display) > 40:
                        url_display = url_display[:40] + "..."
                    st.caption(f"🔗 {url_display}")
                
                # Content preview
                if show_preview:
                    st.markdown("---")
                    st.write(result['content'])
                
                # Action buttons
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button(f"📖 Read", key=f"read_{result['doc_id']}"):
                        if enable_click_tracking:
                            success = log_click(
                                query_id=search_results.get('query_id', 'query_' + str(int(time.time()))),
                                doc_id=result['doc_id'],
                                position=result['rank'],
                                dwell_time=30.0
                            )
                            if success:
                                st.success("Click tracked!")
                        
                        st.markdown(f"[Open Article]({result['url']})")
                
                st.markdown("---")
    else:
        st.warning(f"No results found for '{query}'. Try different keywords.")

# Statistics Section
with st.sidebar:
    st.header("📊 Statistics")
    
    stats = get_api_stats()
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", f"{stats.get('num_documents', 0):,}")
        with col2:
            st.metric("Queries", f"{stats.get('num_queries', 0):,}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clicks", f"{stats.get('num_clicks', 0):,}")
        with col2:
            st.metric("Model", "✓" if stats.get('has_model', False) else "✗")

# Analytics Dashboard
if st.checkbox("📊 Show Search History"):
    st.header("Search History")
    
    # Search history analysis
    if st.session_state.search_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.search_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Search volume over time
            fig = px.line(
                history_df,
                x='timestamp',
                y='results',
                title='Search Results Over Time',
                labels={'results': 'Number of Results', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Popular queries
            query_counts = history_df['query'].value_counts().head(10)
            if not query_counts.empty:
                fig = px.bar(
                    x=query_counts.values,
                    y=query_counts.index,
                    orientation='h',
                title='Top Search Queries',
                labels={'x': 'Count', 'y': 'Query'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Stats summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Searches", len(history_df))
        with col2:
            st.metric("Unique Queries", history_df['query'].nunique())
        with col3:
            avg_results = history_df['results'].mean()
            st.metric("Avg Results", f"{avg_results:.1f}")
        with col4:
            st.metric("Session Time", f"{(datetime.now() - history_df['timestamp'].min()).seconds // 60} min")

# Our API doesn't support trending articles yet

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit + FastAPI | Powered by RankNet</p>
        <p>Learning to Rank • BM25 • TF-IDF • Feature Engineering</p>
    </div>
    """,
    unsafe_allow_html=True
)