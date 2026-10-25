def select_view(at, page_index, label):
    """Select a tracked workspace tab in Streamlit's application test harness."""
    at.session_state['view_'+str(page_index)] = label
    return at.run()
