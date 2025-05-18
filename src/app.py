import gradio as gr
import base64
import os
import glob
from matcher import Matcher, PRODUCT_IMAGES_DIR

# Initialize the Matcher once when the app starts
matcher = Matcher()

def process_query(text_query: str, image_query: str, top_k: int, threshold: float) -> str:
    """Process the query and return formatted results as HTML"""
    # Determine active input type
    if image_query:
        q_emb = matcher.embed_image(image_query)
    else:
        q_emb = matcher.embed_text(text_query)
    
    # Get matches and apply threshold filter
    results = matcher.match(q_emb, top_k=top_k)
    filtered_results = [res for res in results if res['score'] >= threshold]
    
    html_output = ""
    for res in filtered_results:
        pid = res['id']
        score = res['score']
        meta = res.get('metadata', {}) or {}
        
        # Find all image instances using wildcardf
        img_html = ""
        try:
            pattern = os.path.join(PRODUCT_IMAGES_DIR, f"{pid}_*.jpg")
            img_paths = sorted(glob.glob(pattern))
            
            if not img_paths:
                img_html = "<p>No images found for this product</p>"
            else:
                img_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
                for path in img_paths:
                    with open(path, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode()
                        filename = os.path.basename(path)
                        img_html += f'''
                        <div style="flex: 1 1 200px;">
                            <img src="data:image/jpeg;base64,{img_base64}" 
                                 style="max-width: 200px; max-height: 200px; margin: 5px;">
                            <p style="text-align: center; margin: 2px 0;">{filename}</p>
                        </div>
                        '''
                img_html += '</div>'
        except Exception as e:
            img_html = f"<p>Error loading images: {str(e)}</p>"
        
        # Generate metadata HTML
        meta_html = "<ul>"
        for k, v in meta.items():
            meta_html += f"<li><strong>{k}</strong>: {v}</li>"
        meta_html += "</ul>"
        
        # Build result card
        html_output += f"""
        <div style="margin: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
            <div style="display: flex;">
                <div style="flex: 1; min-width: 300px;">
                    {img_html}
                </div>
                <div style="flex: 2; padding-left: 20px;">
                    <h3 style="margin-top: 0;">Product ID: {pid}</h3>
                    <p><strong>Match Score:</strong> {score:.4f}</p>
                    <h4>Metadata:</h4>
                    {meta_html}
                </div>
            </div>
        </div>
        """
    
    return f"<div style='margin: 20px;'>{html_output}</div>" if html_output else "<p>No results found</p>"

def clear_image():
    return None

def clear_text():
    return ""

with gr.Blocks(title="Product Search Demo") as demo:
    gr.Markdown("# 🔍 Product Search with CLIP Embeddings")
    
    with gr.Row():
        with gr.Tabs() as tabs:
            with gr.TabItem("Text Search", id="text_tab") as text_tab:
                text_input = gr.Textbox(label="Enter your search query", lines=2)
            with gr.TabItem("Image Search", id="image_tab") as image_tab:
                image_input = gr.Image(type="filepath", label="Upload query image")
        
        with gr.Column():
            top_k = gr.Slider(1, 5, value=1, step=1, label="Number of results to show")
            top_match_only = gr.Checkbox(label="Retrieve only the top match", value=True)
            threshold = gr.Slider(0, 1, value=0.2, step=0.05, label="Score threshold")

    submit_btn = gr.Button("Search", variant="primary")
    output = gr.HTML(label="Search Results")
    
    # Clear inputs when switching tabs
    text_tab.select(fn=clear_image, inputs=None, outputs=image_input)
    image_tab.select(fn=clear_text, inputs=None, outputs=text_input)
    
    def process_query_with_top_match(text_query: str, image_query: str, top_k: int, top_match_only: bool, threshold: float) -> str:
        """Process the query and return formatted results as HTML"""
        # Override top_k if top_match_only is checked
        if top_match_only:
            top_k = 1

        # Process the query as before
        return process_query(text_query, image_query, top_k, threshold)

    submit_btn.click(
        fn=process_query_with_top_match,
        inputs=[text_input, image_input, top_k, top_match_only, threshold],
        outputs=output
    )

    gr.Markdown("""
    ## User Instructions
    - **Search Modes**:
      - Text Search: Describe products using natural language
      - Image Search: Upload a product photo to find similar items
    - **Parameters**:
      - Number of Results: Controls how many matches to return (1-8)
      - Score Threshold: Filters out matches below this similarity score (0-1) [Reccomend 0.2 for text, 0.6 for image]
    - Results show: Product images, match scores, and metadata
    - Switch between tabs to automatically clear previous input type
    """)

if __name__ == "__main__":
    demo.launch(share=False)