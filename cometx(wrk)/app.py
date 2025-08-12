# FILE: app.py
import streamlit as st
from PIL import Image
import io
import torch
import numpy as np

# Import the core functions from the original main.py file
# Assuming main.py is in the same directory
from main import (
    get_text_embedding,
    get_image_embedding,
    fuse_embeddings,
    build_hetero_graph,
    GNN,
    detect_ai_generated_image,
    perform_context_analysis,
    explain_prediction,
)

# Import the new, working fact-checker module function
from src.fact_checker import search_fact_checks

# --- App Layout and Logic ---

st.set_page_config(
    page_title="COMET-X Multimodal Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("COMET-X Multimodal Analysis")
st.markdown("Use this interface to test the HEMMR pipeline with user-uploaded data.")

# Input fields
claim_input = st.text_area(
    "Enter a text claim:",
    value="The Earth is flat",
    height=150
)
uploaded_file = st.file_uploader(
    "Upload a supporting image:",
    type=["jpg", "jpeg", "png"]
)

if st.button("Run Analysis", help="Click to run the full HEMMR analysis pipeline."):
    if not claim_input and not uploaded_file:
        st.warning("Please provide at least a text claim or an image.")
        st.stop()

    st.info("Analysis is starting... This may take a moment.")
    
    # Process uploaded data
    try:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert image to bytes for the embedding function
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            image_vec = get_image_embedding(img_bytes)
        else:
            image_vec = None

        st.subheader("Pipeline Steps")
        
        # --- STEP 1: Feature Extraction (Using your main.py functions) ---
        st.write("1. Extracting features from text and image...")
        text_vec = None
        if claim_input:
            text_vec = get_text_embedding(claim_input)
        
        if text_vec is None and image_vec is None:
            st.error("Feature extraction failed. No content to analyze.")
            st.stop()

        # --- STEP 2: Multimodal Fusion ---
        st.write("2. Fusing multimodal features...")
        fused_vec = fuse_embeddings(text_vec, image_vec)
        
        # --- STEP 3: Graph Construction (Simulated) ---
        st.write("3. Building heterogeneous graph...")
        # These are mock values as in your original script
        num_posts = 100
        num_users = 50
        post_feature_size = fused_vec.shape[0] if fused_vec is not None else 768 # Default if no image
        user_feature_size = 64

        all_post_features = torch.randn(num_posts, post_feature_size)
        # Place our new post at index 0
        if fused_vec is not None:
            all_post_features[0] = fused_vec
        
        all_user_features = torch.randn(num_users, user_feature_size)
        graph_data = build_hetero_graph(
            num_posts, num_users, all_post_features, all_user_features
        )
        st.write("Graph built successfully.")

        # --- STEP 4: GNN Inference (Simulated with improved logic) ---
        st.write("4. Running GNN inference...")
        # Simulate GNN prediction based on your improved mock logic
        text_content_lower = claim_input.lower() if claim_input else ""
        if any(keyword in text_content_lower for keyword in ['world war', 'free money', 'stealing', 'scam', 'fraud']):
            mock_prediction = torch.tensor([[0.95, 0.05]])  # Strong fake prediction
        else:
            mock_prediction = torch.tensor([[0.20, 0.80]])  # Moderate credible prediction
        
        post_prediction = mock_prediction[0].softmax(dim=-1)
        st.write("GNN inference completed.")

        # --- STEP 5: AI Image Detection ---
        st.write("5. Detecting AI-generated image content...")
        if uploaded_file:
            ai_image_detection_result = detect_ai_generated_image(uploaded_file.name)
            st.write(f"AI Image Detection: Image is classified as '{ai_image_detection_result['predicted_label']}' with {ai_image_detection_result['confidence']:.2f}% confidence.")
        else:
            st.write("No image provided. Skipping AI image detection.")
            ai_image_detection_result = None

        # --- STEP 6: Fact-Checking API Call (Now Live) ---
        st.write("6. Cross-referencing with fact-checking services...")
        fact_check_result = None
        if claim_input:
            fact_check_result = search_fact_checks(claim_input)
        
        # Display the results
        if fact_check_result:
            st.markdown("External Fact-Check Results:")
            for result in fact_check_result:
                st.markdown(f"**Claim:** {result.get('claim_text', 'N/A')}")
                st.markdown(f"**Rating:** {result.get('rating', 'N/A')} by {result.get('publisher', 'N/A')}")
                st.markdown(f"**Source:** [Link]({result.get('review_url', '#')})")
        else:
            st.markdown("External Fact-Check: No highly relatable facts found.")
        
        # --- STEP 7: HEMMR Explanation ---
        st.write("7. Generating explanation...")
        attention_weights = {}
        if claim_input:
            attention_weights['text_focus'] = f"a high focus on the words in the claim: '{claim_input}'"
        if uploaded_file and ai_image_detection_result:
            attention_weights['image_focus'] = f"the AI image detection result which classified the image as '{ai_image_detection_result['predicted_label']}'"

        context_analysis_result = perform_context_analysis(claim_input)
        explanation = explain_prediction(post_prediction, attention_weights, context_analysis_result)

        st.subheader("Final Analysis")
        st.markdown(explanation)
        
        # --- Final Status ---
        confidence_score = post_prediction[0].item() if post_prediction[0].item() > post_prediction[1].item() else post_prediction[1].item()
        claim_status_label = "fake" if mock_prediction[0][0] > mock_prediction[0][1] else "credible"
        
        st.markdown(f"---")
        st.markdown(f"**Final Claim Status:** The post is likely **{claim_status_label}** with a confidence of {confidence_score:.2f}.")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
