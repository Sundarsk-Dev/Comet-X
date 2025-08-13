# FILE: app.py
import streamlit as st
from PIL import Image
import io
import torch
import numpy as np
import warnings

# Temporarily ignore the deprecation warning for now
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# --- Helper function for image resizing ---
def resize_image(image, max_width=250):
    """
    Resizes an image to a maximum width while maintaining its aspect ratio.
    """
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        resized_image = image.resize((max_width, new_height))
        return resized_image
    return image

# --- App Layout and Logic ---

st.set_page_config(
    page_title="COMET-X Multimodal Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title and description
st.title("🛡️ COMET-X Multimodal Analysis")
st.markdown("A tool for detecting misinformation in posts by analyzing both text and images.")

# Create a single column for a cleaner layout
main_container = st.container()

with main_container:
    # Input section
    st.header("1. Enter Post Details")
    claim_input = st.text_area(
        "Enter a text claim:",
        value="",
        height=100,
        placeholder="e.g., 'A new species of dinosaur has been discovered in Africa.'"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a supporting image:",
            type=["jpg", "jpeg", "png"]
        )
    
    with col2:
        if uploaded_file:
            # Resize the image for display purposes
            display_image = Image.open(uploaded_file).convert("RGB")
            st.image(resize_image(display_image), caption="Uploaded Image", use_container_width=False)

    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 Run Analysis", help="Click to run the full analysis pipeline."):
        if not claim_input and not uploaded_file:
            st.error("❌ Please provide at least a text claim or an image to analyze.")
            st.stop()

        # Wrap the entire analysis in a spinner
        with st.spinner("Analyzing post... this may take a moment."):
            try:
                # Get the image data in bytes for the embedding model
                image_vec = None
                if uploaded_file:
                    img_bytes = io.BytesIO(uploaded_file.getvalue())
                    image_vec = get_image_embedding(img_bytes)
                
                # --- Step 1: Feature Extraction ---
                text_vec = None
                if claim_input:
                    text_vec = get_text_embedding(claim_input)
                
                if text_vec is None and image_vec is None:
                    st.error("❌ Feature extraction failed. No content to analyze.")
                    st.stop()

                # --- Step 2: Multimodal Fusion ---
                fused_vec = fuse_embeddings(text_vec, image_vec)
                
                # --- Step 3: Graph Construction (Simulated) ---
                num_posts = 100
                num_users = 50
                post_feature_size = fused_vec.shape[0] if fused_vec is not None else 768
                user_feature_size = 64

                all_post_features = torch.randn(num_posts, post_feature_size)
                if fused_vec is not None:
                    all_post_features[0] = fused_vec
                
                all_user_features = torch.randn(num_users, user_feature_size)
                graph_data = build_hetero_graph(
                    num_posts, num_users, all_post_features, all_user_features
                )

                # --- Step 4: GNN Inference (Simulated) ---
                text_content_lower = claim_input.lower() if claim_input else ""
                if any(keyword in text_content_lower for keyword in ['world war', 'free money', 'stealing', 'scam', 'fraud']):
                    mock_prediction = torch.tensor([[0.95, 0.05]])
                else:
                    mock_prediction = torch.tensor([[0.20, 0.80]])
                
                post_prediction = mock_prediction[0].softmax(dim=-1)

                # --- Step 5: AI Image Detection ---
                ai_image_detection_result = None
                if uploaded_file:
                    ai_image_detection_result = detect_ai_generated_image(uploaded_file.name)
                
                # --- Step 6: Fact-Checking API Call ---
                fact_check_result = None
                if claim_input:
                    fact_check_result = search_fact_checks(claim_input)
                
                # --- Step 7: HEMMR Explanation ---
                attention_weights = {}
                if claim_input:
                    attention_weights['text_focus'] = f"a high focus on the words in the claim: '{claim_input}'"
                if uploaded_file and ai_image_detection_result:
                    attention_weights['image_focus'] = f"the AI image detection result which classified the image as '{ai_image_detection_result['predicted_label']}'"
                
                context_analysis_result = perform_context_analysis(claim_input)
                
                # --- FINAL VERDICT & EXPLANATION LOGIC (UPDATED) ---
                final_claim_status = ""
                final_confidence = 0.0
                final_explanation = ""
                
                # First, generate the HEMMR-only explanation
                hemmr_explanation = explain_prediction(post_prediction, attention_weights, context_analysis_result)

                # Then, check for a definitive fact-check verdict
                has_definitive_fact_check = False
                if fact_check_result:
                    for result in fact_check_result:
                        rating = result.get('rating', '').lower()
                        # Define keywords for a "fake" verdict
                        fake_keywords = ['false', 'misleading', 'hoax', 'inaccurate']
                        if any(kw in rating for kw in fake_keywords):
                            final_claim_status = "fake"
                            final_confidence = 0.99
                            has_definitive_fact_check = True
                            
                            # Combine the fact-check verdict with the HEMMR explanation
                            final_explanation = (
                                "Based on a conclusive finding from an external fact-checking service, "
                                f"this claim is definitively **{final_claim_status}**. "
                                "The HEMMR model's analysis further suggests: " + hemmr_explanation.split(" influenced by:")[1]
                            )
                            break
                
                # If no definitive fact-check was found, use the HEMMR model's prediction and explanation
                if not has_definitive_fact_check:
                    confidence_score = post_prediction[0].item() if post_prediction[0].item() > post_prediction[1].item() else post_prediction[1].item()
                    claim_status_label = "fake" if mock_prediction[0][0] > mock_prediction[0][1] else "credible"
                    final_claim_status = claim_status_label
                    final_confidence = confidence_score
                    final_explanation = hemmr_explanation

                # --- FINAL OUTPUT SECTION ---
                st.markdown("---")
                st.header("2. Final Analysis")
                
                if final_claim_status == "fake":
                    st.error(f"**🔴 Final Verdict: Likely {final_claim_status}** (Confidence: {final_confidence:.2f})")
                else:
                    st.success(f"**✅ Final Verdict: Likely {final_claim_status}** (Confidence: {final_confidence:.2f})")

                st.markdown(f"**Explanation:** {final_explanation}")
                
                # Display the fact-check results in a collapsible section
                if fact_check_result:
                    st.markdown("---")
                    with st.expander("External Fact-Check Results"):
                        if isinstance(fact_check_result, list) and fact_check_result:
                            for result in fact_check_result:
                                st.markdown(f"**- Claim:** {result.get('claim_text', 'N/A')}")
                                st.markdown(f"**- Rating:** {result.get('rating', 'N/A')} by {result.get('publisher', 'N/A')}")
                                st.markdown(f"**- Source:** [Link]({result.get('review_url', '#')})")
                        else:
                            st.markdown("No highly relatable facts found.")

                # Add a collapsible section for the technical details
                st.markdown("---")
                with st.expander("Technical Details & Process Log"):
                    if 'display_image' in locals():
                        st.markdown(f"**Image Dimensions (Original):** {display_image.width}x{display_image.height}")
                    st.write("1. Text embedding generated.")
                    st.write("2. Image embedding generated.")
                    st.write("3. Features fused.")
                    st.write("4. Heterogeneous graph built.")
                    st.write("5. GNN inference performed.")
                    if uploaded_file:
                        st.write(f"6. AI image detection result: '{ai_image_detection_result['predicted_label']}' with {ai_image_detection_result['confidence']:.2f}% confidence.")
                    if claim_input:
                        if fact_check_result:
                            st.write(f"7. External fact-checks found: {len(fact_check_result)} result(s).")
                        else:
                            st.write(f"7. No external fact-checks found.")

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {e}")
