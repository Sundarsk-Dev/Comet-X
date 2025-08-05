import torch

def explain_prediction(prediction, attention_weights):
    """
    Generates a human-readable explanation from a model's output.
    """
    predicted_class = torch.argmax(prediction).item()
    confidence_score = prediction[predicted_class].item()

    if predicted_class == 0:
        prediction_text = "misinformation"
    else:
        prediction_text = "credible information"

    explanation = (
        f"Based on our analysis, this post is likely to be {prediction_text} "
        f"with a confidence score of {confidence_score:.2f}. "
        f"The model's decision was primarily influenced by: "
        f"1. **Text Content**: It detected {attention_weights['text_focus']}. "
        f"2. **Visual Content**: It found {attention_weights['image_focus']}. "
        f"3. **Propagation Pattern**: It focused on {attention_weights['network_focus']}."
    )

    return explanation