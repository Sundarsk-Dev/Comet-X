Overall Ideas Implemented:
The COMET-X project is a sophisticated, full-stack prototype designed to combat misinformation by analyzing social media content using both text and images. The core idea is to create a robust and transparent pipeline that provides not just a verdict, but also a clear, human-readable explanation for its decision.

The system works by following a series of integrated steps:

Multimodal Feature Extraction: The process begins with advanced feature extraction. For text, it utilizes a powerful pre-trained RoBERTa model to generate detailed text embeddings. For images, it employs a fine-tuned Vision Transformer (ViT) model to convert visual content into rich image embeddings. This allows the system to understand the semantic meaning of both text and visual elements.

Multimodal Fusion: The individual text and image embeddings are then combined into a single, comprehensive feature vector. This fusion is crucial for capturing the interplay between the two modalities, as misinformation often relies on a mismatch between a sensational headline and a misleading image.

HEMMR Logic: The HEMMR component is the project's central intelligence. It takes the fused embeddings and performs a multi-faceted analysis, which includes:

Contextual Analysis: It scans the text for keywords and patterns indicative of common misinformation tropes, such as financial scams, conspiracy theories, or health hoaxes. This is the source of the system's "skepticism."

AI Image Detection: It incorporates a mock function to simulate the detection of AI-generated images, which is a key indicator of fabricated content.

Fact-Checking Prioritization: It integrates an external fact-checking API. The system is designed to prioritize a definitive "Fake" verdict from this API, using it to strengthen its own conclusion. If no definitive external fact-check is available, the HEMMR model's internal analysis takes precedence.

Simulated GNN Integration: To demonstrate how this tool would work in a real-world social network, we implemented a simulated GNN. This GNN models how misinformation spreads across users and posts, with the intention of showcasing how the system could eventually identify the source and trajectory of fake news.
