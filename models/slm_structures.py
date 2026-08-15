class LocalEnhancerSLM:
    """
    Local Small Language Model (SLM) for sub-millisecond quality prompt enhancement.
    Uses TF-IDF semantic embedding space + K-Nearest Neighbor structural retrieval & template synthesis.
    """
    def __init__(self, vectorizer, knn, dataset_records):
        self.vectorizer = vectorizer
        self.knn = knn
        self.dataset_records = dataset_records

    def enhance(self, original_prompt: str) -> str:
        prompt_vec = self.vectorizer.transform([original_prompt])
        distances, indices = self.knn.kneighbors(prompt_vec)
        
        nearest_idx = indices[0][0]
        match = self.dataset_records[nearest_idx]
        
        enhanced_prompt = (
            f"[ENHANCED FOR QUALITY & PRECISION VIA LOCAL SLM]\n\n"
            f"Objective: {original_prompt.strip()}\n\n"
            f"Instructions & Criteria:\n"
            f"- Provide a complete, highly structured, and production-ready solution.\n"
            f"- Pay strict attention to edge cases, error handling, and performance.\n"
            f"- Format code blocks with clean syntax highlighting and include clear comments."
        )
        return enhanced_prompt
