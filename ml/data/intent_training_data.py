"""Small seed dataset for the query-intent classifier.

Three intents:
- greeting: small talk / pleasantries, answered without touching the LLM
- chit_chat: general questions unrelated to the uploaded documents
- document_question: questions that should go through the RAG pipeline

This is intentionally small — it is meant to catch the obvious cases
cheaply, not to be a general-purpose intent model. Extend it with real
traffic once the app has usage.
"""

TRAINING_EXAMPLES = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hi there", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("thanks", "greeting"),
    ("thank you", "greeting"),
    ("thanks a lot", "greeting"),
    ("bye", "greeting"),
    ("goodbye", "greeting"),
    ("who are you", "chit_chat"),
    ("what is your name", "chit_chat"),
    ("how are you", "chit_chat"),
    ("what's the weather today", "chit_chat"),
    ("tell me a joke", "chit_chat"),
    ("what can you do", "chit_chat"),
    ("are you a real person", "chit_chat"),
    ("what time is it", "chit_chat"),
    ("do you like music", "chit_chat"),
    ("what does this document say about the termination clause", "document_question"),
    ("summarize section 3 of the report", "document_question"),
    ("what are the payment terms mentioned in the contract", "document_question"),
    ("explain the methodology described in the paper", "document_question"),
    ("what is the total revenue reported in the financial statement", "document_question"),
    ("list the key findings from the uploaded document", "document_question"),
    ("who is the author of this document", "document_question"),
    ("what date was this agreement signed", "document_question"),
    ("what are the risks mentioned in the document", "document_question"),
    ("give me a summary of the uploaded pdf", "document_question"),
    ("what obligations does the vendor have according to the contract", "document_question"),
    ("how many pages does the report have", "document_question"),
    ("what is the conclusion of the study", "document_question"),
    ("does the document mention any deadlines", "document_question"),
    ("what are the key metrics in this quarterly report", "document_question"),
]
