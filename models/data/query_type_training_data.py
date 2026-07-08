"""Labeled dataset for the query-type classifier.

Three labels, describing what kind of answer a question needs rather than
whether it's about the document at all (that's `ml/intent_classifier.py`'s
job):

- factual: a single fact/value can answer it directly from one chunk
- summarization: needs condensing a section/document into a shorter answer
- comparison: needs relating two or more things to each other

`ask_question` uses this to skip the LLM entirely for factual questions
that retrieval answers with high confidence, since composing a sentence
around one clearly-retrieved fact rarely needs generation.
"""

TRAINING_EXAMPLES = [
    ("what is the effective date of this agreement", "factual"),
    ("who is the vendor named in the contract", "factual"),
    ("what is the total contract value", "factual"),
    ("what is the notice period for termination", "factual"),
    ("what governing law applies to this contract", "factual"),
    ("when is the payment due", "factual"),
    ("what currency are payments made in", "factual"),
    ("how long does the agreement last", "factual"),
    ("who signed the contract on behalf of the client", "factual"),
    ("what is the late payment penalty", "factual"),
    ("what is the insurance coverage amount required", "factual"),
    ("what is the renewal term", "factual"),
    ("what jurisdiction handles disputes", "factual"),
    ("what is the maximum liability cap", "factual"),
    ("what date was this signed", "factual"),
    ("how many pages does the report have", "factual"),
    ("what is the author of this document", "factual"),
    ("what is the minimum order quantity", "factual"),
    ("what deadline is set for the first milestone", "factual"),
    ("what percentage discount is offered", "factual"),
    ("summarize the vendor's obligations in this document", "summarization"),
    ("summarize the termination clause", "summarization"),
    ("give me a summary of the risks section", "summarization"),
    ("summarize the payment terms section", "summarization"),
    ("summarize the methodology described in the study", "summarization"),
    ("summarize the conclusions of the report", "summarization"),
    ("summarize the confidentiality obligations", "summarization"),
    ("provide a summary of the warranty section", "summarization"),
    ("summarize the scope of work", "summarization"),
    ("summarize the key findings from the quarterly report", "summarization"),
    ("give me an overview of this document", "summarization"),
    ("what is this contract about in a few sentences", "summarization"),
    ("summarize the entire agreement briefly", "summarization"),
    ("condense the executive summary into two sentences", "summarization"),
    ("tl;dr this document for me", "summarization"),
    ("how do the payment terms in section 2 compare to section 5", "comparison"),
    ("what is the difference between the vendor's and client's obligations", "comparison"),
    ("compare the termination notice periods for each party", "comparison"),
    ("how does this year's revenue compare to last year's", "comparison"),
    ("compare the liability caps in the two schedules", "comparison"),
    ("what are the differences between the standard and premium tiers", "comparison"),
    ("compare the confidentiality terms in the agreement versus the appendix", "comparison"),
    ("how do the renewal terms differ from the initial term", "comparison"),
    ("compare section 3 obligations with section 7", "comparison"),
    ("what changed between the old contract and the new one", "comparison"),
    ("which vendor offers better payment terms, A or B", "comparison"),
    ("contrast the warranty periods in both agreements", "comparison"),
]
