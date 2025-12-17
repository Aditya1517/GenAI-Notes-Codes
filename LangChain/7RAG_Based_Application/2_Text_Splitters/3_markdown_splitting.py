# we can split codes aslo

impoer langchain.text_splitter import RecurssiveCharacterTextSplitter

spitter = RecurssiveCharacterTextSplitter.form_language(
    language = Language.Python,
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = spitter.split_text(text)