from langchain.text_splitter import CharacterTextSplitter

text = """
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
"""

spitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = spitter.split_text(text)

print(result)

# we can load the document and can perform text splitting on it
# and to perform it we can use split_document