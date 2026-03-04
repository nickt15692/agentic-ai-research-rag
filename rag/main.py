import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def main():
    print("PDF Q&A Chatbot")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Step 1 - Load PDFs from ./documents folder
    loader = DirectoryLoader(
        "./documents",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2 - Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Step 3 - Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created")

    # Step 4 - Create modern RAG chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, just say you don't know.

    Context: {context}
    Question: {question}
    Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("QA chain ready")

    # Step 5 - Interactive Q&A loop
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        # Get answer
        result = qa_chain.invoke(question)
        print(f"\nBot: {result}\n")

        # Print sources
        docs = retriever.invoke(question)
        if docs:
            print("Sources:")
            for doc in docs[:2]:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                print(f"  {source} (Page {page})")


if __name__ == "__main__":
    main()
