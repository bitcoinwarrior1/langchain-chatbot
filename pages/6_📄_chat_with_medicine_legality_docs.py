import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your documents (Basic RAG)')
st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class CustomDataChatbot:

    def __init__(self):
        self.openai_model = utils.configure_openai()

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, directory):
        # Load documents from the specified directory
        docs = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask me anything!")
        qa_chain = self.setup_qa_chain('../docs')

        utils.display_msg(user_query, 'user')

        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            result = qa_chain.invoke(
                {"question":user_query},
                {"callbacks": [st_cb]}
            )
            response = result["answer"]
            st.session_state.messages.append({"role": "assistant", "content": response})

            # to show references
            for idx, doc in enumerate(result['source_documents'],1):
                filename = os.path.basename(doc.metadata['source'])
                page_num = doc.metadata['page']
                ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                with st.popover(ref_title):
                    st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()