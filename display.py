import streamlit as st
from main import generate, parse_text

def main():
    st.title("Text Generation App")

    user_prompt = st.text_input("Enter a prompt:", "bread,egg,milk")

    if st.button("Generate Text"):
        generated_text = generate(user_prompt)
        st.subheader("Generated Text:")
        parse_text(generated_text)

if __name__ == "__main__":
    main()
