
# import streamlit and other necessary modules
import streamlit as st
import pickle
import string
import numpy as np
import pandas as pd
from textblob import TextBlob
import time

# Setting custom title for the browser tab
st.set_page_config(
    page_title="ReviewInsight Classifier | Samit",
    page_icon="⭐",
    layout="wide",
)


pickle_in = open("sentiment_model.pkl", "rb")
sentiment_model = pickle.load(pickle_in)
pickle_in = open("tfidf.pkl", "rb")
tfidf = pickle.load(pickle_in)
pickle_in = open("stopwords.pkl", "rb")
stopw = pickle.load(pickle_in)

@st.cache_data()

def sentiment(user_input):   
    text= user_input.lower()
    text= text.translate(str.maketrans('', '', string.punctuation))
    text= text.translate(str.maketrans('', '', string.digits))
    text=' '.join(x for x in text.split() if x not in stopw)

    proba = np.round(sentiment_model.predict_proba([text])*100,2)[0]
    classes = ['Negative', 'Neutral', 'Positive']    
    df = pd.DataFrame(data=proba, index=classes, columns=['Percentage'])
    value = df.Percentage.idxmax()

    return df,value

@st.cache_data()

def keywords(user_input):   
    text= user_input.lower()
    text= text.translate(str.maketrans('', '', string.punctuation))
    text= text.translate(str.maketrans('', '', string.digits))
    text=' '.join(x for x in text.split() if x not in stopw)
    
    tf_idf_vector = tfidf.transform([text])
    tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    feature_names = tfidf.get_feature_names_out()
    
    score_val = []
    feature_val = []
    for i,score in sorted_items:
        score_val.append(round(score,3))
        feature_val.append(feature_names[i])
        
    results= {}
    n=len(sorted_items)
    for i in range(min(20,n)):
        results[feature_val[i]]=score_val[i]
        
    imp_words = pd.DataFrame.from_dict(results, orient='index', columns=['score']).reset_index(names='Keywords')
    return imp_words

def main():
    # display the app title
    # st.title("")

    st.markdown(
    """
    <div style="font-family: Trebuchet MS; font-size:35px; text-align:left; color:#ff4b4b;">
        Sentiment Analyzer and Keywords Extractor
    </div>
    """, 
    unsafe_allow_html=True
    )

    # Set the font family for the sidebar titles
    st.sidebar.markdown(
        """
        <style>
            .sidebar-title {
                font-family: 'Trebuchet MS', sans-serif;
                font-size: 25px;
            }
            .sidebar-header {
                font-family: 'Trebuchet MS', sans-serif;
                font-size: 16px;
            }
        </style>
        <div class="sidebar-title"> Hotel Reviews Classifier</div>
        <br/>
        <div class="sidebar-header">It is a web-app built using Streamlit for Sentiment Analysis on Hotel Reviews and to Extract the Elements that are influencing more in forming positive review and improves hotel brand image.</div>
        <br/>

        <div class="sidebar-header">You can checkout any of the hotel reviews just by typing or copy-pasting the reviews into text box displayed and hit Analyze and Extract button to get the Sentiment and Keywords.</div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("\n")
    st.write("\n")
    # get user input as a text area widget
    placeholderText = st.empty()
    user_input = placeholderText.text_area('Type your Reviews here...', key='user_input', height=200)

    # when the analyze button is clicked
    
    if st.button("Analyze and Extract"):
        with st.spinner("Analyzing and Extracting..."):
            time.sleep(3)
            st.markdown(
                """
                <hr style="border-top: 1px solid #fff"/>
                """,
                unsafe_allow_html=True
            )
            df,value = sentiment(user_input)
            st.subheader("Sentiment Scores of the Review are: ")
            st.dataframe(df,width=1500)
            st.subheader("Hence, the Sentiment of the review is: ")
            if value == 'Positive':
                st.success(value)
            elif value == 'Neutral':
                st.warning(value)
            elif value == 'Negative':
                st.error(value)

            st.markdown(
                """
                <hr style="border-top: 1px solid #fff"/>
                """,
                unsafe_allow_html=True
            )
            
            st.subheader("Elements influencing in forming hotel brand image are: ")
            imp_words = keywords(user_input)
            st.dataframe(imp_words, width=1500)

    # Adding "Developed by Samit Dhawal" at the bottom right
    st.markdown(
        """
        <div style="position: fixed; bottom: 0.5%; left: 46%; text-align: left; font-size:15px; font-family: Cursive">
            Developed by <span style="font-family: Brush Script MT; font-size:25px"><a href=https://linkedin.com/in/samitdhawal/ style="color:#ff4b4b" >Samit Dhawal </a></span>
        </div>
        """,
        unsafe_allow_html=True
    )

# run the main function
if __name__ == "__main__":
    main()
