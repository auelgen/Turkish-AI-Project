WILL IMPROVE THIS SECTION SOON..
# Turkish-AI-Project
Turkish artificial intelligence project, NLP and ML algorithms were used.

This repository showcases a natural language processing (NLP) application tailored for Turkish. The project integrates machine learning and AI-powered tools to classify and analyze user input, generate topic summaries, and perform web research while storing results in a PostgreSQL database for further use.
## Features
Text Classification: Using a trained machine learning model and categorizes input text into primary and secondary topics.
Summarization: Leverages advanced NLP techniques to extract meaningful summaries from Turkish text.
Web Search Integration: Automatically fetches search results based on the analyzed topics.
Database Storage: Saves the results (topics and research outputs) into a PostgreSQL database for easy retrieval and analysis.

## Technologies Used
NLP: Spacy with pyTextRank for summarization and text processing.
Machine Learning: Scikit-learn for TD-IDF model training.
Database: PostgreSQL for structured data storage.
Web Scraping: BeautifulSoup for integrating Google search results.

## Step-by-Step Instructions:
1-) Search for any query/question in Turkish language. For seeing results you can write 'q' and press enter. Or you can add more queries. If you do so, program simply will detect what the topic and categories are and find a single result.
![1](https://github.com/user-attachments/assets/61407238-caf7-4c72-a7a0-8039a377cb96)

2-) Turkish newspaper clippings and some sample AI prompts were classified according to their subjects and the model was trained. The program determines the subject of the text you enter. You can see the performance of the trained model.
![2](https://github.com/user-attachments/assets/0cf72962-e57b-497f-8233-d13771496d7f)

3-) The program successfully determined main and sub-topics. The topic summary was found and a prompt  ready for search. The results from the search engine were listed. 
![3](https://github.com/user-attachments/assets/0c1c20f9-44a1-4c26-a978-8cf0d2144661)

4-) Main-topic,sub-topic and output saved to exist database via Postgresql.
![4](https://github.com/user-attachments/assets/425d8662-bd84-4500-a942-e406178a64b9)





