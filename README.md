## PSU CS 445/545 Machine Learning Group Project
<hr />
<div align="center">

### Topic Modeling of COVID-19 Tweets
_Brandon Le, Yves Wienecke, Angie McGraw, Tu Vu, Keaton Kraiger_
</div><br />

The COVID-19 pandemic is a major world event, which has had an extensive impact on the ebb and flow of society. The pandemic has significantly disrupted the manner in which people communicate and learn. As people turn to social media to voice their thoughts and emotions regarding the new ‘normal’ of everyday life, websites like Facebook, Twitter, and Instagram reflect the turbulence and confusion during this stressful time. Given the sheer magnitude and disorganized nature of the data that are exposed by these websites, our group is interested in applying machine learning techniques to identify patterns and make sense of the madness. We hope to utilize natural language processing (NLP) to learn about how people are communicating during the pandemic. Moreover, we believe this project will help elucidate what people are talking about and how these topics may change by geographic location, time, or both. Our motivation for this project stems primarily from the significant effect that COVID-19 has had on each of us individually. In addition, some of us have a connection to the medical field and see this situation as an opportunity to interweave experience in the fields of medicine and computer science. Although we are isolated from the community and living in our new realities, we are nevertheless interested in understanding the new realities of others.

This repository is structured to separate our **research, code, and dataset**. The `papers/` directory contains the project proposal and research paper associated with this project. The `code/` directory contains all of the python scripts used for downloading and preparing the dataset, running the topic modeling algorithms, and visualizing the results. The `data/` directory is the location in which the dataset will be stored, separated by city and date. This repository only shows examples of the structure of the data and will not include the complete datasets.
<br /><br />

#### TODO
-  **Prepare the dataset**
    - Figuring out how to download the dataset
    - Understand the format of the dataset and the format of the coordinates (i think it's [latitude][longitude], but I'm not sure what the bounds of these numbers are)
    - Decide which geographic coordinates we would like to focus on (probably create enums that connect some city name to a coordinate pair, ex. PORTLAND = [latitude][longitude])
    - Create a script/function that will filter our dataset to only the selected coordinates
    - Create a script /function that will 'hydrate' the filtered dataset and give us .json files
    - Create a function that will select tweets written in English only
    - Analyze the data and see what we must do to for pre-processing (cleaning, normalizing, removing non-ASCII chars, n-grams, etc.)
- **Prepare the models**
    - Understand the Gensim library, topic modeling, and topic modeling algorithms
    - Download the pre-trained gensim models
    - (?) optimize the hyperparameter for the # of topics
    - Train the gensim models on each partitioned set of our data to get topic distributions w/ associated keywords/keyphrases
    - Visualize our results in tabular & graphical format
    - Use KLD measure or some other topic modeling metric to quantitatively compare the divergence/similarity of topics btwn each data partition
    - Qualitatively compare the results of each topic modeling algorithms and our results
- **Write the paper**
    - Write the final paper
<br />

#### References

