# ADR Binary Classifier README #
## Developer version

This repository contains the Adverse Drug Reaction (ADR) classification scripts released by the DIEGO LAB at the Department of Biomedical Informatics, Arizona State University.

The ADR classification scripts here are developed using Python and sklearn. Code for feature generation can be added to featureextractionutilities/FeatureExtractionUtilities

Data for the binary classification task can be found at: http://diego.asu.edu/Publications/ADRClassify.html

We have attempted to keep the code simple and easily customizable for other social media text classification tasks. 


###Brief content descriptions ###

* We use support vector machines for the classification task. 
* Features currently present:
- n-grams
- synsets
- change phrases
- ADR topic scores
- ADR lexicon scores
- several sentiment scores
- basic structural features
- word cluster features


### Publications discussing our classification work ###
We have used our social media text classifier for a variety of classification tasks in addition to ADR classification. The following publications discuss our approaches:


> Sarker A, Gonzalez G; Portable Automatic Text Classification for Adverse Drug Reaction Detection via Multi-corpus Training, Journal of Biomedical Informatics, 2015 Feb;53:196-207. doi: 10.1016/j.jbi.2014.11.002. Epub 2014 Nov 8.
>(resources for feature extraction for this task can be found at: http://diego.asu.edu/Publications/ADRClassify.html)

###Please cite the above paper if you are using our scripts###

Note: the scripts here do not contain UMLS CUIs and Semantic Types. In the paper, these were generated using MetaMap. Users of the script can use MetaMap or other tools to generate these information and incorporate the features in the same way as the synset or topic text feature. Also, the experiments discussed in the paper were performed using Weka, but we have decided to share this alternative python implementation due to ease of use/modification. 

Note 2: We have added several more sentiment scores.

Note 3: We have added a word cluster feature (discussed in the publications below) and we have found this feature to be useful in text classification.

-> Sarker A, O'Connor K, Ginn R, Scotch M, Smith K, Malone D, Gonzalez G.; Social media mining for toxicovigilance: automatic monitoring of prescription medication abuse from Twitter, Drug Safety, 2016 Mar;39(3):231-40. doi: 10.1007/s40264-015-0379-4.
(data for this study available at: http://diego.asu.edu/Publications/DrugAbuse_DrugSafety.html)

-> Sarker A, Gonzalez G. DiegoLab16 at SemEval-2016 Task 4: Sentiment Analysis in Twitter using Centroids, Clusters, and Sentiment Lexicons. Submitted to SemEval 2016 on 5th March.
(the current implementation and classes reflect some of the modifications used for this task)

### We will periodically add/remove modules to this classifier
### Note: Please be considerate with the various classification resources. Please adhere to the license agreements of the individual items. This is the developer version and so it's a little difficult to ensure that all the license details are up-to-date on a daily basis.