# my_lesk
My slightly modified version of Lesk's algorithm for Word Sense Disambiguation. (Really my solution to Question 2 of COMP 599's Assignment #3, at McGill.)

The repo consists of the following:
+ multilingual-all-words.en.xml - The publicly available data set of SemEval 2013 Shared Task #12 (Navigli and Jurgens, 2013). (You can find more information on the data set [here](https://www.cs.york.ac.uk/semeval-2013/task12/).)
+ wordnet.en.key - The corresponding gold standard senses.
+ loader.py - A loader for the aforementioned data.
+ lesk.py - My implementation of Lesk's algorithm with some added special sauce. The main method compares my implementation against a most-frequent sense baseline and NLTK's implementation of Lesk's algorithm.

If you have any questions, concerns, or comments, shoot me an email at (my GitHub username with - replaced with .)@mail.mcgill.ca. Enjoy!

(License: MIT)
