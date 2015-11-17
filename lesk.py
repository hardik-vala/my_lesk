"""
My implementation of Lesk's algorithm (for Question 2 of COMP 599's Assignment
#3).
"""

import argparse
import csv
import numpy as np
import random
import sys

from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.wsd import lesk
from string import punctuation

from loader import load_instances, load_key


def to_sense_key(syn):
	"""
	Returns the sense key for the given synset.
	
	@param syn - Synset
	@return Corresponding sense key as string
	"""

	return syn.lemmas()[0].key()


def intersection(set1, set2):
	"""
	Calculates the intersection size between two sets, used to compute overlap
	between a word context and a definition's signature.

	@param set1 - First set
	@param set2 - Second set
	@return Intersection size
	"""

	return len(set(set1) & set(set2))


def jaccards_index(set1, set2):
	"""
	Calculates Jaccard's index between two sets, used to compute overlap
	between a word context and a definition's signature.

	@param set1 - First set
	@param set2 - Second set
	@return Jaccard's index
	"""

	set1 = set(set1)
	set2 = set(set2)
	return float(len(set1 & set2)) / len(set1 | set2)


def lcs(l1, l2):
	"""
	Calculates the longest common subsequence between lists of tokens l1 and
	l2, used to compute overlap between a word context and a definition's
	signature.

	@param set1 - First list
	@param set2 - Second list
	@return Length of the longest common subsequence
	"""

	m, n = len(l1), len(l2)
	trellis = np.zeros((m + 1 )* (n + 1)).reshape(m + 1, n + 1)

	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if l1[i - 1] == l2[j - 1]:
				trellis[i, j] = 1 + trellis[i - 1, j - 1]
			else:
				trellis[i, j] = max(trellis[i - 1, j], trellis[i, j - 1])

	return trellis[m, n]


class Data(object):
	"""
	Manages the data. (Expects data files to be in the current directory.)
	"""

	def __init__(self):
		data_f = 'multilingual-all-words.en.xml'
		key_f = 'wordnet.en.key'

		self.dev_instances, self.test_instances = load_instances(data_f)
		self.dev_key, self.test_key = load_key(key_f)


class WSDisambiguator(object):
	"""
	Interface for Word Sense Disambiguation algorithms.
	"""

	def __init__(self):
		pass

	def disambiguate(self, wsd_instance):
		"""
		Disambiguates the given instance, returning the predicted lemma sense
		key.

		@param wsd_instance - WSDInstance to disambiguate
		@return Sense key as a string if a sense could be found, None
			otherwise
		"""

		raise NotImplementedError

	def predict(self, wsd_instances):
		"""
		Predicts the sense for each instance in the given list according to,
		returning a dictionary mapping instance id's to predicted sense.

		@param wsd_instances - List of WSDInstance's to disambiguate
		@return Dictionary from WSDInstance id's to predicted sense (as sense
			key)
		"""

		return {wsd_id: self.disambiguate(wi) for wsd_id, wi
			in wsd_instances.iteritems()}


class BaselineWSDisambiguator(WSDisambiguator):
	"""
	Most frequent sense baseline.
	"""

	def __init__(self):
		pass

	def disambiguate(self, wsd_instance):
		"""
		Disambiguates the given instance, returning the predicted lemma sense
		key.

		@param wsd_instance - WSDInstance to disambiguate
		@return Sense key as a string if a sense could be found, None
			otherwise
		"""

		syns = wordnet.synsets(wsd_instance.lemma)

		if len(syns) > 0:
			# TODO: Verify actually counting the lemma frequencies give the same
			# results.
			return to_sense_key(syns[0])

		return None


class NLTKLeskWSDisambiguator(WSDisambiguator):
	"""
	NLTK's implementation of Lesk's algorithm.
	"""

	def __init__(self):
		pass

	def disambiguate(self, wsd_instance):
		"""
		Disambiguates the given instance, returning the predicted lemma sense
		key.

		@param wsd_instance - WSDInstance to disambiguate
		@return Sense key as a string if a sense could be found, None
			otherwise
		"""

		syn = lesk(wsd_instance.context, wsd_instance.lemma, 'n')
		if syn is not None:
			return to_sense_key(syn)
		
		return None

class MyLeskDisambiguator(WSDisambiguator):
	"""
	My implementation of Lesk's algorithm.
	"""

	def __init__(self, init_mfs=False, overlap_metric=intersection,
		incl_examples=False, rec_level=1):
		"""
		@param init_mfs - If True, the default synset returned is the most
			frequent one
		@param overlap_metric - Overlap metric to use between word context and
			signatures (Default is intersection)
		@param incl_examples - Boolean for whether or not to include examples in
			the signature (Default is False)
		@param rec_level - The recursive depth to use for synset signatures.
			(A depth of 1 (default) corresponds to using just the definition of
			the synset, a depth of 2 corresponds to using the definition and the
			defnition of words inside the defintion, and so on.)
		"""

		self.init_mfs = init_mfs
		self.overlap_metric = overlap_metric
		self.incl_examples = incl_examples
		self.rec_level = rec_level

	def disambiguate(self, wsd_instance):
		"""
		Disambiguates the given instance, returning the predicted lemma sense
		key.

		@param wsd_instance - WSDInstance to disambiguate
		@return Sense key as a string if a sense could be found, None
			otherwise
		"""

		syns = wordnet.synsets(wsd_instance.lemma)

		if len(syns) == 0:
			return None

		# Remove the query word from the context.
		context = wsd_instance.context[:wsd_instance.index] + \
			(wsd_instance.context[wsd_instance.index + 1:]
				if wsd_instance.index < len(wsd_instance.context) - 1 else
				[])

		# Get rid of any punctuation in the context.
		context = [l for l in context if l not in punctuation]

		tokenizer = RegexpTokenizer(r'[\w_-]+')
		lemmatizer = WordNetLemmatizer()
		stops = stopwords.words('english')

		# Computes the signature of a definition by tokenizing, stripping
		# punctuation and cardinals, lemmatizing, and removing stop words.
		def transform_def(definition):
			tokens = tokenizer.tokenize(definition)
			tokens = [t for t in tokens
				if t not in punctuation and t != '@card@']
			tokens = [lemmatizer.lemmatize(t) for t in tokens]
			return [t for t in tokens if t not in stops]

		mx_overlap, mx_syn = 0.0, syns[0] if self.init_mfs else None
		for syn in syns:
			signature = transform_def(syn.definition())

			if self.incl_examples:
				for ex in syn.examples():
					signature += transform_def(ex)

			for _ in range(self.rec_level - 1):
				for w in list(signature):
					w_syns = wordnet.synsets(w)
					if len(w_syns) > 0:
						signature += transform_def(w_syns[0].definition())

			overlap = self.overlap_metric(context, signature)
			if mx_overlap < overlap:
				mx_overlap, mx_syn = overlap, syn

		if mx_syn is None:
			return None

		return to_sense_key(mx_syn)


class Evaluator(object):
	"""
	Evaluates word sense disambiguators using the standard measures of
	precision, recall, and F1-score.
	"""

	def __init__(self):
		pass

	def cnt_tps(self, predict_key, gold_key):
		"""
		Counts the # TP's.

		@param predict_key - Predicted senses dictionary
		@param gold_key - Gold senses dictionary
		@return # TP's
		"""

		return len([wsd_id for wsd_id, sense in predict_key.iteritems()
			if wsd_id in gold_key and sense in gold_key[wsd_id]])

	def precision(self, predict_key, gold_key):
		"""
		Calculates precision.

		@param predict_key - Predicted senses dictionary
		@param gold_key - Gold senses dictionary
		@return Precision
		"""

		tp_cnt = self.cnt_tps(predict_key, gold_key)

		return float(tp_cnt) / \
			len([sense for _, sense in predict_key.iteritems()
				if sense is not None])

	def recall(self, predict_key, gold_key):
		"""
		Calculates recall.

		@param predict_key - Predicted senses dictionary
		@param gold_key - Gold senses dictionary
		@return Recall
		"""

		tp_cnt = self.cnt_tps(predict_key, gold_key)

		return float(tp_cnt) / len(gold_key)

	def f1(self, predict_key, gold_key):
		"""
		Calculates F1-score.

		@param predict_key - Predicted senses dictionary
		@param gold_key - Gold senses dictionary
		@return F1-score
		"""

		prec = self.precision(predict_key, gold_key)
		rec = self.recall(predict_key, gold_key)

		return 2 * prec * rec / (prec + rec)


def main():
	parser_description = ("Runs the MFS baseline, Lesk's algorithm and my "
		"variant of Lesk's algorithm on the dataset from the SemEval 2013 "
		"Shared Task.")
	parser = argparse.ArgumentParser(description=parser_description)

	parser.add_argument('out_path',
		help="Output file path for results (in .tsv format)")

	args = parser.parse_args()

	data = Data()
	evaluator = Evaluator()

	with open(args.out_path, 'wb') as f:
		writer = csv.writer(f, delimiter='\t', quotechar='"')

		# Write header.
		writer.writerow(['MODEL', 'DATASET', 'PRECISION', 'RECALL', 'F1'])

		# Evaluates the given model and writes it to file. Returns a triple with
		# the precision, recall, and F1-score (in that order).
		def evaluate_and_write(model, name, dataset):
			instances = data.dev_instances if dataset == 'dev' else \
				data.test_instances
			key = data.dev_key if dataset == 'dev' else data.test_key

			predict_key = model.predict(instances)

			prec = evaluator.precision(predict_key, key)
			rec = evaluator.recall(predict_key, key)
			f1 = evaluator.f1(predict_key, key)

			writer.writerow([
				name,
				dataset,
				"%0.4f" % prec,
				"%0.4f" % rec,
				"%0.4f" % f1
			])

			return (prec, rec, f1)

		## Evaluation on development set.

		baseline = BaselineWSDisambiguator()
		evaluate_and_write(baseline, "Baseline", 'dev')
		
		nltk_lesk = NLTKLeskWSDisambiguator()
		evaluate_and_write(nltk_lesk, "NLTK's Lesk algorithm", 'dev')
	
		# Returns the model name for an instantiation of my Lesk algorithm (i.e.
		# for a particular parameter setting).
		def get_my_lesk_name(my_lesk):
			if my_lesk.overlap_metric == intersection:
				overlap_name = 'intersection'
			elif my_lesk.overlap_metric == jaccards_index:
				overlap_name = 'jaccards_index'
			elif my_lesk.overlap_metric == lcs:
				overlap_name = 'lcs'

			return ("My Lesk algorithm (with parameters init_mfs=%s, "
				"overlap_metric=%s, incl_examples=%s, rec_level=%d)" %
				(str(my_lesk.init_mfs), overlap_name,
					str(my_lesk.incl_examples), my_lesk.rec_level))

		# Perform grid-search through parameter space for my algorithm,
		# outputting the results for each parameter setting to file (Takes a
		# while to run).
		best_score, best_model = 0.0, None
		for init_mfs, overlap_metric, incl_examples, rec_level in [
				(init_mfs, overlap_metric, incl_examples, rec_level)
				for init_mfs in [True, False]
				for overlap_metric in [intersection, jaccards_index, lcs]
				for incl_examples in [True, False]
				for rec_level in range(1, 6)
			]:

			my_lesk = MyLeskDisambiguator(init_mfs=init_mfs,
				overlap_metric=overlap_metric, incl_examples=incl_examples,
				rec_level=rec_level)
			name =  get_my_lesk_name(my_lesk)
			prec, rec, f1 = evaluate_and_write(my_lesk, name, 'dev')

			if best_score < f1:
				best_score, best_model = f1, my_lesk

		## Evaluation on test set.

		evaluate_and_write(baseline, "Baseline", 'test')
		evaluate_and_write(nltk_lesk, "NLTK's Lesk algorithm", 'test')
		evaluate_and_write(best_model, get_my_lesk_name(best_model), 'test')

		## Print some sample output using the best model.

		print "* Sample output from best model (%s) *" % \
			get_my_lesk_name(best_model)

		for _ in range(10):
			rand_key = random.choice(data.dev_instances.keys())
			rand_instance = data.dev_instances[rand_key]

			gold_keys = data.dev_key[rand_key] if rand_key in data.dev_key \
				else None

			print ""
			print rand_key
			print "Sentence (as lemmas): %s" % ' '.join(rand_instance.context)
			print "Target lemma: %s" % rand_instance.lemma
			print "Output sense key: %s" % \
				best_model.disambiguate(rand_instance)
			print "Gold sense keys: %s" % (', '.join(gold_keys) if gold_keys is
				not None else "(None)")


if __name__ == '__main__':
	main()
