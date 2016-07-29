import nltk
import A
from collections import defaultdict
from nltk.align import Alignment, AlignedSent

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
	self.t, self.q = self.train(align_sents, num_iter)
	
    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
#	#will return german --> english alignments
 	alignments = []
        german = align_sent.words
        english = align_sent.mots
        len_g = len(german)
        len_e = len(english)

        for j in range(len_g):
		g = german[j]
		best_prob = (self.t[(g,None)] * self.q[(0,j,len_e,len_g)], None)
		best_alignment_point = None
		for i in range(len_e):
                	e = english[i]
 	 		ge_prob = (self.t[(e,g)]*self.q[(j,i,len_g,len_e)], i)
			eg_prob = (self.t[(g,e)]*self.q[(i,j,len_e,len_g)], i)
			best_prob = max(best_prob, ge_prob, eg_prob)
			
		alignments.append((j, best_prob[1]))

	return AlignedSent(align_sent.words, align_sent.mots, alignments)

  
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
	MIN_PROB = 1.0e-12
	#INITIALIZATION
	#defining the vocabulary for each language:
	#german = words
	#english = mots
	g_vocab = set()
	e_vocab = set()
	for sentence in aligned_sents:
		g_vocab.update(sentence.words)
		e_vocab.update(sentence.mots)

	# initializing translation table for english --> german and german --> english
	t = defaultdict(float)
	for g in g_vocab:
		for e in e_vocab:
			t[(g,e)] = 1.0 / float(len(g_vocab))
			t[(e,g)] = 1.0 / float(len(e_vocab))
	
	# initializing separate alignment tables for english --> german and german --> english
	q_eg = defaultdict(float)
	q_ge = defaultdict(float)
	for sentence in aligned_sents:
		len_e=len(sentence.mots)
		len_g=len(sentence.words)
		for i in range(len_e):
			for j in range(len_g):
				q_eg[(i,j,len_e,len_g)] = 1.0 / float((len_e+1))
				q_ge[(j,i,len_g,len_e)] = 1.0 / float((len_g+1))

	print 'Initialization complete'
	#INITIALIZATION COMPLETE

	for i in range(num_iters):
		print 'Iteration ' + str(i+1) + ' /' + str(num_iters)
		#E step
		count_g_given_e = defaultdict(float)
		count_any_g_given_e = defaultdict(float)
		eg_alignment_count = defaultdict(float)
		eg_alignment_count_for_any_i = defaultdict(float)
		count_e_given_g = defaultdict(float)
		count_any_e_given_g = defaultdict(float)
		ge_alignment_count = defaultdict(float)
		ge_alignment_count_for_any_j = defaultdict(float)
		
		for sentence in aligned_sents:
			g_sentence = sentence.words
			e_sentence = sentence.mots
			len_e = len(sentence.mots)
			len_g = len(sentence.words)
			eg_total = defaultdict(float)
			ge_total = defaultdict(float)

			#E step (a): compute normalization
			for j in range(len_g):
				
				g = g_sentence[j]
	
				for i in range(len_e):
					
					e = e_sentence[i]

					eg_count = (t[(g_sentence[j],e_sentence[i])] * q_eg[(i,j,len_e,len_g)])
					eg_total[g] += eg_count

					ge_count = (t[(e_sentence[i], g_sentence[j])] * q_ge[(j,i,len_g,len_e)])
					ge_total[e] += ge_count 

			# E step (b): collect fractional counts
			for j in range(len_g):
				
				g = g_sentence[j]

				for i in range(len_e):
					
					e = e_sentence[i]
	
					#English --> German
					eg_count = (t[(g_sentence[j],e_sentence[i])] * q_eg[(i,j,len_e,len_g)])
					eg_normalized = eg_count / eg_total[g]

					#German --> English
					ge_count = (t[(e_sentence[i], g_sentence[j])] * q_ge[(j,i,len_g,len_e)])
					ge_normalized = ge_count / ge_total[e]

					#Averaging the probablities
					avg_normalized = (eg_normalized + ge_normalized) / 2.0
					#Storing counts
					count_g_given_e[(g,e)] += avg_normalized
					count_any_g_given_e[e] += avg_normalized
					eg_alignment_count[(i,j,len_e,len_g)] += avg_normalized
					eg_alignment_count_for_any_i[(j,len_e,len_g)] += avg_normalized
					count_e_given_g[(e,g)] += avg_normalized
					count_any_e_given_g[g] += avg_normalized
					ge_alignment_count[(j,i,len_g,len_e)] += avg_normalized
					ge_alignment_count_for_any_j[(i,len_g,len_e)] += avg_normalized

		#M step
		q = defaultdict(float)
		for sentence in aligned_sents:
			for e in sentence.mots:
				for g in sentence.words:
					#eng --> germ
					t[(g,e)]= count_g_given_e[(g,e)] / count_any_g_given_e[e]
					#germ --> eng
					t[(e,g)]= count_e_given_g[(e,g)] / count_any_e_given_g[g]

			len_e=len(sentence.mots)
			len_g=len(sentence.words)
			for i in range(len_e):
				for j in range(len_g):
					#eng --> germ
					q[(i,j,len_e,len_g)] = eg_alignment_count[(i,j,len_e,len_g)] / eg_alignment_count_for_any_i[(j,len_e, len_g)]
					#germ --> eng
					q[(j,i,len_g,len_e)] = ge_alignment_count[(j,i,len_g,len_e)] / ge_alignment_count_for_any_j[(i,len_g,len_e)]
	return (t,q)


def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
