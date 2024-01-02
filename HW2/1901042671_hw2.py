from nltk import ngrams, FreqDist, SimpleGoodTuringProbDist
from turkishnlp import detector
import random
import math

####################################
# FILE AND SYLLABICATE OPERATIONS  #
####################################

# file operations
main_file = 'wiki_00.txt'
syllables_seperated = 'seperated.txt'
one_gram_seperated = 'seperated_onegram.txt'
two_gram_seperated = 'seperated_twogram.txt'
three_gram_seperated = 'seperated_threegram.txt'
file = open(main_file, 'r', encoding='utf-8')
file_seperated = open(syllables_seperated, 'w', encoding='utf-8')
file_onegram = open(one_gram_seperated, 'w', encoding='utf-8')
file_twogram = open(two_gram_seperated, 'w', encoding='utf-8')
file_threegram = open(three_gram_seperated, 'w', encoding='utf-8')

# TurkishNlp object
obj = detector.TurkishNLP()

# significant line size
initial_line = 100_000 
line_size = 270_000

# get lines from wiki_00.txt
lines = file.readlines()

# avoiding numbers, punctiations, blanks and other symbols
def purifyLines(line: str):
    return ''.join(char for char in line if char.isalpha() or char == ' ')

# ignore non-word typos
def ignoreDocs(line: str):
    if line.startswith('<doc') or line.startswith('</doc'):
        return[]
    return [syllable for syllable in obj.syllabicate(purifyLines(line))]

# seperating syllables of first n lines
syllables = [syllables for line in lines[initial_line:initial_line + line_size] for syllables in ignoreDocs(line)]

# make an .txt file that contains seperated words
file_seperated.write('\n'.join(syllables))

# avoiding '' character for ngram operations
syllables = [syllables for line in lines[initial_line:initial_line + line_size] for syllables in ignoreDocs(line) if syllables != '']

#############################################################
# OBTAINING N-GRAM TABLES AND APPLYING SMOOTHING OPERATIONS #
#############################################################

# finding 1, 2 and 3 ngrams of syllables
one_gram = list(ngrams(syllables, 1))
two_gram = list(ngrams(syllables, 2))
three_gram = list(ngrams(syllables, 3))

# finding distributions of ngrams
distribution_one = FreqDist(one_gram)
print("one gram done")
distribution_two = FreqDist(two_gram)
print("two gram done")
distribution_three = FreqDist(three_gram)
print("three gram done")

print("\nmost common 5 syllables for one-gram: ", distribution_one.most_common(5))
print("most common 5 syllables for two-gram: ", distribution_two.most_common(5))
print("most common 5 syllables for three-gram: ", distribution_three.most_common(5))

file_onegram.write('\n'.join(['{}: {}'.format(item[0], item[1]) for item in distribution_one.items()]))
file_twogram.write('\n'.join(['{}: {}'.format(item[0], item[1]) for item in distribution_two.items()]))
file_threegram.write('\n'.join(['{}: {}'.format(item[0], item[1]) for item in distribution_three.items()]))

# implementing Good Turing smoothing each n-gram distribution
smoothed_one = SimpleGoodTuringProbDist(distribution_one)
print("\none smoothed one-gram done")
smoothed_two = SimpleGoodTuringProbDist(distribution_two)
print("two smoothed two-gram done")
smoothed_three = SimpleGoodTuringProbDist(distribution_three)
print("three smoothed three-gram done")

#######################################################
# CALCULATING PERPLEXITY WITH GIVEN EXAMPLE SENTENCE  #
#######################################################

def calculate_perplexity(test_set, ngram_model, distribution, smoothed_model):
    N = len(test_set)
    log_sum = 0.0

    for i in range(len(test_set) - ngram_model + 1):
        ngram = tuple(test_set[i:i + ngram_model])
        probability = smoothed_model.prob(ngram) if ngram in distribution else smoothed_model.prob(('<UNK>',))
        log_sum += math.log(probability)

    log_perplexity = -log_sum / N
    perplexity = math.exp(log_perplexity)
    return perplexity

example_text = "cengiz han buhara da iki gün kaldıktan sonra sonra semerkant a ilerledi"
test_tokens = example_text.split()
print("\nsentence: ", example_text)
perplexity_1gram = calculate_perplexity(test_tokens, 1, distribution_one, smoothed_one)
print(f"Perplexity for 1-gram model: {perplexity_1gram}")
perplexity_2gram = calculate_perplexity(test_tokens, 2, distribution_two, smoothed_two)
print(f"Perplexity for 2-gram model: {perplexity_2gram}")
perplexity_3gram = calculate_perplexity(test_tokens, 3, distribution_three, smoothed_three)
print(f"Perplexity for 3-gram model: {perplexity_3gram}")


#################################################
# GENERATING SENTENCES WITH USING N-GRAM TABLES #
#################################################

# Choose the top 5 syllables based on their distribution rates
top_syllables = [item[0] for item in distribution_one.most_common(5)]
print("\ntop 5 one-gram syllables: ", top_syllables)

# Function to generate a sentence with one syllable from the top 5 and nine additional syllables
def generateSentenceForOneGram(top_syllables, distribution, num_sentences=10):
    sentences = []
    for _ in range(num_sentences):        
        # Generate nine more syllables
        syllables = tuple(j for i in (random.choice(top_syllables) for _ in range(15)) for j in i)
        # Combine to form a sentence
        sentence = ''.join(syllables)
        sentences.append(sentence)
    
    return sentences

# Generate 10 sentences
sentences = generateSentenceForOneGram(top_syllables, distribution_one, num_sentences=10)

# Print the sentences
print("\n10 one-gram sentences")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

# Choose the top 5 syllables based on their distribution rates
top_syllables = [item[0] for item in distribution_two.most_common(5)]
print("\ntop 5 two-gram syllables: ", top_syllables)

def getSentencesForTwoGram(top_syllables):
    syllables = []
    first_syllable = random.choice(top_syllables)
    syllable = str(first_syllable[0])
    syllables.append(syllable)
    for i in range (14):
        most_common_five_syllables = [(first, second) for (first, second), count in distribution_two.items() if first == syllable]
        most_common_syllable = random.choice(most_common_five_syllables)
        syllable = str(most_common_syllable[1])
        syllables.append(syllable)
    sentence = ''.join(syllables)
    return sentence

# Function to generate a sentence with two syllables from the top 5 and nine additional syllables
def generateSentenceForTwoGram(top_syllables, num_sentences=10):
    sentences = []
    for _ in range(num_sentences):        
        sentence = getSentencesForTwoGram(top_syllables)
        sentences.append(sentence)
    return sentences

# Generate 10 sentences
sentences = generateSentenceForTwoGram(top_syllables, num_sentences=10)
print("\n10 two-gram sentences")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")


# Choose the top 5 syllables based on their distribution rates
top_syllables = [item[0] for item in distribution_three.most_common(5)]
print("\ntop 5 three-gram syllables: ", top_syllables)

def getSentencesForThreeGram(top_syllables):
    syllables = []
    first_syllable = random.choice(top_syllables)
    syllable = str(first_syllable[0])
    syllables.append(syllable)
    syllable1 = str(first_syllable[1])
    syllables.append(syllable1)
    for i in range (13):
        most_common_five_syllables = [(first, second, third) for (first, second, third), count in distribution_three.items() if first == syllable and second == syllable1]
        most_common_syllable = random.choice(most_common_five_syllables)
        syllable = str(most_common_syllable[1])
        syllable1 = str(most_common_syllable[2])
        syllables.append(syllable1)
    sentence = ''.join(syllables)
    return sentence

# Function to generate a sentence with three syllables from the top 5 and nine additional syllables
def generateSentenceForThreeGram(top_syllables, num_sentences=10):
    sentences = []
    for _ in range(num_sentences):        
        sentence = getSentencesForThreeGram(top_syllables)
        sentences.append(sentence)
    return sentences

# Generate 10 sentences
sentences = generateSentenceForThreeGram(top_syllables, num_sentences=10)
print("\n10 three-gram sentences")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")    



        


