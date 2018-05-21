package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;

public class GibbsSamplingDMM_Inf_Single {

	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter alpha
	public int numTopics; // Number of topics
	public int numIterations; // Number of Gibbs sampling iterations
	public int topWords; // Number of most probable words for each topic

	public double alphaSum; // alpha * numTopics
	public double betaSum; // beta * vocabularySize

	public List<List<Integer>> corpus; // Word ID-based corpus
	public List<Integer> topicAssignments; // Topics assignments for documents
	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
														// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
														// given an ID
	public int vocabularySize; // The number of word types in the corpus

	// Number of documents assigned to a topic
	public int[] docTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type assigned to the topic
	public int[][] topicWordCount;
	// Total number of words assigned to a topic
	public int[] sumTopicWordCount;

	// Double array used to sample a topic
	public double[] multiPros;

	// Path to the directory containing the corpus
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;

	// Given a document, number of times its i^{th} word appearing from
	// the first index to the i^{th}-index in the document
	// Example: given a document of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
	public List<List<Integer>> occurenceToIndexCount;

	public String expName = "DMMinf";
	public String orgExpName = "DMMinf";
	public String tAssignsFilePath = "";
	public int savestep = 0;

	public GibbsSamplingDMM_Inf_Single(String pathToTrainingParasFile,
		String pathToUnseenCorpus, int inNumIterations, int inTopWords,
		String inExpName, int inSaveStep)
		throws Exception
	{
		HashMap<String, String> paras = parseTrainingParasFile(pathToTrainingParasFile);
		if (!paras.get("-model").equals("DMM")) {
			throw new Exception("Wrong pre-trained model!!!");
		}
		alpha = new Double(paras.get("-alpha"));
		beta = new Double(paras.get("-beta"));
		numTopics = new Integer(paras.get("-ntopics"));

		numIterations = inNumIterations;
		topWords = inTopWords;
		savestep = inSaveStep;
		expName = inExpName;
		orgExpName = expName;

		String trainingCorpus = paras.get("-corpus");
		String trainingCorpusfolder = trainingCorpus.substring(
			0,
			Math.max(trainingCorpus.lastIndexOf("/"),
				trainingCorpus.lastIndexOf("\\")) + 1);
		String topicAssignment4TrainFile = trainingCorpusfolder
			+ paras.get("-name") + ".topicAssignments";

		word2IdVocabulary = new HashMap<String, Integer>();
		id2WordVocabulary = new HashMap<Integer, String>();
		initializeWordCount(trainingCorpus, topicAssignment4TrainFile);

		corpusPath = pathToUnseenCorpus;
		folderPath = pathToUnseenCorpus.substring(
			0,
			Math.max(pathToUnseenCorpus.lastIndexOf("/"),
				pathToUnseenCorpus.lastIndexOf("\\")) + 1);
		System.out.println("Reading unseen corpus: " + pathToUnseenCorpus);
		corpus = new ArrayList<List<Integer>>();
		occurenceToIndexCount = new ArrayList<List<Integer>>();
		numDocuments = 0;
		numWordsInCorpus = 0;

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToUnseenCorpus));
			for (String doc; (doc = br.readLine()) != null;) {
				if (doc.trim().length() == 0)
					continue;

				String[] words = doc.trim().split("\\s+");
				List<Integer> document = new ArrayList<Integer>();

				List<Integer> wordOccurenceToIndexInDoc = new ArrayList<Integer>();
				HashMap<String, Integer> wordOccurenceToIndexInDocCount = new HashMap<String, Integer>();

				for (String word : words) {
					if (word2IdVocabulary.containsKey(word)) {
						document.add(word2IdVocabulary.get(word));
						int times = 0;
						if (wordOccurenceToIndexInDocCount.containsKey(word)) {
							times = wordOccurenceToIndexInDocCount.get(word);
						}
						times += 1;
						wordOccurenceToIndexInDocCount.put(word, times);
						wordOccurenceToIndexInDoc.add(times);
					}
					else {
						// Skip this unknown-word
					}
				}
				numDocuments++;
				numWordsInCorpus += document.size();
				corpus.add(document);
				occurenceToIndexCount.add(wordOccurenceToIndexInDoc);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

		docTopicCount = new int[numTopics];
		multiPros = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			multiPros[i] = 1.0 / numTopics;
		}

		alphaSum = numTopics * alpha;
		betaSum = vocabularySize * beta;

		System.out.println("Corpus size: " + numDocuments + " docs, "
			+ numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + vocabularySize);
		System.out.println("Number of topics: " + numTopics);
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("Number of sampling iterations: " + numIterations);
		System.out.println("Number of top topical words: " + topWords);

		initialize();
	}

	private HashMap<String, String> parseTrainingParasFile(
		String pathToTrainingParasFile)
		throws Exception
	{
		HashMap<String, String> paras = new HashMap<String, String>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToTrainingParasFile));
			for (String line; (line = br.readLine()) != null;) {

				if (line.trim().length() == 0)
					continue;

				String[] paraOptions = line.trim().split("\\s+");
				paras.put(paraOptions[0], paraOptions[1]);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return paras;
	}

	private void initializeWordCount(String pathToTrainingCorpus,
		String pathToTopicAssignmentFile)
	{
		System.out.println("Loading pre-trained model...");
		List<List<Integer>> trainCorpus = new ArrayList<List<Integer>>();
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToTrainingCorpus));
			for (String doc; (doc = br.readLine()) != null;) {

				if (doc.trim().length() == 0)
					continue;

				String[] words = doc.trim().split("\\s+");
				List<Integer> document = new ArrayList<Integer>();

				for (String word : words) {
					if (word2IdVocabulary.containsKey(word)) {
						document.add(word2IdVocabulary.get(word));
					}
					else {
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						document.add(indexWord);
					}
				}
				trainCorpus.add(document);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

		vocabularySize = word2IdVocabulary.size();
		topicWordCount = new int[numTopics][vocabularySize];
		sumTopicWordCount = new int[numTopics];

		try {
			br = new BufferedReader(new FileReader(pathToTopicAssignmentFile));
			int docId = 0;
			for (String line; (line = br.readLine()) != null;) {
				String[] strTopics = line.trim().split("\\s+");
				for (int j = 0; j < strTopics.length; j++) {
					int wordId = trainCorpus.get(docId).get(j);
					int topic = new Integer(strTopics[j]);
					topicWordCount[topic][wordId] += 1;
					sumTopicWordCount[topic] += 1;
				}
				docId++;
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void initialize() 
	{
		topicAssignments = new ArrayList<Integer>();
	}
	
	public void sampleInSingleIteration(int dIndex)
	{
		int topic = topicAssignments.get(0);
		List<Integer> document = corpus.get(dIndex);
		int docSize = document.size();

		// Decrease counts
		docTopicCount[topic] -= 1;
		for (int wIndex = 0; wIndex < docSize; wIndex++) {
			int word = document.get(wIndex);
			topicWordCount[topic][word] -= 1;
			sumTopicWordCount[topic] -= 1;
		}

		// Sample a topic
		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = document.get(wIndex);
				multiPros[tIndex] *= (topicWordCount[tIndex][word] + beta
					+ occurenceToIndexCount.get(dIndex).get(wIndex) - 1)
					/ (sumTopicWordCount[tIndex] + betaSum + wIndex);
			}
		}
		topic = FuncUtils.nextDiscrete(multiPros);

		// Increase counts
		docTopicCount[topic] += 1;
		for (int wIndex = 0; wIndex < docSize; wIndex++) {
			int word = document.get(wIndex);
			topicWordCount[topic][word] += 1;
			sumTopicWordCount[topic] += 1;
		}
		// Update topic assignments
		topicAssignments.set(0, topic);
	}

	public void writeParameters()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".paras"));
		writer.write("-model" + "\t" + "DMM");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-ntopics" + "\t" + numTopics);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta" + "\t" + beta);
		writer.write("\n-niters" + "\t" + numIterations);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);
		if (tAssignsFilePath.length() > 0)
			writer.write("\n-initFile" + "\t" + tAssignsFilePath);
		if (savestep > 0)
			writer.write("\n-sstep" + "\t" + savestep);

		writer.close();
	}

	public void writeDictionary()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".vocabulary"));
		for (int id = 0; id < vocabularySize; id++)
			writer.write(id2WordVocabulary.get(id) + " " + id + "\n");
		writer.close();
	}

	public void writeIDbasedCorpus()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".IDcorpus"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(corpus.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicAssignments(int dIndex)
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topicAssignments"));
		int docSize = corpus.get(dIndex).size();
		int topic = topicAssignments.get(0);
		for (int wIndex = 0; wIndex < docSize; wIndex++) {
			writer.write(topic + " ");
		}
		writer.write("\n");
		writer.close();
	}

	public void writeTopTopicalWords()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topWords"));

		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
				wordCount.put(wIndex, topicWordCount[tIndex][wIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);

			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					double pro = (topicWordCount[tIndex][index] + beta)
						/ (sumTopicWordCount[tIndex] + betaSum);
					pro = Math.round(pro * 1000000.0) / 1000000.0;
					writer.write(" " + id2WordVocabulary.get(index) + "(" + pro
						+ ")");
					count += 1;
				}
				else {
					writer.write("\n\n");
					break;
				}
			}
		}
		writer.close();
	}

	public void writeTopicWordPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".phi"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				double pro = (topicWordCount[i][j] + beta)
					/ (sumTopicWordCount[i] + betaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicWordCount()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".WTcount"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				writer.write(topicWordCount[i][j] + " ");
			}
			writer.write("\n");
		}
		writer.close();

	}

	
	public void writeDocTopicPros(int dIndex)
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".theta"));

		int i = dIndex;
		int docSize = corpus.get(i).size();
		double sum = 0.0;
		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = corpus.get(i).get(wIndex);
				multiPros[tIndex] *= (topicWordCount[tIndex][word] + beta)
					/ (sumTopicWordCount[tIndex] + betaSum);
			}
			sum += multiPros[tIndex];
		}
		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write((multiPros[tIndex] / sum) + " ");
		}
		writer.write("\n");
		writer.close();
	}

	/**
	 * Log useful information when inference single document
	 * @param dIndex : test document id
	 */
	public void write(int dIndex)
		throws IOException
	{
		writeTopTopicalWords();
		writeDocTopicPros(dIndex);
		writeTopicAssignments(dIndex);
		writeTopicWordPros();
	}
	
	/**
	 * Randomly initialize topic assignment for current test document, update related word count variables.
	 * @param dIndex : test document index
	 */
	public void initializeSingleDocTopic(int dIndex)
		throws IOException
	{
		System.out.println("Randomly initialzing topic assignment for document " + dIndex);
		int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
		docTopicCount[topic] += 1;
		int docSize = corpus.get(dIndex).size();
		for (int j = 0; j < docSize; j++) {
			topicWordCount[topic][corpus.get(dIndex).get(j)] += 1;
			sumTopicWordCount[topic] += 1;
		}
		topicAssignments.set(0, topic);
	}

	/**
	 * clean count for document dIndex
	 * @param dIndex : document id
	 */
	public void cleanSingleDocCount(int dIndex)
	{
		int topic = topicAssignments.get(0);
		docTopicCount[topic] -= 1;
		int docSize = corpus.get(dIndex).size();
		for (int j = 0; j < docSize; j++) {
			topicWordCount[topic][corpus.get(dIndex).get(j)] -= 1;
			sumTopicWordCount[topic] -= 1;
		}
	}
	

	/**
	 * inference topic for single document
	 * 
	 * @param dIndex : document index
	 */
	public void inferenceSingleDoc(int dIndex)
			throws IOException
	{
		initializeSingleDocTopic(dIndex);

		System.out.println("[document " + dIndex + "] Running Gibbs sampling inference : ");

		for (int iter = 1; iter <= numIterations; iter++) {

			System.out.println("\tSampling iteration: " + (iter));
			// System.out.println("\t\tPerplexity: " + computePerplexity());

			sampleInSingleIteration(dIndex);

			if ((savestep > 0) && (iter % savestep == 0)
				&& (iter < numIterations)) {
				System.out.println("\t\tSaving the output from the " + iter
					+ "^{th} sample");
				
				// modify log file prefix
				expName = orgExpName + "doc-" + dIndex + "-" + iter;
				write(dIndex);
			}
		}
		
		// modify log file prefix and write logs
		expName = orgExpName + "doc-" + dIndex;
		System.out.println("[document " + dIndex + "] Writing output from the last sample ...");
		write(dIndex);

		// recover log file prefix
		expName = orgExpName;
		
		// clean count
		cleanSingleDocCount(dIndex);
		System.out.println("[document " + dIndex + "] Sampling completed!");
	}
	
	/**
	 * inference all topics for all documents
	 */
	public void inferenceAllDoc() throws IOException
	{
		writeParameters();
		writeDictionary();

		for (int dIndex = 0; dIndex < numDocuments; dIndex++) 
		{
			inferenceSingleDoc(dIndex);
		}
	}

	
	/**
	 * Main function for test purpose
	 */
	public static void main(String args[])
		throws Exception
	{
		GibbsSamplingDMM_Inf_Single dmm = new GibbsSamplingDMM_Inf_Single(
			"test/testDMM.paras", "test/unseenTest.txt", 100, 20, "testDMMinf",
			0);

		dmm.inferenceAllDoc();
	}
}
